"""DM-VTON wrapper for the LIVE mode WebSocket.

DM-VTON synthesises a full image of a person wearing the supplied garment,
but the result is the model's IDEA of a person — not the actual customer.
To make the kiosk feel like a real mirror (the customer sees themselves +
the dress draped on their body), we use the AI render only inside a
pose-detected torso polygon, and let the live webcam show through
everywhere else (face, arms, hands, background).

Per-frame pipeline:
    1. Detect 33 body keypoints with MediaPipe Pose (~5-10 ms)
    2. Run DM-VTON at its native 256×192 (~80-150 ms on RTX 4080)
    3. Upscale the AI render back to the frame's original resolution
    4. Build a soft torso polygon mask from shoulder + hip landmarks,
       extended outward and downward (so dresses cover legs too)
    5. Alpha-composite: AI render inside the polygon, webcam outside
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image

log = logging.getLogger("dm-vton")


# How far below the hips the torso polygon extends, as a fraction of the
# shoulder→hip distance. Longer garments need more bottom margin.
HEM_EXTEND: dict[str, float] = {
    "top":    0.30,
    "bottom": 1.80,
    "dress":  1.80,
}
COLLAR_RISE: dict[str, float] = {
    "top":    0.15,
    "bottom": -0.95,
    "dress":  0.18,
}
SIDE_PAD = 0.25  # extra width on each side, as fraction of shoulder span


@dataclass
class DMVTONPipe:
    pipeline: Any                # DMVTONPipeline instance
    device: str
    tensor_dtype: Any            # torch.float32
    pose: Any = None             # MediaPipe Pose
    _last_polygon: Any = field(default=None)  # cache last good polygon for missing-pose frames

    def warp(
        self,
        frame: Image.Image,
        garment: Image.Image,
        category: Literal["top", "bottom", "dress"] = "dress",
    ) -> Image.Image:
        import torch
        from torchvision import transforms
        import cv2

        H_MODEL, W_MODEL = 256, 192   # DM-VTON's native resolution
        orig_w, orig_h = frame.size

        # ── 1. Pose detection on the FULL-RES frame ─────────────────
        frame_full_rgb = frame.convert("RGB")
        frame_full = np.array(frame_full_rgb)
        pose_results = self.pose.process(frame_full)

        # ── 2. DM-VTON inference at the model's native res ──────────
        frame_small = frame_full_rgb.resize((W_MODEL, H_MODEL))
        garment_rgba = garment.convert("RGBA").resize((W_MODEL, H_MODEL))
        alpha = np.array(garment_rgba.split()[3])
        edge = (alpha > 128).astype(np.float32)

        norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * 3, (0.5,) * 3),
        ])
        person_t  = norm(frame_small).unsqueeze(0).to(self.device, self.tensor_dtype)
        clothes_t = norm(garment_rgba.convert("RGB")).unsqueeze(0).to(self.device, self.tensor_dtype)
        edge_t    = torch.from_numpy(edge)[None, None, ...].to(self.device, self.tensor_dtype)

        with torch.no_grad():
            p_tryon, _warped = self.pipeline(person_t, clothes_t, edge_t, phase="test")

        ai_small = ((p_tryon.clamp(-1, 1) + 1) / 2)
        ai_small = ai_small.squeeze(0).permute(1, 2, 0).float().cpu().numpy()
        ai_small = (ai_small * 255).astype(np.uint8)

        # ── 3. Upscale AI render to full resolution ─────────────────
        ai_full = np.array(
            Image.fromarray(ai_small).resize((orig_w, orig_h), Image.LANCZOS)
        )

        # ── 4. Build torso polygon mask from pose landmarks ────────
        polygon = self._build_torso_polygon(pose_results, category, orig_w, orig_h)
        if polygon is None:
            # No body in frame → just return the live webcam
            return frame_full_rgb

        mask = np.zeros((orig_h, orig_w), dtype=np.float32)
        cv2.fillPoly(mask, [polygon], 1.0)
        # Soft edge so the dress blends into the webcam silhouette instead of
        # a hard polygon outline
        blur = int(min(orig_w, orig_h) * 0.05) | 1   # odd kernel
        mask = cv2.GaussianBlur(mask, (blur, blur), 0)
        mask = mask[..., None]   # (H, W, 1) for broadcast

        # ── 5. Composite: AI inside the polygon, webcam outside ────
        composited = ai_full * mask + frame_full * (1 - mask)
        return Image.fromarray(composited.astype(np.uint8))

    def _build_torso_polygon(self, pose_results, category: str, w: int, h: int):
        """Return a 4-vertex int32 polygon outlining where the garment goes,
        in (x, y) pixel coords, or None if no usable pose is detected.
        Caches the last good polygon so transient pose-loss frames keep the
        garment painted instead of flicking off."""
        if not pose_results.pose_landmarks:
            return self._last_polygon  # may be None on the first frame
        lms = pose_results.pose_landmarks.landmark
        # MediaPipe Pose landmark indices:
        #   11 = left shoulder, 12 = right shoulder
        #   23 = left hip,      24 = right hip
        def pt(i):
            return (int(lms[i].x * w), int(lms[i].y * h))
        def vis(i): return lms[i].visibility

        if min(vis(11), vis(12), vis(23), vis(24)) < 0.4:
            return self._last_polygon

        lsh, rsh = pt(11), pt(12)
        lhp, rhp = pt(23), pt(24)

        shoulder_span = abs(lsh[0] - rsh[0])
        torso_h = abs((lhp[1] + rhp[1]) / 2 - (lsh[1] + rsh[1]) / 2)
        side_pad   = int(shoulder_span * SIDE_PAD)
        collar_dy  = int(torso_h * COLLAR_RISE.get(category, 0.18))
        hem_dy     = int(torso_h * HEM_EXTEND.get(category, 1.80))

        # Order matters for fillPoly — go around the polygon (CW or CCW).
        # MediaPipe's "left" shoulder is on the SUBJECT's left, which appears
        # on the screen RIGHT after the browser mirrors the selfie.
        polygon = np.array([
            [lsh[0] + side_pad, lsh[1] - collar_dy],  # subject's left shoulder, outset
            [rsh[0] - side_pad, rsh[1] - collar_dy],  # subject's right shoulder, outset
            [rhp[0] - side_pad, rhp[1] + hem_dy],     # below subject's right hip
            [lhp[0] + side_pad, lhp[1] + hem_dy],     # below subject's left hip
        ], dtype=np.int32)
        self._last_polygon = polygon
        return polygon


def load_dm_vton_trt(path: Path) -> DMVTONPipe:
    """Load DM-VTON's `DMVTONPipeline` from the cloned ../DM-VTON/ repo +
    initialise a single-person MediaPipe Pose for torso localisation."""
    import torch
    import mediapipe as mp

    repo_root = Path(__file__).resolve().parent / "DM-VTON"
    if not (repo_root / "pipelines").exists():
        raise FileNotFoundError(
            f"DM-VTON repo not found at {repo_root}. Run scripts/setup_dm_vton.sh first."
        )
    sys.path.insert(0, str(repo_root))

    # Match by substring — handle whatever naming the Drive variant uses
    def _find(needle: str) -> Path:
        candidates = sorted(path.glob(f"*{needle}*.pt")) + sorted(path.glob(f"*{needle}*.pth"))
        if not candidates:
            raise FileNotFoundError(
                f"No *{needle}*.pt or *{needle}*.pth in {path}. Download from "
                f"https://drive.google.com/drive/folders/1wfWGsR0vWC5LrA26xhj92ec_GoCKV80A"
            )
        return candidates[0]

    warp_ckpt = _find("warp")
    gen_ckpt  = _find("gen")
    log.info("found warp checkpoint: %s", warp_ckpt.name)
    log.info("found gen checkpoint:  %s", gen_ckpt.name)

    from pipelines import DMVTONPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # FP32 — partial .half() left dtype mismatches inside DM-VTON's submodules
    dtype = torch.float32

    log.info("instantiating DMVTONPipeline…")
    pipeline = DMVTONPipeline(
        align_corners=True,
        checkpoints={"warp": str(warp_ckpt), "gen": str(gen_ckpt)},
    ).to(device).eval()

    log.info("loading MediaPipe Pose…")
    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    log.info("DM-VTON pipeline ready (PyTorch %s on %s, with MediaPipe Pose)", dtype, device)
    return DMVTONPipe(pipeline=pipeline, device=device, tensor_dtype=dtype, pose=pose)
