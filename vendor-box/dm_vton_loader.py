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
    "top":    0.40,
    "bottom": 2.20,
    "dress":  2.30,   # extends below knees for proper dress length
}
COLLAR_RISE: dict[str, float] = {
    "top":    0.22,
    "bottom": -0.95,
    "dress":  0.28,   # collar a bit higher so the dress looks worn, not stuck on
}
# Extra width on each side, as fraction of shoulder span. Bigger = the dress
# extends further past your shoulders (more flowy / draped look).
SIDE_PAD: dict[str, float] = {
    "top":    0.30,
    "bottom": 0.30,
    "dress":  0.40,
}

# Polygon smoothing — how fast the polygon catches up to a new pose detection.
# 1.0 = no smoothing (snaps each frame), 0.3 = lazy & smooth, 0.6 = balanced.
POLYGON_SMOOTH_ALPHA = 0.55


@dataclass
class DMVTONPipe:
    pipeline: Any                # DMVTONPipeline instance
    device: str
    tensor_dtype: Any            # torch.float32
    pose: Any = None             # MediaPipe PoseLandmarker (Tasks API)
    segmenter: Any = None        # MediaPipe ImageSegmenter (selfie body mask)
    _last_polygon: Any = field(default=None)  # cache last good polygon
    _smoothed_polygon: Any = field(default=None)  # EMA-smoothed polygon for jitter reduction

    def warp(
        self,
        frame: Image.Image,
        garment: Image.Image,
        category: Literal["top", "bottom", "dress"] = "dress",
    ) -> Image.Image:
        import torch
        from torchvision import transforms
        import cv2
        import mediapipe as mp

        H_MODEL, W_MODEL = 256, 192   # DM-VTON's native resolution
        orig_w, orig_h = frame.size

        # ── 1. Pose + body segmentation on the FULL-RES frame ─────────
        frame_full_rgb = frame.convert("RGB")
        frame_full = np.array(frame_full_rgb)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_full)
        pose_results = self.pose.detect(mp_image)
        # Selfie segmentation: per-pixel mask of the person silhouette.
        # The segmenter returns a list of category masks; channel 0 is the
        # person-vs-background mask (1.0 inside the body).
        seg_results = self.segmenter.segment(mp_image)
        try:
            body_mask = seg_results.confidence_masks[0].numpy_view().astype(np.float32)
        except (AttributeError, IndexError):
            body_mask = None

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

        # EMA-smooth the polygon vertices to reduce frame-to-frame jitter
        if self._smoothed_polygon is None:
            self._smoothed_polygon = polygon.astype(np.float32)
        else:
            self._smoothed_polygon = (
                POLYGON_SMOOTH_ALPHA * polygon.astype(np.float32)
                + (1 - POLYGON_SMOOTH_ALPHA) * self._smoothed_polygon
            )
        poly_smoothed = self._smoothed_polygon.astype(np.int32)

        # Polygon mask (where the garment CAN go geometrically)
        poly_mask = np.zeros((orig_h, orig_w), dtype=np.float32)
        cv2.fillPoly(poly_mask, [poly_smoothed], 1.0)
        blur = int(min(orig_w, orig_h) * 0.04) | 1
        poly_mask = cv2.GaussianBlur(poly_mask, (blur, blur), 0)

        # ── 5. Combine with body silhouette so edges follow the body ──
        # The polygon defines WHERE the dress goes; the body silhouette makes
        # the mask follow the actual person outline (smooth, curved, not
        # square). Final mask = polygon ∩ body_silhouette.
        if body_mask is not None and body_mask.shape[:2] == (orig_h, orig_w):
            # Soften the body silhouette edge so the composite blends instead of
            # cutting hard at one-pixel boundaries
            soft_body = cv2.GaussianBlur(body_mask, (15, 15), 0)
            mask = poly_mask * soft_body
        else:
            mask = poly_mask
        mask = np.clip(mask, 0.0, 1.0)[..., None]   # (H, W, 1) for broadcast

        # ── 6. Composite: AI render inside mask, webcam outside ────
        composited = ai_full * mask + frame_full * (1 - mask)
        return Image.fromarray(composited.astype(np.uint8))

    def _build_torso_polygon(self, pose_results, category: str, w: int, h: int):
        """Return a 4-vertex int32 polygon outlining where the garment goes,
        in (x, y) pixel coords, or None if no usable pose is detected.
        Caches the last good polygon so transient pose-loss frames keep the
        garment painted instead of flicking off.

        Note: new Tasks API returns `pose_landmarks` as List[List[Landmark]] —
        one inner list per detected person. We just use the first person.
        """
        plm = getattr(pose_results, "pose_landmarks", None) or []
        if not plm:
            return self._last_polygon
        lms = plm[0]  # first detected person's landmarks
        # MediaPipe Pose landmark indices (same as solutions API):
        #   11 = left shoulder, 12 = right shoulder
        #   23 = left hip,      24 = right hip
        def pt(i):
            return (int(lms[i].x * w), int(lms[i].y * h))
        def vis(i):
            return getattr(lms[i], "visibility", 1.0)

        if min(vis(11), vis(12), vis(23), vis(24)) < 0.4:
            return self._last_polygon

        lsh, rsh = pt(11), pt(12)
        lhp, rhp = pt(23), pt(24)

        shoulder_span = abs(lsh[0] - rsh[0])
        torso_h = abs((lhp[1] + rhp[1]) / 2 - (lsh[1] + rsh[1]) / 2)
        side_pad   = int(shoulder_span * SIDE_PAD.get(category, 0.30))
        collar_dy  = int(torso_h * COLLAR_RISE.get(category, 0.22))
        hem_dy     = int(torso_h * HEM_EXTEND.get(category, 2.00))

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


def _ensure_mp_task(parent_dir: Path, filename: str, url: str) -> Path:
    """Download a MediaPipe .task file from Google's CDN if not on disk."""
    task_path = parent_dir / filename
    if task_path.exists() and task_path.stat().st_size > 100_000:
        return task_path
    parent_dir.mkdir(parents=True, exist_ok=True)
    import urllib.request
    log.info("downloading %s from %s", filename, url)
    urllib.request.urlretrieve(url, str(task_path))
    log.info("%s → %s (%d bytes)", filename, task_path, task_path.stat().st_size)
    return task_path


def load_dm_vton_trt(path: Path) -> DMVTONPipe:
    """Load DM-VTON's `DMVTONPipeline` from the cloned ../DM-VTON/ repo +
    initialise the new MediaPipe Tasks PoseLandmarker for torso localisation.

    (The legacy `mediapipe.solutions.pose` API was removed in 0.10.20+; the
    Tasks API is the supported way now.)
    """
    import torch
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    repo_root = Path(__file__).resolve().parent / "DM-VTON"
    if not (repo_root / "pipelines").exists():
        raise FileNotFoundError(
            f"DM-VTON repo not found at {repo_root}. Run scripts/setup_dm_vton.sh first."
        )
    sys.path.insert(0, str(repo_root))

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
    dtype = torch.float32

    log.info("instantiating DMVTONPipeline…")
    pipeline = DMVTONPipeline(
        align_corners=True,
        checkpoints={"warp": str(warp_ckpt), "gen": str(gen_ckpt)},
    ).to(device).eval()

    mp_dir = path.parent / "mediapipe"

    log.info("loading MediaPipe PoseLandmarker (tasks API)…")
    pose_task = _ensure_mp_task(
        mp_dir, "pose_landmarker_lite.task",
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    )
    pose_detector = mp_vision.PoseLandmarker.create_from_options(
        mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(pose_task)),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_poses=1,
        )
    )

    log.info("loading MediaPipe ImageSegmenter (selfie body mask)…")
    # Image segmenter ships as .tflite, not .task. URL pattern is different
    # from PoseLandmarker — versioned path, no "latest" alias.
    seg_task = _ensure_mp_task(
        mp_dir, "selfie_segmenter.tflite",
        "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
        "selfie_segmenter/float16/1/selfie_segmenter.tflite",
    )
    segmenter = mp_vision.ImageSegmenter.create_from_options(
        mp_vision.ImageSegmenterOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(seg_task)),
            running_mode=mp_vision.RunningMode.IMAGE,
            output_confidence_masks=True,
            output_category_mask=False,
        )
    )

    log.info("DM-VTON pipeline ready (PyTorch %s on %s, with Pose + SelfieSegmentation)",
             dtype, device)
    return DMVTONPipe(
        pipeline=pipeline, device=device, tensor_dtype=dtype,
        pose=pose_detector, segmenter=segmenter,
    )
