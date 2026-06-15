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


# Per-category polygon geometry (all values are fractions of body landmark
# distances so they scale with the user's size on screen). Tune these to
# adjust how the garment sits on the body.
#
# HEM_EXTEND:   fraction of shoulder→hip distance to extend below the hips
# COLLAR_RISE:  fraction of shoulder→hip distance to extend above shoulders
#               (positive = up toward chin, negative = below shoulder line)
# SIDE_PAD:     extra width on each side, as fraction of shoulder span
# HAS_SLEEVES:  true for tops → polygon includes shoulder→elbow regions so
#               the rendered "dress" follows raised arms

HEM_EXTEND: dict[str, float] = {
    "top":    0.55,
    "bottom": 2.40,
    "dress":  3.20,   # ankle-length: dress drapes well past the visible body
}
COLLAR_RISE: dict[str, float] = {
    # MUST stay small — anything > ~0.20 puts the polygon top into the face.
    # 0.05-0.10 sits the collar just above the shoulder line where a real
    # garment's neckline is.
    "top":    0.08,
    "bottom": -0.95,
    "dress":  0.08,
}
SIDE_PAD: dict[str, float] = {
    "top":    0.45,
    "bottom": 0.40,
    "dress":  0.55,
}

# 8-vertex sleeve polygon disabled by default: tracks elbows but creates
# weird "garment moves with arm" artifacts the user explicitly didn't want.
# Arms always show through the webcam outside the torso polygon — cleaner.
HAS_SLEEVES: dict[str, bool] = {
    "top":    False,
    "bottom": False,
    "dress":  False,
}
SLEEVE_OUTSET: float = 0.35

# Below the hip line, do NOT clip the polygon by the body silhouette.
# Dresses drape past the body (skirt fabric hangs in front of legs / off-frame),
# so clipping makes the lower dress vanish whenever legs aren't fully visible.
USE_BODY_MASK_BELOW_HIPS: dict[str, bool] = {
    "top":    True,
    "bottom": False,
    "dress":  False,
}

# Polygon smoothing — how fast the polygon catches up to a new pose detection.
POLYGON_SMOOTH_ALPHA = 0.55

# Pixels of dilation to apply to the body silhouette mask before intersecting
# with the polygon. Closes thin gaps where the segmenter undercut the body.
BODY_DILATE_PX = 4


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

        # EMA-smooth the polygon vertices to reduce frame-to-frame jitter.
        # Snap to the new polygon if the vertex count changed (e.g. category
        # switched between has-sleeves and not-has-sleeves).
        if (self._smoothed_polygon is None
                or self._smoothed_polygon.shape != polygon.shape):
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
            # Dilate the body silhouette by a few pixels so thin gaps in the
            # segmenter output (around fingers, hair, etc.) don't leave the
            # composite with little black holes where the AI dress should be.
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (BODY_DILATE_PX * 2 + 1, BODY_DILATE_PX * 2 + 1)
            )
            body_dilated = cv2.dilate(body_mask, kernel)
            soft_body = cv2.GaussianBlur(body_dilated, (15, 15), 0)

            # For dresses + bottoms, the garment extends BELOW the visible
            # body. Clipping by body silhouette there makes the lower dress
            # disappear whenever legs aren't fully in frame. Override the
            # body mask to 1.0 below the hip line so the polygon's hem area
            # drapes freely.
            effective_body = soft_body
            if not USE_BODY_MASK_BELOW_HIPS.get(category, True):
                # Find the hip y from the smoothed polygon (last vertex on the
                # bottom edge). For a 4-vertex polygon: index 2 or 3 has hem y.
                hip_y = self._hip_y_from_pose(pose_results, orig_h)
                if hip_y is not None:
                    below = np.zeros((orig_h, orig_w), dtype=np.float32)
                    below[hip_y:, :] = 1.0
                    # Soft transition across ~10% of frame height
                    soft_below = cv2.GaussianBlur(below, (61, 61), 0)
                    effective_body = soft_body * (1 - soft_below) + 1.0 * soft_below
            mask = poly_mask * effective_body
        else:
            mask = poly_mask
        mask = np.clip(mask, 0.0, 1.0)[..., None]   # (H, W, 1) for broadcast

        # ── 6. Composite: AI render inside mask, webcam outside ────
        composited = ai_full * mask + frame_full * (1 - mask)
        return Image.fromarray(composited.astype(np.uint8))

    def _hip_y_from_pose(self, pose_results, h: int):
        """Return the average hip y in pixels, or None if not detectable."""
        plm = getattr(pose_results, "pose_landmarks", None) or []
        if not plm:
            return None
        lms = plm[0]
        try:
            l_vis = getattr(lms[23], "visibility", 1.0)
            r_vis = getattr(lms[24], "visibility", 1.0)
            if min(l_vis, r_vis) < 0.4:
                return None
            return int((lms[23].y + lms[24].y) / 2 * h)
        except (IndexError, AttributeError):
            return None

    def _build_torso_polygon(self, pose_results, category: str, w: int, h: int):
        """Return an N-vertex int32 polygon outlining where the garment goes.

        - 4-vertex (rectangle-ish) for dress / bottom / sleeveless: shoulders
          outward to hips + extended hem
        - 8-vertex for tops with sleeves: includes shoulder→elbow rectangles
          on each arm so the garment follows the user's gestures

        Caches the last good polygon so a transient pose-loss frame doesn't
        flicker the dress off.
        """
        plm = getattr(pose_results, "pose_landmarks", None) or []
        if not plm:
            return self._last_polygon
        lms = plm[0]
        # MediaPipe Pose landmark indices:
        #   11 L shoulder  12 R shoulder
        #   13 L elbow     14 R elbow
        #   15 L wrist     16 R wrist
        #   23 L hip       24 R hip
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

        if not HAS_SLEEVES.get(category, False):
            # ── Simple 4-vertex polygon (dress, sleeveless top, bottom) ──
            polygon = np.array([
                [lsh[0] + side_pad, lsh[1] - collar_dy],  # subject's left shoulder, outset up
                [rsh[0] - side_pad, rsh[1] - collar_dy],  # subject's right shoulder, outset up
                [rhp[0] - side_pad, rhp[1] + hem_dy],     # below subject's right hip
                [lhp[0] + side_pad, lhp[1] + hem_dy],     # below subject's left hip
            ], dtype=np.int32)
        else:
            # ── 8-vertex polygon with sleeves (top + most dresses) ──
            # Add elbow points so the polygon follows the arms when the user
            # raises them. If an elbow isn't visible, fall back to the shoulder
            # position so the sleeve degenerates back to a normal top.
            if vis(13) > 0.4 and vis(14) > 0.4:
                lel, rel = pt(13), pt(14)
            else:
                lel, rel = lsh, rsh
            sleeve_out = int(shoulder_span * SLEEVE_OUTSET)

            # Build polygon clockwise starting from the subject's left shoulder
            # (= screen-right after the mirror). Order matters for fillPoly.
            polygon = np.array([
                [lsh[0] + side_pad, lsh[1] - collar_dy],          # L shoulder cap (up + out)
                [lel[0] + sleeve_out, lel[1]],                     # L elbow (outer sleeve)
                [lel[0], lel[1] + sleeve_out // 3],                # L elbow (inner sleeve)
                [lhp[0] + side_pad, lhp[1] + hem_dy],              # below L hip
                [rhp[0] - side_pad, rhp[1] + hem_dy],              # below R hip
                [rel[0], rel[1] + sleeve_out // 3],                # R elbow (inner)
                [rel[0] - sleeve_out, rel[1]],                     # R elbow (outer)
                [rsh[0] - side_pad, rsh[1] - collar_dy],           # R shoulder cap
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
