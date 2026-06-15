#!/usr/bin/env python3
# Build a high-quality GLB from a 360° turntable image set, by:
#   1. Removing the studio background from every frame (rembg + alpha mask)
#   2. Extracting the garment silhouette from the front frame to drive the
#      mesh's per-height width profile
#   3. Concatenating all frames side-by-side into a panoramic texture
#   4. Building a smooth elliptical-cross-section mesh whose width follows
#      the silhouette, with cylindrical UV mapping that wraps the panorama
#      around the garment
#   5. Exporting a self-contained .glb
#
# Why this looks better than TripoSR-on-one-frame:
#   - Each viewing angle shows the ACTUAL product photo (not a hallucination)
#   - Geometry is smooth procedural, not noisy diffusion output
#   - Output mesh is ~5k verts (small, fast) instead of ~100k chunky verts
#
# Usage:
#   python build_360_glb.py /path/to/frames_dir /path/to/output.glb [--frames N]
#
# Where frames_dir contains N images named such that sorted() produces the
# rotation order. The script will sample up to --frames evenly.
import os
import sys
import argparse
import glob
import math
import numpy as np
from PIL import Image, ImageFilter
from rembg import remove, new_session
import trimesh

# Geometry constants. All in arbitrary units; live AR scales the mesh to fit
# the user's body, so absolute size doesn't matter — only ratios do.
HEIGHT = 1.0           # mesh's vertical extent (Y axis)
RADIAL_SEGMENTS = 64   # how many segments around the cylinder
HEIGHT_SEGMENTS = 48   # how many rows up the cylinder
DEPTH_RATIO = 0.55     # depth (Z) ÷ width (X) — garments are wider than deep


def load_and_alpha(frames_dir: str, max_frames: int) -> list[np.ndarray]:
    """Load up to max_frames evenly-spaced images from frames_dir, run rembg
    on each to obtain an RGBA where alpha > 0 marks the garment pixels."""
    files = sorted(glob.glob(os.path.join(frames_dir, '*.jpg')) +
                   glob.glob(os.path.join(frames_dir, '*.png')))
    if not files:
        sys.exit(f"no images in {frames_dir}")
    n = len(files)
    if n > max_frames:
        step = n / max_frames
        files = [files[int(i * step)] for i in range(max_frames)]
    print(f"[build] using {len(files)} of {n} source frames")

    sess = new_session('u2net')
    frames = []
    for i, f in enumerate(files):
        img = Image.open(f).convert('RGBA')
        cut = remove(img, session=sess)
        frames.append(np.array(cut))
        if (i + 1) % 10 == 0 or i == len(files) - 1:
            print(f"[build]   bg-removed {i+1}/{len(files)}")
    return frames


def silhouette_profile(frame: np.ndarray, segments: int) -> np.ndarray:
    """From an RGBA front frame, return a 1D array of length `segments` where
    each value ∈ [0,1] is the garment's width at that height row (top→bottom).

    Used to drive the mesh's per-row width: a tshirt is roughly uniform; a
    dress flares out at the bottom. We measure left-and-right-most opaque
    pixel per row.
    """
    H, W = frame.shape[:2]
    alpha = frame[:, :, 3]
    # Find rows that actually have garment pixels
    has_pixels = alpha.max(axis=1) > 50
    rows = np.where(has_pixels)[0]
    if len(rows) == 0:
        return np.ones(segments)
    top, bot = rows.min(), rows.max()
    # Sample `segments` heights evenly between top and bottom
    sample_rows = np.linspace(top, bot, segments).astype(int)
    widths = np.zeros(segments)
    for i, r in enumerate(sample_rows):
        cols = np.where(alpha[r] > 50)[0]
        if len(cols):
            widths[i] = cols.max() - cols.min()
    # Normalise so max width = 1
    if widths.max() > 0:
        widths = widths / widths.max()
    # Smooth a tiny bit
    k = 3
    kernel = np.ones(k) / k
    widths = np.convolve(widths, kernel, mode='same')
    # Top row has v=0, bottom has v=1. Reverse so widths[0] is the TOP of the
    # mesh and matches mesh's +Y end.
    return widths[::-1]


def crop_and_pad(frame: np.ndarray, target_aspect: float) -> np.ndarray:
    """Tight-crop the garment by alpha then pad to a uniform aspect ratio.
    Ensures every frame contributes the same vertical region of the panorama,
    so wrapping looks consistent as the user turns."""
    alpha = frame[:, :, 3]
    rows = np.where(alpha.max(axis=1) > 30)[0]
    cols = np.where(alpha.max(axis=0) > 30)[0]
    if len(rows) == 0 or len(cols) == 0:
        return frame
    r0, r1 = rows.min(), rows.max() + 1
    c0, c1 = cols.min(), cols.max() + 1
    cropped = frame[r0:r1, c0:c1]
    h, w = cropped.shape[:2]
    # Pad horizontally to hit target aspect = w/h
    desired_w = int(h * target_aspect)
    if desired_w > w:
        pad = (desired_w - w) // 2
        out = np.zeros((h, desired_w, 4), dtype=cropped.dtype)
        out[:, pad:pad+w] = cropped
        return out
    return cropped


def build_panorama(frames: list[np.ndarray], strip_h: int = 1024) -> Image.Image:
    """Build a wide RGBA panorama by laying every frame side-by-side at
    a uniform height. Returns a PIL image."""
    target_aspect = 1 / 2.5  # each frame ~2.5x taller than wide
    strip_w = int(strip_h * target_aspect)
    panel = Image.new('RGBA', (strip_w * len(frames), strip_h), (0, 0, 0, 0))
    for i, f in enumerate(frames):
        f = crop_and_pad(f, target_aspect)
        pil = Image.fromarray(f).resize((strip_w, strip_h), Image.LANCZOS)
        panel.paste(pil, (i * strip_w, 0))
    # Convert to RGB with white background (so missing pixels are pleasant)
    rgb = Image.new('RGB', panel.size, (250, 250, 250))
    rgb.paste(panel, (0, 0), panel)
    return rgb


def build_mesh(widths: np.ndarray, max_x: float = 0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build vertices, faces, normals, UVs for an elliptical-cross-section
    mesh whose width(v) follows the silhouette profile. Returns numpy arrays.
    """
    H = HEIGHT_SEGMENTS
    R = RADIAL_SEGMENTS
    assert len(widths) == H, f"widths len {len(widths)} != HEIGHT_SEGMENTS {H}"

    verts = []
    uvs = []
    normals = []
    for v_idx in range(H):
        v = v_idx / (H - 1)
        # Bend the silhouette so y=+HEIGHT/2 is the TOP (small index) and
        # y=-HEIGHT/2 is the BOTTOM (large index).
        y = HEIGHT * (0.5 - v)
        radius_x = widths[v_idx] * max_x
        radius_z = radius_x * DEPTH_RATIO
        for u_idx in range(R + 1):  # +1 to close the seam
            u = u_idx / R
            # Cylindrical UV: u wraps around, v goes top→bottom.
            # The texture's left edge (u=0) is the FRONT view (frame 0). The
            # live AR places the cylinder ~1.2m in front of the Three.js camera
            # which looks down +Z, so the VISIBLE side of the cylinder is its
            # local -Z hemisphere. To make u=0 (frame 0) visible to the user,
            # we map theta=0 to -Z instead of +Z. theta now wraps anticlockwise
            # so that frame 1 ends up on the user's LEFT shoulder (screen right
            # due to mirroring), which matches the natural turntable direction.
            theta = u * 2 * math.pi
            x = math.sin(theta) * radius_x
            z = -math.cos(theta) * radius_z   # ← was +cos, now -cos
            verts.append([x, y, z])
            uvs.append([u, v])
            # Normal: outward from the ellipse axis (Y).
            nx = math.sin(theta)
            nz = -math.cos(theta)
            n_len = math.hypot(nx, nz)
            normals.append([nx / n_len, 0.0, nz / n_len])

    verts = np.array(verts, dtype=np.float32)
    uvs = np.array(uvs, dtype=np.float32)
    normals = np.array(normals, dtype=np.float32)

    faces = []
    cols = R + 1
    for v_idx in range(H - 1):
        for u_idx in range(R):
            a = v_idx * cols + u_idx
            b = a + 1
            c = a + cols
            d = c + 1
            faces.append([a, c, b])
            faces.append([b, c, d])
    faces = np.array(faces, dtype=np.int64)
    return verts, faces, normals, uvs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('frames_dir')
    p.add_argument('output_glb')
    p.add_argument('--frames', type=int, default=32,
                   help='max frames to sample from frames_dir (default 32)')
    args = p.parse_args()

    print(f"[build] input:  {args.frames_dir}")
    print(f"[build] output: {args.output_glb}")

    frames = load_and_alpha(args.frames_dir, max_frames=args.frames)

    print("[build] extracting silhouette profile from front frame")
    widths = silhouette_profile(frames[0], HEIGHT_SEGMENTS)

    print(f"[build] building panorama ({len(frames)} frames)")
    panorama = build_panorama(frames)

    print("[build] building mesh")
    verts, faces, normals, uvs = build_mesh(widths)

    mesh = trimesh.Trimesh(
        vertices=verts,
        faces=faces,
        vertex_normals=normals,
        process=False,
    )
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=panorama,
        metallicFactor=0.0,
        roughnessFactor=0.85,
    )
    mesh.visual = trimesh.visual.TextureVisuals(uv=uvs, material=material)

    mesh.export(args.output_glb)
    size = os.path.getsize(args.output_glb)
    print(f"[build] wrote {args.output_glb}  ({size/1e6:.2f} MB)")
    print(f"[build] verts={len(verts)}  faces={len(faces)}")


if __name__ == '__main__':
    main()
