#!/usr/bin/env python3
# Build a textured GLB from exactly two product photos: front + back.
# This is the PRODUCTION ingestion path. No AI inference, no Brush, no TripoSR.
# Runs anywhere (Mac dev / vendor box / cloud) in ~5 seconds per garment.
#
# Pipeline:
#   1. Background-remove front and back (rembg if PNGs aren't already alpha)
#   2. Extract the garment silhouette from the front image — drives the mesh's
#      per-height width profile (dresses flare at the hem; tops are uniform)
#   3. Build a smooth elliptical-cross-section "torso" mesh whose width follows
#      that silhouette. ~3k verts, no AI artifacts.
#   4. Compose front+back into a panorama texture, with a small linear blend
#      band at the seams so the side-view transitions are soft.
#   5. Wrap the panorama around the mesh cylindrically. theta=0 (mesh's local
#      -Z, the side that faces the camera in our live AR scene) maps to the
#      CENTRE of the front PNG. theta=π maps to the centre of the back PNG.
#   6. Export self-contained GLB.
#
# Usage:
#   python build_2view_glb.py front.png back.png output.glb
import sys
import argparse
import math
import numpy as np
from PIL import Image
from rembg import remove, new_session
import trimesh

HEIGHT = 1.0
RADIAL_SEGMENTS = 64
HEIGHT_SEGMENTS = 48
DEPTH_RATIO = 0.55          # mesh depth ÷ width — garments wider than deep
TARGET_TEXTURE_HEIGHT = 1024 # both source PNGs scaled to this height before blending
SEAM_BLEND_FRAC = 0.06       # ~6% of each PNG width gets blended at the side seams


def open_with_alpha(path: str, rembg_sess=None) -> Image.Image:
    """Return RGBA. If image has no real alpha, run rembg."""
    img = Image.open(path).convert('RGBA')
    a = np.array(img.split()[3])
    if (a < 250).mean() < 0.02:
        # Effectively no transparency — run rembg
        if rembg_sess is None:
            rembg_sess = new_session('u2net')
        img = remove(img, session=rembg_sess)
    return img


def composite_on_white(rgba: Image.Image) -> Image.Image:
    """Composite a transparent image onto white background, return RGB."""
    bg = Image.new('RGB', rgba.size, (255, 255, 255))
    bg.paste(rgba, (0, 0), rgba)
    return bg


def normalise_to_height(rgba: Image.Image, target_h: int) -> Image.Image:
    """Tight-crop by alpha, then resize to target height keeping aspect."""
    arr = np.array(rgba)
    alpha = arr[:, :, 3]
    rows = np.where(alpha.max(axis=1) > 30)[0]
    cols = np.where(alpha.max(axis=0) > 30)[0]
    if len(rows) and len(cols):
        rgba = rgba.crop((cols.min(), rows.min(), cols.max() + 1, rows.max() + 1))
    w, h = rgba.size
    new_w = int(w * target_h / h)
    return rgba.resize((new_w, target_h), Image.LANCZOS)


def silhouette_profile(front_rgba: Image.Image, segments: int) -> np.ndarray:
    """Return per-row width values (length=segments, normalised so max=1).
    widths[0] is the TOP of the mesh (collar) and widths[-1] is the BOTTOM (hem)."""
    arr = np.array(front_rgba)
    alpha = arr[:, :, 3]
    H, W = alpha.shape
    rows_with_pixels = np.where(alpha.max(axis=1) > 50)[0]
    if len(rows_with_pixels) == 0:
        return np.ones(segments)
    top, bot = rows_with_pixels.min(), rows_with_pixels.max()
    sample_rows = np.linspace(top, bot, segments).astype(int)
    widths = np.zeros(segments)
    for i, r in enumerate(sample_rows):
        cols = np.where(alpha[r] > 50)[0]
        if len(cols):
            widths[i] = cols.max() - cols.min()
    widths = widths / max(1, widths.max())
    # 3-tap smoothing
    kernel = np.ones(3) / 3
    widths = np.convolve(widths, kernel, mode='same')
    return widths  # PNG y=0 is image top = mesh +Y top, so order matches


def build_panorama(front_rgb: Image.Image, back_rgb: Image.Image) -> Image.Image:
    """Lay front | back side-by-side at uniform height. Add a small linear
    blend band at the seams (front→back middle, and back→front wrap) to
    soften side-view transitions.

    Layout:                                                                          u
        |--- front  ---|--- back  ---|                                              0 …… 1
                       ^seam                                                  also: 0 (front-L) → 0.5 (back-L) → 1 (back-R == front-L)
    """
    h = front_rgb.size[1]
    assert back_rgb.size[1] == h, "front and back must be same height before this step"
    fw, bw = front_rgb.size[0], back_rgb.size[0]

    # Match widths by padding the narrower one symmetrically (centred)
    target_w = max(fw, bw)
    def pad(img, target):
        w = img.size[0]
        if w == target: return img
        out = Image.new('RGB', (target, h), (250, 250, 250))
        out.paste(img, ((target - w) // 2, 0))
        return out
    front = pad(front_rgb, target_w)
    back  = pad(back_rgb, target_w)

    panorama = Image.new('RGB', (target_w * 2, h), (250, 250, 250))
    panorama.paste(front, (0, 0))
    panorama.paste(back, (target_w, 0))

    # Soft blend across the central seam (front right edge ↔ back left edge)
    blend_w = max(1, int(target_w * SEAM_BLEND_FRAC))
    pano_arr = np.array(panorama, dtype=np.float32)
    for i in range(blend_w):
        alpha = (i + 1) / (blend_w + 1)  # 0..1
        x_left = target_w - blend_w + i  # column in left (front) half
        x_right = target_w + i           # corresponding column in right (back) half
        front_col = pano_arr[:, x_left, :].copy()
        back_col  = pano_arr[:, x_right, :].copy()
        # Linearly interpolate: left side weights more toward front, right toward back
        pano_arr[:, x_left, :]  = (1 - alpha) * front_col + alpha * back_col
        pano_arr[:, x_right, :] = alpha * front_col + (1 - alpha) * back_col
    return Image.fromarray(pano_arr.astype(np.uint8))


def build_mesh(widths: np.ndarray):
    H = HEIGHT_SEGMENTS
    R = RADIAL_SEGMENTS
    max_x = 0.5

    verts, uvs, normals, faces = [], [], [], []
    for v_idx in range(H):
        v = v_idx / (H - 1)
        y = HEIGHT * (0.5 - v)
        radius_x = widths[v_idx] * max_x
        radius_z = radius_x * DEPTH_RATIO
        for u_idx in range(R + 1):
            u = u_idx / R
            # theta=0 is at LOCAL -Z (the side facing the camera). The texture
            # is laid out [front | back] so the CENTRE of the front PNG must
            # land at theta=0. The front PNG spans u_tex ∈ [0, 0.5] with centre
            # at 0.25, so we shift mesh u by 0.25 so mesh u=0 indexes texel u=0.25.
            theta = u * 2 * math.pi
            x = math.sin(theta) * radius_x
            z = -math.cos(theta) * radius_z
            verts.append([x, y, z])
            # Texture index: mesh u=0 → texel u=0.25 (front centre)
            u_tex = (u + 0.25) % 1.0
            uvs.append([u_tex, v])
            nl = math.hypot(math.sin(theta), -math.cos(theta))
            normals.append([math.sin(theta) / nl, 0.0, -math.cos(theta) / nl])

    cols = R + 1
    for v_idx in range(H - 1):
        for u_idx in range(R):
            a = v_idx * cols + u_idx
            b = a + 1
            c = a + cols
            d = c + 1
            faces.append([a, c, b])
            faces.append([b, c, d])

    return (np.array(verts, dtype=np.float32),
            np.array(faces, dtype=np.int64),
            np.array(normals, dtype=np.float32),
            np.array(uvs, dtype=np.float32))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('front_png')
    p.add_argument('back_png')
    p.add_argument('output_glb')
    args = p.parse_args()

    print(f"[2view] front: {args.front_png}")
    print(f"[2view] back:  {args.back_png}")
    print(f"[2view] out:   {args.output_glb}")

    sess = new_session('u2net')
    front_rgba = normalise_to_height(open_with_alpha(args.front_png, sess), TARGET_TEXTURE_HEIGHT)
    back_rgba  = normalise_to_height(open_with_alpha(args.back_png,  sess), TARGET_TEXTURE_HEIGHT)
    print(f"[2view] front {front_rgba.size}, back {back_rgba.size}")

    widths = silhouette_profile(front_rgba, HEIGHT_SEGMENTS)
    print(f"[2view] silhouette widths: top={widths[0]:.2f} mid={widths[len(widths)//2]:.2f} bot={widths[-1]:.2f}")

    panorama = build_panorama(composite_on_white(front_rgba),
                              composite_on_white(back_rgba))
    print(f"[2view] panorama: {panorama.size}")

    verts, faces, normals, uvs = build_mesh(widths)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals, process=False)
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=panorama,
        metallicFactor=0.0,
        roughnessFactor=0.85,
    )
    mesh.visual = trimesh.visual.TextureVisuals(uv=uvs, material=material)
    mesh.export(args.output_glb)

    import os
    print(f"[2view] wrote {args.output_glb}  ({os.path.getsize(args.output_glb)/1e6:.2f} MB)")
    print(f"[2view] verts={len(verts)}  faces={len(faces)}")


if __name__ == '__main__':
    main()
