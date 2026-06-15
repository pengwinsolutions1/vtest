#!/usr/bin/env python3
# Build a textured GLB from front (and optional back) product photos using
# Hunyuan3D-2mini for geometry + cylindrical UV projection of the real photos
# for the texture.
#
# Why this combination:
#   - Hunyuan generates a REAL 3D mesh (proper sleeves, neckline, curvature) —
#     much better geometric variety than a procedural elliptical cylinder.
#   - Cylindrical UV with the actual product photos preserves real colors and
#     patterns — Hunyuan's "paint" pipeline would re-imagine the texture; we'd
#     rather show the actual garment.
#   - Result: real 3D shape + real product photos, like a hand-modelled asset.
#
# Time: ~8-12 min per garment on M1 Pro 16 GB (Hunyuan shape ~7 min, rest ~1 min).
# Output: ~500 KB - 2 MB GLB at 30k tris.
#
# Usage:
#   python build_hunyuan_glb.py front.png back.png output.glb [--steps N]
#   python build_hunyuan_glb.py front.png _ output.glb   # back omitted, front used for both halves
#
# License note: Hunyuan3D-2 is Tencent Non-Commercial. For commercial deployment
# swap the shape model to TRELLIS (MIT) or buy a Tencent license.
import sys
import os
import argparse
import time
import math
import numpy as np
from PIL import Image
import trimesh

# Hunyuan API
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Hunyuan3D-2'))
from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

import open3d as o3d


SEAM_BLEND_FRAC = 0.06            # soft blend at the side seams
TARGET_TEXTURE_HEIGHT = 1024      # both source PNGs scaled to this before pano
TARGET_TRIS = 30_000              # decimate Hunyuan's ~1.5M tris down to this


# ============================================================
# 1. Background removal + image normalisation
# ============================================================

def open_with_alpha(path: str, rembg_sess=None) -> Image.Image:
    img = Image.open(path).convert('RGBA')
    a = np.array(img.split()[3])
    if (a < 250).mean() < 0.02:
        if rembg_sess is None:
            rembg_sess = new_session('u2net')
        img = remove(img, session=rembg_sess)
    return img


def composite_on_white(rgba: Image.Image) -> Image.Image:
    bg = Image.new('RGB', rgba.size, (255, 255, 255))
    bg.paste(rgba, (0, 0), rgba)
    return bg


def normalise_to_height(rgba: Image.Image, target_h: int) -> Image.Image:
    arr = np.array(rgba)
    alpha = arr[:, :, 3]
    rows = np.where(alpha.max(axis=1) > 30)[0]
    cols = np.where(alpha.max(axis=0) > 30)[0]
    if len(rows) and len(cols):
        rgba = rgba.crop((cols.min(), rows.min(), cols.max() + 1, rows.max() + 1))
    w, h = rgba.size
    new_w = int(w * target_h / h)
    return rgba.resize((new_w, target_h), Image.LANCZOS)


def build_panorama(front_rgb: Image.Image, back_rgb: Image.Image) -> Image.Image:
    """Lay front | back side-by-side at uniform height, blend at the central seam."""
    h = front_rgb.size[1]
    assert back_rgb.size[1] == h
    fw, bw = front_rgb.size[0], back_rgb.size[0]
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

    blend_w = max(1, int(target_w * SEAM_BLEND_FRAC))
    pa = np.array(panorama, dtype=np.float32)
    for i in range(blend_w):
        alpha = (i + 1) / (blend_w + 1)
        x_left = target_w - blend_w + i
        x_right = target_w + i
        fc = pa[:, x_left, :].copy()
        bc = pa[:, x_right, :].copy()
        pa[:, x_left, :]  = (1 - alpha) * fc + alpha * bc
        pa[:, x_right, :] = alpha * fc + (1 - alpha) * bc
    return Image.fromarray(pa.astype(np.uint8))


# ============================================================
# 2. Hunyuan shape generation
# ============================================================

def run_hunyuan_shape(front_rgba: Image.Image, steps: int, device: str) -> trimesh.Trimesh:
    print(f"[hunyuan] loading shape pipeline…")
    t0 = time.time()
    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        'tencent/Hunyuan3D-2mini',
        subfolder='hunyuan3d-dit-v2-mini',
        device=device,
    )
    print(f"[hunyuan] pipeline ready in {time.time() - t0:.1f}s")

    print(f"[hunyuan] generating mesh ({steps} steps)…")
    t0 = time.time()
    mesh = pipe(image=front_rgba, num_inference_steps=steps, guidance_scale=5.0)[0]
    print(f"[hunyuan] shape: verts={len(mesh.vertices)} faces={len(mesh.faces)} in {time.time() - t0:.1f}s")
    return mesh


# ============================================================
# 3. Mesh post-processing (decimate, orient, normalize)
# ============================================================

def decimate_and_orient(mesh: trimesh.Trimesh, target_tris: int) -> trimesh.Trimesh:
    """Decimate to target tri count, centre at origin, normalise so max dim=1,
    flip to collar-up if necessary."""
    om = o3d.geometry.TriangleMesh()
    om.vertices = o3d.utility.Vector3dVector(np.array(mesh.vertices))
    om.triangles = o3d.utility.Vector3iVector(np.array(mesh.faces))
    om = om.simplify_quadric_decimation(target_number_of_triangles=target_tris)
    om.compute_vertex_normals()
    verts = np.asarray(om.vertices)
    faces = np.asarray(om.triangles)

    # Centre + normalise
    centre = (verts.max(0) + verts.min(0)) / 2
    verts = verts - centre
    maxdim = (verts.max(0) - verts.min(0)).max()
    verts = verts / maxdim

    # Collar-up heuristic: top 10% of Y should be narrower than bottom 10%
    y = verts[:, 1]
    y_max, y_min = y.max(), y.min(); y_range = y_max - y_min
    top = verts[y > y_max - y_range * 0.1]
    bot = verts[y < y_min + y_range * 0.1]
    top_w = top[:, 0].max() - top[:, 0].min() if len(top) else 0
    bot_w = bot[:, 0].max() - bot[:, 0].min() if len(bot) else 0
    if top_w > bot_w:
        print("[hunyuan] flipping mesh 180° about X (collar was at bottom)")
        R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
        verts = verts @ R.T
    return trimesh.Trimesh(vertices=verts, faces=faces, process=True)


# ============================================================
# 4. Cylindrical UV mapping (theta=0 → -Z, the camera-visible side)
# ============================================================

def apply_cylindrical_uvs(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Compute per-vertex UVs by cylindrical projection around Y axis."""
    verts = np.array(mesh.vertices)
    uvs = np.zeros((len(verts), 2), dtype=np.float32)
    # theta = atan2(x, -z) gives 0 at +X / pi at -X, with theta=0 at local -Z (camera side)
    # Actually we want: theta=0 ↔ -Z ↔ front-of-garment ↔ u=0.25 in panorama [front|back]
    for i, v in enumerate(verts):
        x, y, z = v
        # theta = 0 should be at -Z. atan2(x, -z) does that.
        theta = math.atan2(x, -z)        # range (-pi, pi], theta=0 at -Z
        u_mesh = (theta + math.pi) / (2 * math.pi)   # 0..1; u_mesh=0.5 is theta=0 (-Z)
        # Panorama layout: [front | back] with u_tex in [0,1]
        # Map u_mesh=0.5 → u_tex=0.25 (front centre)
        # u_mesh=1.0 (theta=π = +Z = back centre) → u_tex=0.75
        # Formula: u_tex = (u_mesh - 0.25) mod 1
        u_tex = (u_mesh - 0.25) % 1.0
        # v: top of mesh (y=+0.5) → image top (v=0); bottom (y=-0.5) → image bottom (v=1)
        v_tex = 0.5 - y  # y in [-0.5, 0.5] → v in [1.0, 0.0]
        v_tex = max(0.0, min(1.0, v_tex))
        uvs[i] = [u_tex, v_tex]
    mesh.visual = trimesh.visual.TextureVisuals(uv=uvs)
    return mesh


# ============================================================
# 5. Main
# ============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument('front_png')
    p.add_argument('back_png', help="path to back PNG, or '_' to use front for both halves")
    p.add_argument('output_glb')
    p.add_argument('--steps', type=int, default=25)
    p.add_argument('--device', default='mps')
    args = p.parse_args()

    # Import rembg lazily so the script still works if rembg fails to load
    global remove, new_session
    from rembg import remove, new_session

    # Load + normalise images
    front_rgba = normalise_to_height(open_with_alpha(args.front_png), TARGET_TEXTURE_HEIGHT)
    if args.back_png in ('_', '-', 'none'):
        # No back provided. Generate a fabric-like back by sampling the
        # garment's dominant colour from the front and overlaying a heavily
        # blurred version of the front for subtle shading variation.
        # Much better than a literal mirror (which makes the back look like
        # a backwards garment with the neckline/buttons facing the wrong way).
        import numpy as np
        from PIL import ImageFilter, ImageEnhance
        arr = np.array(front_rgba)
        alpha = arr[:, :, 3]
        rgb_pixels = arr[alpha > 128][:, :3]
        if len(rgb_pixels) == 0:
            dom_rgb = (200, 100, 100)
        else:
            dom_rgb = tuple(int(c) for c in rgb_pixels.mean(axis=0))
        print(f"[build] no back image — synthesising fabric-like back, dominant color = {dom_rgb}")
        # Blur the front aggressively (radius ≈ 12% of width)
        blur_r = max(8, int(front_rgba.size[0] * 0.12))
        blurred = front_rgba.filter(ImageFilter.GaussianBlur(radius=blur_r))
        # Composite blurred (with original alpha) over a solid dominant-color background
        bg_solid = Image.new('RGBA', front_rgba.size, dom_rgb + (255,))
        back_rgba = Image.alpha_composite(bg_solid, blurred)
        # Slight desaturation so it doesn't compete with the detailed front
        back_rgba = ImageEnhance.Color(back_rgba.convert('RGB')).enhance(0.85).convert('RGBA')
        # Preserve the original alpha (silhouette)
        back_rgba.putalpha(front_rgba.split()[3])
    else:
        back_rgba = normalise_to_height(open_with_alpha(args.back_png), TARGET_TEXTURE_HEIGHT)

    # 1) Shape from Hunyuan
    shape_mesh = run_hunyuan_shape(front_rgba, args.steps, args.device)

    # 2) Decimate + orient + normalise
    mesh = decimate_and_orient(shape_mesh, TARGET_TRIS)
    bb = mesh.bounds[1] - mesh.bounds[0]
    print(f"[build] post-decimate: verts={len(mesh.vertices)} bb={bb.round(3).tolist()}")

    # 3) Cylindrical UVs
    apply_cylindrical_uvs(mesh)

    # 4) Build panorama + attach as material
    panorama = build_panorama(composite_on_white(front_rgba), composite_on_white(back_rgba))
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=panorama,
        metallicFactor=0.0,
        roughnessFactor=0.85,
    )
    mesh.visual = trimesh.visual.TextureVisuals(uv=mesh.visual.uv, material=material)

    # 5) Export
    mesh.export(args.output_glb)
    print(f"[build] wrote {args.output_glb}  ({os.path.getsize(args.output_glb)/1e6:.2f} MB)")


if __name__ == '__main__':
    main()
