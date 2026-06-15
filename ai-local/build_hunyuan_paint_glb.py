#!/usr/bin/env python3
# High-quality single-image-to-textured-GLB using BOTH Hunyuan3D-2 pipelines:
#   1. Hunyuan3D-2mini (shape DiT) — single front image → 3D mesh
#   2. Hunyuan3D-2 paint — same image + shape → multi-view inferred RGB
#      baked as a UV-unwrapped texture atlas
#
# This is the proper "trial room quality" path: the back of the mesh is
# AI-generated to match the front (not mirrored, not blended), with smooth
# inferred sides.
#
# Time on M1 Pro 16 GB: shape ~7 min + paint ~5–15 min ≈ 15–25 min per garment.
# License: Tencent Non-Commercial. Swap to a commercial model for paid prod.
#
# Usage:
#   python build_hunyuan_paint_glb.py path/to/front.png path/to/output.glb [--steps 25]
import sys
import os
import argparse
import time
from PIL import Image

# Bring the cloned Hunyuan3D-2 source onto sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Hunyuan3D-2'))

import trimesh
import numpy as np
from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.texgen import Hunyuan3DPaintPipeline

import open3d as o3d


TARGET_TRIS = 30_000  # decimate after paint for web delivery


def open_image(path: str) -> Image.Image:
    img = Image.open(path).convert('RGBA')
    a = np.array(img.split()[3])
    has_alpha = (a < 250).mean() >= 0.02
    if not has_alpha:
        print("[paint] no real alpha — running rembg")
        rembg = BackgroundRemover()
        img = rembg(img)
    return img


def decimate_and_orient(mesh: trimesh.Trimesh, target_tris: int) -> trimesh.Trimesh:
    """Decimate, centre at origin, normalise to max dim=1, flip if upside down.
    Preserves UVs + texture when present."""
    has_texture = (
        hasattr(mesh.visual, 'material') and getattr(mesh.visual.material, 'baseColorTexture', None) is not None
    )

    om = o3d.geometry.TriangleMesh()
    om.vertices = o3d.utility.Vector3dVector(np.array(mesh.vertices))
    om.triangles = o3d.utility.Vector3iVector(np.array(mesh.faces))
    # If we have vertex UVs, decimation will mangle them. Skip decimate when textured.
    if not has_texture:
        om = om.simplify_quadric_decimation(target_number_of_triangles=target_tris)
        om.compute_vertex_normals()
        verts = np.asarray(om.vertices)
        faces = np.asarray(om.triangles)
    else:
        verts = np.array(mesh.vertices)
        faces = np.array(mesh.faces)

    centre = (verts.max(0) + verts.min(0)) / 2
    verts = verts - centre
    maxdim = (verts.max(0) - verts.min(0)).max()
    verts = verts / maxdim

    y = verts[:, 1]
    y_max, y_min = y.max(), y.min(); y_range = y_max - y_min
    top = verts[y > y_max - y_range * 0.1]
    bot = verts[y < y_min + y_range * 0.1]
    top_w = top[:, 0].max() - top[:, 0].min() if len(top) else 0
    bot_w = bot[:, 0].max() - bot[:, 0].min() if len(bot) else 0
    if top_w > bot_w:
        print("[paint] flipping 180° around X (collar was at bottom)")
        verts[:, [1, 2]] = -verts[:, [1, 2]]

    out = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    # Preserve materials/UVs if present (don't process=True or trimesh drops them)
    if has_texture:
        out.visual = mesh.visual
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('front_png')
    p.add_argument('output_glb')
    p.add_argument('--steps', type=int, default=25, help='shape diffusion steps')
    p.add_argument('--device', default='mps')
    args = p.parse_args()

    print(f"[paint] front: {args.front_png}")
    print(f"[paint] out:   {args.output_glb}")

    img = open_image(args.front_png)

    # ── Stage 1: shape ──────────────────────────────────────────────────
    print("[paint] loading shape pipeline…")
    t0 = time.time()
    shape_pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        'tencent/Hunyuan3D-2mini',
        subfolder='hunyuan3d-dit-v2-mini',
        device=args.device,
    )
    print(f"[paint] shape pipeline ready in {time.time() - t0:.1f}s")

    print(f"[paint] generating shape ({args.steps} steps)…")
    t0 = time.time()
    shape_mesh = shape_pipe(image=img, num_inference_steps=args.steps, guidance_scale=5.0)[0]
    print(f"[paint] shape: verts={len(shape_mesh.vertices)} faces={len(shape_mesh.faces)} in {time.time() - t0:.1f}s")

    # Decimate the bare shape BEFORE paint so painting is faster and the
    # final mesh stays under the web-delivery budget.
    shape_mesh = decimate_and_orient(shape_mesh, TARGET_TRIS)
    print(f"[paint] decimated to {len(shape_mesh.faces)} tris")

    # Free shape pipeline before loading paint to keep peak memory down
    del shape_pipe
    import gc; gc.collect()

    # ── Stage 2: paint ──────────────────────────────────────────────────
    print("[paint] loading paint pipeline…")
    t0 = time.time()
    paint_pipe = Hunyuan3DPaintPipeline.from_pretrained('tencent/Hunyuan3D-2')
    print(f"[paint] paint pipeline ready in {time.time() - t0:.1f}s")

    print("[paint] painting mesh…")
    t0 = time.time()
    textured_mesh = paint_pipe(shape_mesh, image=img)
    print(f"[paint] painted in {time.time() - t0:.1f}s")

    # Final orient + export. Don't decimate again — paint baked UVs.
    textured_mesh = decimate_and_orient(textured_mesh, TARGET_TRIS)
    textured_mesh.export(args.output_glb)
    sz = os.path.getsize(args.output_glb)
    print(f"[paint] wrote {args.output_glb}  ({sz/1e6:.2f} MB)")


if __name__ == '__main__':
    main()
