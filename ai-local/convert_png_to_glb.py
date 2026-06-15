#!/usr/bin/env python3
# Convert a garment image into a textured GLB using TripoSR (Mac MPS).
#
# Usage:
#   python convert_png_to_glb.py path/to/garment.png [output_dir]
#
# Default output_dir is the same dir as the input. Output filename matches the
# input's base name with .glb extension — so dropping it into seed-garments/
# automatically pairs with the matching .png (see webapp/lib/db.ts).
#
# Run from the venv:
#   source /Users/yugandhar/VirtualTryOn/ai-local/venv/bin/activate
#   python convert_png_to_glb.py /path/to/foo.png
#
# Roughly 60s per garment on M1 Pro 16 GB.
#
# Pipeline:
#   1. Feed the image straight to TripoSR — let its built-in rembg isolate
#      the garment from the studio background. (Previously we pre-composited
#      onto a white BG and skipped this; TripoSR then built 3D geometry
#      around the white area too — visible as a white "fluff" surrounding
#      the garment. Now fixed.)
#   2. TripoSR extracts a 3D mesh + bakes a 1024 texture.
#   3. Inspect the mesh bounding box; rotate so the LONGEST principal axis
#      ends up as +Y (vertical) — TripoSR sometimes outputs with the
#      garment's long axis along Z instead of Y, producing a "lying down"
#      mesh in the live AR.
#   4. Trimesh exports a single self-contained .glb.
import sys
import os
import shutil
import subprocess
import tempfile
import numpy as np
from PIL import Image
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
TRIPOSR_DIR = os.path.join(HERE, 'TripoSR')


def run_triposr(image_path: str, work_dir: str) -> tuple[str, str]:
    """Invoke TripoSR via its CLI; returns (mesh.obj path, texture.png path).
    TripoSR's own rembg is used (no --no-remove-bg flag) so the garment is
    cleanly isolated regardless of what the input background looks like."""
    os.makedirs(os.path.join(work_dir, '0'), exist_ok=True)
    cmd = [
        sys.executable, 'run.py', image_path,
        '--output-dir', work_dir,
        '--device', 'mps',
        '--bake-texture',
        '--texture-resolution', '1024',
        # IMPORTANT: do NOT pass --no-remove-bg. We want TripoSR's rembg to
        # cleanly separate the garment from any background pixels.
    ]
    print(f"[convert] running TripoSR (this takes ~60-90s on M1 Pro)…")
    subprocess.run(cmd, cwd=TRIPOSR_DIR, check=True)
    return (
        os.path.join(work_dir, '0', 'mesh.obj'),
        os.path.join(work_dir, '0', 'texture.png'),
    )


def _yup_correct(mesh: trimesh.Trimesh) -> bool:
    """Heuristic: is this mesh's collar/neck at +Y (top)?
    For most garments, the TOP narrows (collar) and the BOTTOM widens (hem).
    Compare the X-width of vertices in the top 10% of Y range vs the bottom 10%.
    Returns True if top is narrower than bottom (correct orientation)."""
    verts = np.array(mesh.vertices)
    y = verts[:, 1]
    y_min, y_max = float(y.min()), float(y.max())
    y_range = y_max - y_min
    if y_range < 1e-4:
        return True  # degenerate; trust it
    top = verts[y > y_max - y_range * 0.1]
    bot = verts[y < y_min + y_range * 0.1]
    if len(top) == 0 or len(bot) == 0:
        return True
    top_w = float(top[:, 0].max() - top[:, 0].min())
    bot_w = float(bot[:, 0].max() - bot[:, 0].min())
    return top_w < bot_w


def fix_orientation(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Rotate the mesh so its longest axis becomes +Y (vertical) AND the
    collar/neck end ends up at +Y (top).

    Two passes:
      1. Detect the longest principal axis and rotate it to align with Y.
      2. Check if the collar is at the top via vertex-width heuristic. If not,
         flip 180° around X to put it the right way up.
    """
    bb = mesh.bounds
    size = bb[1] - bb[0]
    longest = int(np.argmax(size))
    axes = 'XYZ'
    print(f"[convert] raw bbox    x={size[0]:.3f}  y={size[1]:.3f}  z={size[2]:.3f}")
    print(f"[convert] longest axis: {axes[longest]}")

    if longest == 0:
        R = trimesh.transformations.rotation_matrix(-np.pi / 2, [0, 0, 1])
        mesh.apply_transform(R)
        print("[convert] rotated -90° about Z to make X→Y")
    elif longest == 2:
        R = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
        mesh.apply_transform(R)
        print("[convert] rotated +90° about X to make Z→Y")
    else:
        print("[convert] Y is already the longest axis")

    # Flip if upside down
    if not _yup_correct(mesh):
        flip = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
        mesh.apply_transform(flip)
        print("[convert] flipped 180° about X — collar was at bottom")
    else:
        print("[convert] collar already at top (correct)")

    new_size = mesh.bounds[1] - mesh.bounds[0]
    print(f"[convert] final bbox  x={new_size[0]:.3f}  y={new_size[1]:.3f}  z={new_size[2]:.3f}")
    return mesh


def obj_to_glb(obj_path: str, texture_path: str, glb_path: str) -> None:
    """Bake the texture into a single self-contained .glb file, fixing orientation."""
    mesh = trimesh.load(obj_path, force='mesh')
    print(f"[convert] raw mesh: verts={len(mesh.vertices)}  faces={len(mesh.faces)}")

    mesh = fix_orientation(mesh)

    if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None and os.path.exists(texture_path):
        img = Image.open(texture_path)
        material = trimesh.visual.material.PBRMaterial(
            baseColorTexture=img,
            metallicFactor=0.0,
            roughnessFactor=0.85,
        )
        mesh.visual = trimesh.visual.TextureVisuals(uv=mesh.visual.uv, material=material)

    mesh.export(glb_path)
    print(f"[convert] wrote {glb_path} ({os.path.getsize(glb_path) / 1e6:.2f} MB)")


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: convert_png_to_glb.py path/to/garment.png [output_dir]")
    image_path = os.path.abspath(sys.argv[1])
    if not os.path.exists(image_path):
        sys.exit(f"no such file: {image_path}")
    out_dir = os.path.abspath(sys.argv[2]) if len(sys.argv) >= 3 else os.path.dirname(image_path)
    base = os.path.splitext(os.path.basename(image_path))[0]
    glb_path = os.path.join(out_dir, base + '.glb')

    with tempfile.TemporaryDirectory(prefix='triposr_') as work_dir:
        obj_path, tex_path = run_triposr(image_path, work_dir)
        obj_to_glb(obj_path, tex_path, glb_path)


if __name__ == '__main__':
    main()
