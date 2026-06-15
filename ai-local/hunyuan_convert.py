#!/usr/bin/env python3
# Generate a 3D mesh from a single garment image using Hunyuan3D-2mini.
# Runs on Mac MPS. Slow but high quality (best single-image-to-3D model
# we have running locally as of 2026).
#
# Usage:
#   python hunyuan_convert.py path/to/garment.png path/to/output.glb [--steps 30]
#
# LICENSE NOTE: Hunyuan3D-2 is Tencent Non-Commercial. Fine for testing/demos;
# for paid production swap to a commercial-licensed model (Trellis MIT etc.).
import sys
import os
import argparse
import time
from PIL import Image

# Hunyuan API
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Hunyuan3D-2'))
from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline


def main():
    p = argparse.ArgumentParser()
    p.add_argument('image_path')
    p.add_argument('output_glb')
    p.add_argument('--steps', type=int, default=30, help='diffusion steps (more=better+slower; 30=standard)')
    p.add_argument('--guidance', type=float, default=5.0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='mps')
    args = p.parse_args()

    print(f"[hunyuan] image: {args.image_path}")
    print(f"[hunyuan] output: {args.output_glb}")
    print(f"[hunyuan] device: {args.device}  steps: {args.steps}")

    # Background-remove if needed
    img = Image.open(args.image_path).convert('RGBA')
    has_alpha = any(p < 250 for p in img.split()[3].getdata()) if img.mode == 'RGBA' else False
    if not has_alpha:
        print("[hunyuan] running rembg (input has no alpha)")
        bg = BackgroundRemover()
        img = bg(img)

    print("[hunyuan] loading shape pipeline (uses cached weights after first run)…")
    t0 = time.time()
    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        'tencent/Hunyuan3D-2mini',
        subfolder='hunyuan3d-dit-v2-mini',
        device=args.device,
    )
    print(f"[hunyuan] pipeline ready in {time.time() - t0:.1f}s")

    print("[hunyuan] generating mesh…")
    t0 = time.time()
    mesh = pipe(
        image=img,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        generator=None,  # could seed; deterministic when set
    )[0]
    print(f"[hunyuan] mesh in {time.time() - t0:.1f}s  verts={len(mesh.vertices)}  faces={len(mesh.faces)}")

    # Export. Hunyuan returns a trimesh.Trimesh.
    mesh.export(args.output_glb)
    print(f"[hunyuan] wrote {args.output_glb}  ({os.path.getsize(args.output_glb)/1e6:.2f} MB)")


if __name__ == '__main__':
    main()
