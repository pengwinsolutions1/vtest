# Local AI — image-to-3D conversion on Mac

Runs TripoSR (Stability AI, MIT license) on Apple Silicon MPS to convert a
single garment PNG into a textured GLB mesh. Used to populate the webapp's
catalog with real 3D garments without needing the NVIDIA vendor box.

## What's here

- `venv/` — Python 3.11 venv with PyTorch nightly (MPS) + TripoSR deps
- `TripoSR/` — cloned from https://github.com/VAST-AI-Research/TripoSR
- `convert_png_to_glb.py` — single-command CLI: PNG → GLB
- `ComfyUI/` — separate experiment; not part of the current pipeline

## Convert one garment

```bash
source /Users/yugandhar/VirtualTryOn/ai-local/venv/bin/activate
python /Users/yugandhar/VirtualTryOn/ai-local/convert_png_to_glb.py \
  /Users/yugandhar/VirtualTryOn/seed-garments/my-new-garment.png
```

The script:
1. Runs TripoSR on MPS (~60s on M1 Pro 16 GB)
2. Bakes a 1024×1024 texture from the PNG
3. Converts OBJ + texture → single self-contained `.glb` next to the input

The resulting `.glb` sits next to its source `.png` in `seed-garments/`. To get
it into the catalog:

```bash
cd /Users/yugandhar/VirtualTryOn/webapp
rm -f data/tryon.db*
# Server reseeds on next /api/garments hit; the GLB is now paired with the PNG.
```

## Performance & quality on M1 Pro 16 GB

| Step | Time |
|---|---|
| First model load (cold) | ~5s |
| Model forward pass | ~3s |
| Mesh extraction (marching cubes) | ~10s |
| Texture baking @ 1024px | ~37s |
| OBJ → GLB | <1s |
| **Total** | **~55-65s per garment** |

Output: ~100-130k verts, 4-6 MB GLB file with embedded PBR texture.

**Quality:** moderate. TripoSR is the smallest/oldest of the viable image-to-3D
models — production should use Trellis or Hunyuan3D-2 on the vendor box for
higher fidelity. TripoSR is the only model that fits + runs cleanly on Mac.

## Known issues / fixes

- **`transformers` version conflict.** TripoSR requires `transformers==4.35.0`.
  This venv was downgraded from `5.10.2` to make it work. If you re-install
  ComfyUI here it'll upgrade transformers and break TripoSR. Keep these
  separate or pin the version.
- **`/0/` subdir must exist.** TripoSR's `run.py` writes to `{output}/0/mesh.obj`
  but doesn't create the dir. The CLI helper handles this; if you run `run.py`
  directly, `mkdir -p output/0` first.
- **`xatlas==0.0.9` has no arm64 wheel.** Install plain `xatlas` (latest, 0.0.11).
- **`torchmcubes` needs pybind11 at CMake time.** `pip install pybind11` then
  `CMAKE_ARGS="-Dpybind11_DIR=$(python -c 'import pybind11; print(pybind11.get_cmake_dir())')" pip install --no-build-isolation git+https://github.com/tatsy/torchmcubes.git`.

## What I tried for 360° multi-view input (and why it didn't fit on Mac)

The user provided 200 frames of a uniform 360° rotation (`/garmets/women-dress/`).
The proper way to use this kind of input is **multi-view photogrammetry or
Gaussian splatting** — far higher quality than single-image TripoSR. Both
attempts failed on Mac for fundamental reasons:

### COLMAP photogrammetry
- Installed clean (`brew install colmap`, no CUDA).
- Ran SIFT feature extraction on 50 subsampled frames — only **~500–700
  features per frame** because the matte black dress has minimal surface
  texture.
- Sequential matcher produced **zero cross-frame matches**. SfM mapper
  exited with "Failed to create any sparse model".
- COLMAP's CPU MVS would have been hours-long anyway without CUDA — even
  if SfM had succeeded.
- Verdict: structural — texture-poor garments are not solvable with
  feature-based photogrammetry. Patterns and embroidery would help; plain
  matte cloth won't.

### Nerfstudio + splatfacto (Gaussian splatting)
- Installed `nerfstudio` + `gsplat` 1.4 (claims MPS support).
- Generated `transforms.json` with the known 360° camera poses (skipping
  SfM since the rig geometry is regular).
- Splatfacto crashed on MPS at the first forward pass:
  `K = camera.get_intrinsics_matrices().cuda()` — hardcoded CUDA call in
  `splatfacto.py:579`, doesn't honor `--machine.device-type mps`.
- Patching the .cuda() calls is doable but gsplat's rasterizer on MPS
  falls back to pure-Python; expected training time 100× slower than CUDA,
  i.e. **days for 30k iterations**.
- Verdict: hardcoded CUDA + unusable gsplat MPS perf. Mac is not the
  platform for splatfacto in 2026.

### What the vendor box should run
With 16 GB NVIDIA, the right pipeline is:
1. **Nerfstudio + splatfacto** (or **2DGS/SuGaR** for direct mesh output)
   on the 200 frames with the known camera poses.
2. Training ~5–15 min on RTX 4070 Ti Super.
3. Mesh extraction via Open3D Poisson reconstruction or 2DGS native mesh.
4. Texture bake from the original RGB frames.

The resulting mesh would be photogrammetry-grade (real captured geometry
from 360° of views), not single-image guesswork like TripoSR.

## When the NVIDIA vendor box comes online

This Mac path becomes irrelevant. The vendor box runs Trellis (or Hunyuan3D-2)
which produces better meshes, faster (~10-20s per garment). The webapp's
LocalBackend talks to the vendor box over HTTP per
`webapp/docs/vendor-backend-contract.md` → `POST /garment/convert-to-3d`.

This Mac install can stay as a dev playground or be deleted (~10 GB total).
