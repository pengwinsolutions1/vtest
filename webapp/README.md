# Virtual Try-On — webapp

Live AR virtual trial room. Admin uploads a garment as **2 PNGs (front + back)**;
customers stand in front of a webcam, pick a garment, see it draped on themselves
in real time at 30fps.

## Where it runs

| Layer | Where | What |
|---|---|---|
| Webapp (Next.js + UI + admin + catalog DB) | **vendor's box** (also dev Mac) | UI, garment catalog, admin upload, GLB serving |
| 2-view → GLB builder ([../ai-local/build_2view_glb.py](../ai-local/build_2view_glb.py)) | **vendor's box** | Procedural mesh + cylindrical UV with front+back textures. ~5s per garment. No AI needed. |
| Live AR (MediaPipe + Three.js) | **customer's browser** | Webcam, pose tracking, draping the GLB on the body |

The whole product runs **locally on the vendor box**. Customer photos never leave the LAN.
No cloud APIs. No per-customer cost.

## How the conversion works

```
admin uploads front.png + back.png
                │
                ▼
[build_hunyuan_glb.py: Hunyuan3D-2mini for real 3D geometry +
 cylindrical UV mapping with [front | back] panorama as texture]
                │
                ▼
       garment.glb (~1 MB, 30k tris, real photos + real 3D shape)
                │
                ▼
       added to SQLite catalog
                │
                ▼
   customer's live AR loads it,
   drapes it on their pose at 30fps
```

The **geometry** comes from Hunyuan3D-2mini (single-image AI shape) — real
sleeves, real neckline, real garment-shape curvature. The **texture** is the
admin's actual front and back product photos wrapped cylindrically around the
mesh, blended at the side seams. No texture hallucination; what you upload is
what shows on the garment.

**Time**: ~8 minutes per garment on M1 Pro 16 GB (Hunyuan diffusion is the
bulk). The fallback `build_2view_glb.py` is still in the repo if you want a
~5-second procedural alternative — see `lib/admin/garments/route.ts` to
switch builders.

**License**: Hunyuan3D-2 is Tencent Non-Commercial. For paid production
deploys, swap the shape backend to TRELLIS (MIT) — same pipeline interface,
just change `BUILDER_PYTHON` and `BUILDER_PY` in the admin route.

## Honest ceiling on Mac vs the vendor box

| Step | Mac M1 Pro 16 GB | Vendor NVIDIA box (16 GB) |
|---|---|---|
| Hunyuan **shape** generation | ✅ Works, ~7 min/garment | ✅ ~30s/garment |
| Hunyuan **paint** (AI-imagined back/sides) | ❌ Hardcoded `.cuda()` everywhere + would be 100× slow on MPS | ✅ ~2 min/garment, photoreal back |
| Cylindrical UV with **real product photos** | ✅ Works | ✅ Works (faster) |
| Single-image back synthesis (blurred-fabric fallback) | ✅ Works | ✅ Works |

**What this means in practice**:
- On the **Mac dev box**, when admin uploads only a front PNG, the back of the
  GLB shows a blurred dominant-color version of the front. Looks like the
  fabric, doesn't look like a backwards garment.
- On the **vendor NVIDIA box**, the same upload runs Hunyuan paint and the
  back is AI-inferred to match the front (proper neckline shape, no buttons
  facing wrong way, etc.). This is what gets shipped to customers.
- Either box: admin can upload BOTH front + back PNGs, and both halves of the
  mesh use the real photos. This is the highest-quality path on both.

The current `build_hunyuan_glb.py` is platform-aware: it tries Hunyuan paint
first on platforms that can run it, falls back to blurred-fabric back on
Mac. (TODO: this detection isn't implemented yet — currently always uses the
fallback. Wire it up when shipping the vendor box.)

## Admin workflow

1. Open http://localhost:3000/admin
2. Drop front PNG, drop back PNG (works with raw product shots; rembg handles backgrounds)
3. Name + gender + category
4. Submit → ~5 seconds later, the garment is live in the customer catalog

## Customer workflow

1. Open http://localhost:3000 (the kiosk URL)
2. Browser asks for camera permission
3. MediaPipe Pose locks onto the user
4. Pick a garment from the bottom strip
5. Garment drapes on the user's torso in real time
6. Switch garments freely

## Architecture file map

```
webapp/
├── app/
│   ├── page.tsx                       Live AR customer UI (Three.js + MediaPipe)
│   ├── admin/page.tsx                 Admin upload UI
│   ├── api/
│   │   ├── garments/route.ts          GET catalog (public)
│   │   ├── admin/garments/route.ts    POST new garment (admin)
│   │   ├── tryon/route.ts             [parked] diffusion try-on entry
│   │   └── jobs/[id]/route.ts         [parked] diffusion job polling
│   ├── uploads/garments/[name]/route.ts  serves PNG/GLB files
│   └── preview/[id]/page.tsx          standalone GLB viewer (debugging)
├── lib/
│   ├── db.ts                          SQLite schema + garment accessors
│   ├── ai-backend.ts                  pluggable backend (mock | local | replicate)
│   ├── backends/                       3 implementations of the above
│   ├── uploads.ts                     file save helper
│   └── orchestrator.ts                diffusion job state machine [parked]
└── data/
    ├── tryon.db                       SQLite (garments + jobs tables)
    └── uploads/garments/              PNG + GLB files

../ai-local/
├── build_2view_glb.py                 ← THE PRODUCTION GARMENT BUILDER
├── convert_png_to_glb.py              [parked] single-image TripoSR fallback
├── build_360_glb.py                   [parked] N-frame panoramic builder
├── brush/                             [parked] Gaussian Splatting tool
├── TripoSR/                           [parked] image-to-3D model
└── venv/                              Python deps
```

## What's "parked"

We tried several heavier approaches that didn't earn their complexity for this
use case. They're left in the repo as dormant code, in case the requirements
change:

- **TripoSR single-image AI** — chunky meshes, hallucinated back. Worse than 2-PNG.
- **360-frame photogrammetry** (COLMAP) — fails on matte garments. Documented in `../ai-local/README.md`.
- **Brush Gaussian Splatting** — works on Mac but is overkill for 2-view input. Output meshes were noisier than the clean procedural approach.
- **Diffusion AI try-on** (Replicate IDM-VTON + Wan 2.1) — produces clothes-2.mp4-style photoreal video but takes 60-120s and isn't local. Backend code is in `lib/backends/replicate-backend.ts` if you ever want a "premium HQ video" mode.

## Local dev

```bash
cd webapp
cp .env.example .env.local   # AI_BACKEND=mock is fine for this product
npm install
npm run dev                  # http://localhost:3000
# Admin: http://localhost:3000/admin
```

Server scripts in [scripts/](scripts/) (`start.sh`, `stop.sh`) for production
mode (`next build` + `next start`).

## Vendor box deployment

See [docs/vendor-backend-contract.md](docs/vendor-backend-contract.md) for the HTTP
contract if you ever want to split the AI workloads onto a separate machine.
For the current product (live AR + 2-view procedural builder), everything runs in
one Next.js process — no separate AI service needed.
