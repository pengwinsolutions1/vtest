# Virtual Try-On

AR clothing kiosk system — body-tracked 3D garment overlay running in the browser.

## What's in this repo

```
.
├── production/          ← the v1 production stack (what you run)
│   ├── kiosk/           Three.js + MediaPipe Pose client (the user-facing screen)
│   ├── backend/         Node + Express + SQLite (API + asset hosting)
│   ├── admin/           Plain-HTML admin SPA (upload garments, manage kiosks)
│   ├── garment-pipeline/ PNG → rigged GLB pipeline (Python + Blender)
│   ├── scripts/         bootstrap-this-mac.sh, backup.sh, restore.sh
│   ├── infra/aws/       Terraform for future AWS deploy
│   └── docs/            ROADMAP.md, DEPLOY.md
├── tryon-web/           ← v1 prototype (2D-overlay demo, kept for reference)
├── LENS_STUDIO_PLAYBOOK.md  Strategic options doc (Snap, Wanna, custom paths)
├── denim-jacket-sleeveless-with-white-t-shirt/  Original unrigged FBX asset
└── *.png                Source garment images
```

## Running locally on this Mac

```bash
# One-time setup
cd production
./scripts/bootstrap-this-mac.sh

# Already done — server runs from a launchd job on every login.
# Stop:  launchctl unload ~/Library/LaunchAgents/com.tryon.local.plist
# Start: launchctl load ~/Library/LaunchAgents/com.tryon.local.plist
```

URLs (replace `192.168.x.x` with `ipconfig getifaddr en0`):

- Kiosk: `https://192.168.x.x:3443/`
- Admin: `https://192.168.x.x:3443/admin/`  (`admin@tryon.local` / `demo-pass-123`)
- Catalog API: `https://192.168.x.x:3443/catalog`

## Adding garments

```bash
cd production/garment-pipeline
./process.sh /path/to/shirt.png
# outputs: output/shirt.glb + output/shirt.png (thumbnail)
```

Then upload via the admin panel.

## Deploying to AWS

See [production/docs/DEPLOY.md](production/docs/DEPLOY.md).

## History

- v0 — Unity 6 / HDRP attempt (removed, didn't reach production quality)
- v1 (live demo) — 2D PNG overlay in browser via MediaPipe → `tryon-web/`
- **v1.x (current)** — Three.js + rigged 3D GLBs + multi-kiosk admin → `production/`
