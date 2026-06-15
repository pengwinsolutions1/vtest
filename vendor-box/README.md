# Vendor-box AI service

This is the **production AI backend** that runs on the NVIDIA 16 GB GPU box installed at the retail store. It serves two modes:

| Mode | What it does | Latency | Use case |
|---|---|---|---|
| `live`     | Real-time pose-aware garment overlay on the video stream | ~50-70 ms / frame (~15-20 fps) | LIVE try-on — customer sees themselves wearing it as they move |
| `snapshot` | Single photoreal still + short video of customer wearing it | 15-25 s still, +25-60 s video | Premium "wow" mode — kiosk takes a photo, generates a photoreal result |

The Electron webapp (`/webapp`) talks to this service via HTTP and WebSocket over `localhost`. Customer images **never leave the LAN**.

## Hardware target

- NVIDIA RTX 4080 / A4000 / A5000 / similar — minimum 16 GB VRAM
- Ubuntu 22.04 LTS
- CUDA 12.1 (matches the PyTorch wheels we pin)
- 32 GB system RAM recommended
- Local-LAN webcam access from kiosk PC (the kiosk's webapp captures, this service warps/generates)

## What runs where

```
┌─────────────────────────────────────────────────────────┐
│  Kiosk PC (could be same physical box, browser/Electron) │
│                                                          │
│   Electron app → /api/tryon       (Next.js webapp)      │
│                                                          │
└──────────────┬──────────────────────────────┬───────────┘
               │ HTTP                         │ WebSocket
               ▼                              ▼
┌─────────────────────────────────┐   ┌──────────────────┐
│  vendor-box / server.py         │   │  /ws/live        │
│  (FastAPI + Uvicorn)            │   │   (frame stream) │
│                                  │   └──────────────────┘
│  Endpoints:                      │
│    POST /tryon/still     ─►  IDM-VTON (photoreal)
│    POST /tryon/video     ─►  Wan 2.1 i2v (short video)
│    GET  /predictions/{id}
│    WS   /ws/live         ─►  DM-VTON streaming
│    GET  /healthz         ─►  reports {live: bool, snapshot: bool}
└─────────────────────────────────┘
```

## Models used

- **IDM-VTON** (HuggingFace `yisol/IDM-VTON`) — photoreal still try-on. License: non-commercial; swap for FashionTryOn or similar before paid launch.
- **Wan 2.1 i2v** (Alibaba `Wan-Video/Wan2.1`) — image-to-video animation. ~24 fps, 2-3s clip from one image.
- **DM-VTON** (`KiseKloset/DM-VTON`) — distilled real-time VTON, ~50 ms / frame on RTX 4080. License: research-only.

Model weights are downloaded on first boot to `/var/lib/vendorbox/models/` (~25 GB total). The systemd service will block on first launch while they download.

## Quick-start (after vendor box arrives)

```bash
# On the vendor box (Ubuntu 22.04 LTS, CUDA 12.1 already installed)
cd /opt
sudo git clone <this-repo> virtualtryon
cd virtualtryon/vendor-box

# Install Python deps in a venv. Any Python 3.10+ works (3.11 recommended).
# Check what you have first:
python3 --version
# If 3.10/3.11/3.12 → use `python3` below. If 3.9 or older, install 3.11:
#   sudo apt install -y python3.11 python3.11-venv python3.11-dev

python3 -m venv venv      # or python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Pre-download model weights (one-time, ~25 GB)
python scripts/download_models.py

# Run the service
uvicorn server:app --host 0.0.0.0 --port 8000

# Verify
curl http://localhost:8000/healthz
# → {"live": true, "snapshot": true, "model_versions": {...}}
```

## Point the webapp at it

In `/webapp/.env.local`:

```
AI_BACKEND=local
LOCAL_AI_URL=http://localhost:8000
PUBLIC_URL=http://localhost:3000
```

Restart the webapp. The kiosk customer page will now use the local service. When `/healthz` reports `live: true`, the customer page also offers the LIVE mode (overlay on the streamed webcam) on top of the snapshot photo-booth.

## Why we don't ship Mac dev with these models

These models are CUDA-hardcoded. We tried to patch Hunyuan-Paint's CUDA calls for MPS and gave up after the loader bottomed out at `Torch not compiled with CUDA enabled`. IDM-VTON and Wan 2.1 have the same issue. DM-VTON requires TensorRT for real-time inference — not available on macOS.

The Mac is **dev / UX iteration only**. The shipping target is the vendor box.
