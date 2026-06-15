# Split deployment: Mac (UI) ↔ NVIDIA box (AI)

This is the topology we run in dev:

```
┌────────────────────────────────┐         LAN          ┌─────────────────────────────────┐
│  Mac (this machine)            │ ◄──────────────────► │  NVIDIA PC                      │
│  192.168.0.8                   │                       │  <NVIDIA_IP — fill in>          │
│                                 │                       │                                  │
│  - Electron desktop app        │                       │  - vendor-box service           │
│  - Next.js webapp on :3000     │                       │  - uvicorn server:app :8000     │
│  - Catalog + selfie uploads    │                       │  - IDM-VTON, Wan 2.1, DM-VTON  │
└────────────────────────────────┘                       └─────────────────────────────────┘
        ▲                                                              ▲
        │ webapp serves /uploads/garments/*.png      AI fetches selfies + garments from
        │ webapp serves /uploads/selfies/*.jpg       http://192.168.0.8:3000/uploads/...
        │
        │ webapp's POST /api/tryon calls AI:        AI returns prediction_ids, then
        │ POST http://<NVIDIA_IP>:8000/tryon/still   results when polled
```

## Mac side — webapp config

`webapp/.env.local`:
```
AI_BACKEND=local
LOCAL_AI_URL=http://<NVIDIA_IP>:8000
PUBLIC_URL=http://192.168.0.8:3000
```

`PUBLIC_URL` is critical: the NVIDIA box fetches selfies and garment images from this URL. If you leave it as `localhost:3000`, the AI box can't reach the files.

Restart the webapp after editing `.env.local`:
```bash
cd webapp
# kill the existing next dev (it was started earlier)
pkill -f "next dev" || true
npm run electron:dev
```

## NVIDIA side — vendor-box service

On the NVIDIA box (Ubuntu 22.04 + CUDA 12.1):

```bash
# 1. Clone
git clone https://github.com/pengwinsolutions1/vtest.git virtualtryon
cd virtualtryon/vendor-box

# 2. Python env
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt   # ~3-5 min, CUDA wheels

# 3. (Eventually) download model weights — ~25 GB, one-time
python scripts/download_models.py

# 4. Start the service (bind 0.0.0.0 so Mac can reach it)
uvicorn server:app --host 0.0.0.0 --port 8000 --workers 1
```

## Quick connectivity test (before models are wired up)

You can test the network plumbing right now, even before the AI models are integrated:

```bash
# On the NVIDIA box:
curl http://localhost:8000/healthz
# → {"ok": true, "cuda": true, "snapshot": false, "live": false, ...}
# snapshot/live are false because the loader TODOs aren't implemented yet

# On the Mac:
curl http://<NVIDIA_IP>:8000/healthz
# → same response
# If this works, the LAN path is good. If "connection refused" → check
# (a) the NVIDIA service is bound to 0.0.0.0 not 127.0.0.1
# (b) the NVIDIA box's firewall isn't blocking port 8000
# (c) both machines are on the same network (same subnet)
```

```bash
# From the NVIDIA box, verify it can fetch from the Mac webapp:
curl -I http://192.168.0.8:3000/api/garments
# → HTTP/1.1 200 OK
# If this fails, your Mac's firewall is blocking inbound 3000.
# Fix: System Settings → Network → Firewall → allow Node/Electron incoming.
```

## What happens after both sides are up

1. Customer (on Mac, in the Electron kiosk window) taps a dress
2. Mac captures webcam selfie, POSTs to `Mac:3000/api/tryon` (same machine)
3. Mac's Next.js writes selfie to `webapp/data/uploads/selfies/<id>.jpg`
4. Mac's `LocalBackend` POSTs to `NVIDIA:8000/tryon/still` with two URLs:
   - `selfie_url = http://192.168.0.8:3000/uploads/selfies/<id>.jpg`
   - `garment_url = http://192.168.0.8:3000/uploads/garments/<garment>.png`
5. NVIDIA service downloads both, runs IDM-VTON, saves result to `NVIDIA:/results/<id>.png`
6. NVIDIA responds with `prediction_id`
7. Mac polls `NVIDIA:8000/predictions/<prediction_id>` until succeeded
8. Mac shows the result image to the customer

LIVE mode (WebSocket) lands on top of this once the WebSocket client is wired on the Mac side — see the `/ws/live` endpoint in `vendor-box/server.py`. The Electron renderer connects WS directly to the NVIDIA box (does NOT proxy through Mac's Next.js).

## First-deploy checklist

- [ ] NVIDIA box has internet to download model weights
- [ ] NVIDIA box has CUDA 12.1 + Python 3.11
- [ ] NVIDIA box's port 8000 is reachable from the Mac (firewall allows it)
- [ ] Mac's port 3000 is reachable from the NVIDIA box (firewall allows it)
- [ ] You know the NVIDIA box's LAN IP (`ip addr` or `hostname -I` on Linux)
- [ ] You've set `LOCAL_AI_URL` and `PUBLIC_URL` in `webapp/.env.local`
- [ ] You've implemented the loader TODOs in `idm_vton_loader.py`, `wan21_loader.py`, `dm_vton_loader.py` (or are OK testing with `/healthz` first)
