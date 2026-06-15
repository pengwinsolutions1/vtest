# Vendor AI Backend HTTP Contract

The webapp talks to **your** on-prem AI server over plain HTTP. As long as your
server exposes the three endpoints below, the webapp doesn't care how you
generate the images and video — ComfyUI workflows, raw PyTorch scripts,
Triton Inference Server, whatever.

This contract is intentionally tiny so any small team can implement it in a
day. Reference implementation hints at the bottom.

## Endpoints

### `POST /tryon/still`
Start a try-on still-image generation.

**Request body** (`application/json`):
```json
{
  "selfie_url":  "https://store.example.com/uploads/selfie_abc.jpg",
  "garment_url": "https://store.example.com/uploads/garment_xyz.png"
}
```

Both URLs are reachable from your server. Download them, run your try-on model
(e.g. CatVTON, IDM-VTON, OOTDiffusion), and start generation. Don't block on
the result — return immediately with a job id.

**Response** (`200`):
```json
{ "prediction_id": "some-opaque-string" }
```

The webapp will poll `GET /predictions/<prediction_id>` to track progress.

---

### `POST /tryon/video`
Animate a still image into a 5-second video.

**Request body** (`application/json`):
```json
{
  "still_url": "https://your-box.local/outputs/still_abc.png"
}
```

The `still_url` is hosted by you (returned from a prior `/tryon/still` call).
Run your image-to-video model (e.g. Wan 2.1, LTX-Video, CogVideoX) and return
a new prediction id.

**Response** (`200`):
```json
{ "prediction_id": "another-opaque-string" }
```

---

### `POST /garment/convert-to-3d`
Convert one or more images of a garment (front / side / back / 360°) into a
rigged-or-static `.glb` mesh.

**Request body** (`application/json`):
```json
{
  "image_urls": [
    "https://store.example.com/uploads/garment_front.jpg",
    "https://store.example.com/uploads/garment_side.jpg",
    "https://store.example.com/uploads/garment_back.jpg"
  ],
  "category": "top"
}
```

`category` is one of `top` / `bottom` / `dress`. The model uses it as a prior
for mesh shape (e.g. dress = vertical drape, top = torso wrap).

**Response** (`200`):
```json
{ "prediction_id": "some-opaque-string" }
```

The webapp polls `GET /predictions/<prediction_id>` for the result. When
status flips to `succeeded`, the `output_url` field points at the `.glb` file
on the vendor's storage. The webapp downloads it, saves it to
`data/uploads/garments/<garment_id>.glb`, and flips the catalog row's
`glb_status` to `ready`.

**Recommended models (16 GB VRAM):**
- **Trellis** (Microsoft, MIT license) — ~10 GB, 30-60s/garment, best quality/VRAM tradeoff
- **Hunyuan3D-2 mini** (Tencent) — ~12 GB, 60-90s/garment, slightly higher fidelity
- **TripoSR** — ~6 GB, 10-15s/garment, lowest quality but fastest

---

### `GET /predictions/{prediction_id}`
Poll a prediction's status.

**Response** (`200`):
```json
// while running
{ "status": "pending" }

// when done
{ "status": "succeeded", "output_url": "https://your-box.local/outputs/result_abc.mp4" }

// when broken
{ "status": "failed", "error": "OOM during VAE decode" }
```

Status must be one of: `"pending"`, `"succeeded"`, `"failed"`.

`output_url` is the URL to the result image or video on YOUR storage. The
webapp will fetch and display it to the user. Make sure it's reachable from
the webapp's host (usually same LAN).

---

## Auth (optional)

If you set `VENDOR_API_KEY` in the webapp's environment, every request will
include:

```
Authorization: Bearer <your-key>
```

Reject requests without it. If you don't set the env var, no auth header is
sent — fine for a single-LAN deployment behind a firewall, not fine for
internet-exposed servers.

---

## Recommended models

| Step | Model | Quality | VRAM needed |
|---|---|---|---|
| Try-on still | **CatVTON** | Good | ~10 GB |
| Try-on still | **IDM-VTON** | Best (non-commercial) | ~16 GB |
| Try-on still | **OOTDiffusion** | Good | ~12 GB |
| Image → video | **Wan 2.1 I2V 14B** | Best | ~24 GB |
| Image → video | **LTX-Video 2B** | Good (fast) | ~8 GB |
| Image → video | **CogVideoX 5B I2V** | OK | ~16 GB |

For full quality (`Wan 2.1` + `IDM-VTON`), you need a 24 GB+ GPU. Used RTX
3090/4090 boxes start around $1500.

---

## Reference implementation skeleton (Python + FastAPI + ComfyUI)

```python
# server.py — minimal vendor backend
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import uuid, asyncio
import comfyui_client  # your wrapper around ComfyUI's /prompt API

app = FastAPI()
jobs = {}  # prediction_id → {status, output_url, error}

class StillIn(BaseModel):
    selfie_url: str
    garment_url: str

class VideoIn(BaseModel):
    still_url: str

@app.post("/tryon/still")
async def still(body: StillIn):
    pid = str(uuid.uuid4())
    jobs[pid] = {"status": "pending"}
    asyncio.create_task(run_still(pid, body.selfie_url, body.garment_url))
    return {"prediction_id": pid}

@app.post("/tryon/video")
async def video(body: VideoIn):
    pid = str(uuid.uuid4())
    jobs[pid] = {"status": "pending"}
    asyncio.create_task(run_video(pid, body.still_url))
    return {"prediction_id": pid}

@app.get("/predictions/{pid}")
def poll(pid: str):
    if pid not in jobs:
        raise HTTPException(404)
    return jobs[pid]

async def run_still(pid, selfie_url, garment_url):
    try:
        out = await comfyui_client.run_catvton(selfie_url, garment_url)
        jobs[pid] = {"status": "succeeded", "output_url": out}
    except Exception as e:
        jobs[pid] = {"status": "failed", "error": str(e)}

async def run_video(pid, still_url):
    try:
        out = await comfyui_client.run_wan_i2v(still_url)
        jobs[pid] = {"status": "succeeded", "output_url": out}
    except Exception as e:
        jobs[pid] = {"status": "failed", "error": str(e)}
```

Run with: `uvicorn server:app --host 0.0.0.0 --port 8000`

Then in the webapp's `.env.local`:
```
AI_BACKEND=local
LOCAL_AI_URL=http://<vendor-box-ip>:8000
```
