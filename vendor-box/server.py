"""Vendor-box AI service.

FastAPI + Uvicorn. Two modes:
  - snapshot  → /tryon/still + /tryon/video (IDM-VTON, Wan 2.1)
  - live      → /ws/live (DM-VTON streaming, ~50ms/frame on RTX 4080)

This file is the production service that runs on the NVIDIA box. It will NOT
run on Mac dev because every model used here is CUDA-only. The Mac dev path
uses the mock backend (or Replicate for real cloud testing).

Boot order:
  1. uvicorn imports this module → __init__ pipelines as None
  2. lifespan() warms up the GPU pipelines (lazy import so MPS dev imports
     don't crash on torch CUDA checks)
  3. service ready, /healthz reports which modes are available
"""
from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

log = logging.getLogger("vendor-box")
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

# Default to a project-local data dir (writable without sudo). Override with
# VENDORBOX_MODELS_DIR=/var/lib/vendorbox/models for a system-wide install
# where you've pre-created the dir and chowned it to the service user.
_DEFAULT_DATA_ROOT = Path(__file__).resolve().parent / "data"
MODELS_DIR = Path(os.environ.get("VENDORBOX_MODELS_DIR", str(_DEFAULT_DATA_ROOT / "models")))
RESULTS_DIR = Path(os.environ.get("VENDORBOX_RESULTS_DIR", str(_DEFAULT_DATA_ROOT / "results")))


# ============================================================================
# Pipeline holders — populated by lifespan() on startup
# ============================================================================
@dataclass
class Pipelines:
    """Lazy-loaded pipelines. None = unavailable on this box."""
    idm_vton: Any = None
    wan_i2v: Any = None
    dm_vton: Any = None
    device: str = "cpu"
    cuda_available: bool = False


PIPES = Pipelines()


def _load_pipelines() -> None:
    """Import torch + diffusers + load weights. Called from lifespan."""
    global PIPES
    try:
        import torch
        PIPES.cuda_available = torch.cuda.is_available()
        if not PIPES.cuda_available:
            log.warning("CUDA not available — service will start but no inference modes work")
            return
        PIPES.device = "cuda"

        def _try_load(name: str, loader_fn):
            """Run a loader, log appropriately based on exception type.

            NotImplementedError → expected TODO state, log as WARNING (the
            user hasn't wired the real model integration yet). Anything else
            → genuine failure, log as ERROR with stack trace.
            """
            try:
                return loader_fn()
            except NotImplementedError as e:
                log.warning("%s: not implemented yet (TODO) — %s", name, e)
            except Exception as e:
                log.exception("%s load failed: %s", name, e)
            return None

        # ── IDM-VTON (snapshot still) ────────────────────────────────
        log.info("loading IDM-VTON…")
        PIPES.idm_vton = _try_load("IDM-VTON", lambda: (
            __import__("idm_vton_loader").load_idm_vton(MODELS_DIR / "idm-vton", device="cuda")
        ))
        if PIPES.idm_vton: log.info("IDM-VTON ready")

        # ── Wan 2.1 i2v (snapshot video) ─────────────────────────────
        log.info("loading Wan 2.1 i2v…")
        PIPES.wan_i2v = _try_load("Wan 2.1", lambda: (
            __import__("wan21_loader").load_wan_i2v(MODELS_DIR / "wan2.1", device="cuda")
        ))
        if PIPES.wan_i2v: log.info("Wan 2.1 i2v ready")

        # ── DM-VTON (live streaming) ─────────────────────────────────
        log.info("loading DM-VTON (TensorRT)…")
        PIPES.dm_vton = _try_load("DM-VTON", lambda: (
            __import__("dm_vton_loader").load_dm_vton_trt(MODELS_DIR / "dm-vton")
        ))
        if PIPES.dm_vton: log.info("DM-VTON live pipeline ready")

    except ImportError as e:
        log.error("Required Python deps missing: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("vendor-box starting up")
    log.info("  models dir  = %s", MODELS_DIR)
    log.info("  results dir = %s", RESULTS_DIR)
    # Create the dirs we'll need. If these fail with permission errors, the
    # user is pointing the service at a path their account can't write to.
    try:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        log.error(
            "Cannot create data dirs: %s. Set VENDORBOX_MODELS_DIR and "
            "VENDORBOX_RESULTS_DIR to a path your account can write to, "
            "or pre-create the system paths and chown them to this user.", e,
        )
        raise
    _load_pipelines()
    log.info(
        "ready: cuda=%s idm_vton=%s wan_i2v=%s dm_vton=%s",
        PIPES.cuda_available,
        PIPES.idm_vton is not None,
        PIPES.wan_i2v is not None,
        PIPES.dm_vton is not None,
    )
    yield
    log.info("vendor-box shutting down")


app = FastAPI(title="VirtualTryOn vendor-box service", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


# ============================================================================
# Snapshot mode — HTTP, matches the contract the webapp's LocalBackend uses
# ============================================================================
# In-memory job registry. The webapp polls /predictions/{id} until terminal.
# Single-customer-at-a-time deployment → no need for Redis/DB.
@dataclass
class Job:
    id: str
    kind: Literal["still", "video"]
    status: Literal["pending", "succeeded", "failed"] = "pending"
    output_url: str | None = None
    error: str | None = None
    started_at: float = field(default_factory=time.time)


JOBS: dict[str, Job] = {}


class StillReq(BaseModel):
    selfie_url: str
    garment_url: str
    category: Literal["top", "bottom", "dress"]


class VideoReq(BaseModel):
    still_url: str


@app.post("/tryon/still")
async def tryon_still(req: StillReq) -> dict[str, str]:
    """Kick off an IDM-VTON inference. Returns a prediction_id to poll."""
    if PIPES.idm_vton is None:
        raise HTTPException(503, "IDM-VTON not loaded on this box")
    job = Job(id=f"still_{uuid.uuid4().hex}", kind="still")
    JOBS[job.id] = job
    # Fire-and-forget — runs on the GPU executor thread
    asyncio.create_task(_run_still(job, req))
    return {"prediction_id": job.id}


async def _run_still(job: Job, req: StillReq) -> None:
    try:
        # Download inputs from the webapp (same LAN)
        import httpx
        from PIL import Image
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.get(req.selfie_url); r.raise_for_status()
            selfie = Image.open(io.BytesIO(r.content)).convert("RGB")
            r = await client.get(req.garment_url); r.raise_for_status()
            garment = Image.open(io.BytesIO(r.content)).convert("RGB")
        log.info("[%s] inputs fetched, running IDM-VTON", job.id)

        result = await asyncio.to_thread(
            PIPES.idm_vton.run,
            selfie=selfie, garment=garment, category=req.category,
        )

        # Save and serve back
        out_path = RESULTS_DIR / f"{job.id}.png"
        result.save(out_path)
        job.output_url = f"http://localhost:8000/results/{job.id}.png"
        job.status = "succeeded"
        log.info("[%s] still done in %.1fs", job.id, time.time() - job.started_at)
    except Exception as e:
        log.exception("[%s] still failed", job.id)
        job.status = "failed"
        job.error = str(e)


@app.post("/tryon/video")
async def tryon_video(req: VideoReq) -> dict[str, str]:
    if PIPES.wan_i2v is None:
        raise HTTPException(503, "Wan 2.1 i2v not loaded on this box")
    job = Job(id=f"video_{uuid.uuid4().hex}", kind="video")
    JOBS[job.id] = job
    asyncio.create_task(_run_video(job, req))
    return {"prediction_id": job.id}


async def _run_video(job: Job, req: VideoReq) -> None:
    try:
        import httpx
        from PIL import Image
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.get(req.still_url); r.raise_for_status()
            still = Image.open(io.BytesIO(r.content)).convert("RGB")

        log.info("[%s] running Wan 2.1 i2v", job.id)
        video_bytes = await asyncio.to_thread(PIPES.wan_i2v.run, still=still)

        out_path = RESULTS_DIR / f"{job.id}.mp4"
        out_path.write_bytes(video_bytes)
        job.output_url = f"http://localhost:8000/results/{job.id}.mp4"
        job.status = "succeeded"
        log.info("[%s] video done in %.1fs", job.id, time.time() - job.started_at)
    except Exception as e:
        log.exception("[%s] video failed", job.id)
        job.status = "failed"
        job.error = str(e)


@app.get("/predictions/{prediction_id}")
async def get_prediction(prediction_id: str) -> dict[str, Any]:
    job = JOBS.get(prediction_id)
    if not job:
        raise HTTPException(404, "unknown prediction_id")
    return {
        "status": job.status,
        "output_url": job.output_url,
        "error": job.error,
    }


# ============================================================================
# Live mode — WebSocket streaming for real-time DM-VTON warping
# ============================================================================
# Protocol:
#   client → server (binary):  raw JPEG frame from webcam, ~30fps
#   server → client (binary):  raw JPEG frame, garment overlaid via DM-VTON
#                              (returned when ready; may drop frames under load)
#
#   client → server (text):    {"action": "set_garment", "url": "...", "category": "dress"}
#                              switches the active garment for subsequent frames
#
# Backpressure: server processes one frame at a time, discards new frames that
# arrive while busy. Customer perceives ~15-20fps on a single RTX 4080.
@app.websocket("/ws/live")
async def ws_live(ws: WebSocket) -> None:
    await ws.accept()
    if PIPES.dm_vton is None:
        await ws.send_json({"error": "DM-VTON not loaded on this box"})
        await ws.close()
        return

    log.info("ws/live connected")
    garment_state: dict[str, Any] = {"image": None, "category": "dress"}
    busy = False

    try:
        while True:
            msg = await ws.receive()
            if "bytes" in msg and msg["bytes"]:
                # Frame from webcam
                if busy:
                    continue  # drop, server still warping the previous frame
                busy = True
                try:
                    from PIL import Image
                    frame = Image.open(io.BytesIO(msg["bytes"])).convert("RGB")
                    if garment_state["image"] is None:
                        # No garment selected yet — echo the frame back unmodified
                        out_bytes = msg["bytes"]
                    else:
                        out = await asyncio.to_thread(
                            PIPES.dm_vton.warp,
                            frame=frame,
                            garment=garment_state["image"],
                            category=garment_state["category"],
                        )
                        buf = io.BytesIO()
                        out.save(buf, format="JPEG", quality=85)
                        out_bytes = buf.getvalue()
                    await ws.send_bytes(out_bytes)
                finally:
                    busy = False
            elif "text" in msg and msg["text"]:
                import json
                evt = json.loads(msg["text"])
                if evt.get("action") == "set_garment":
                    import httpx
                    from PIL import Image
                    async with httpx.AsyncClient(timeout=10.0) as client:
                        r = await client.get(evt["url"]); r.raise_for_status()
                    garment_state["image"] = Image.open(io.BytesIO(r.content)).convert("RGB")
                    garment_state["category"] = evt.get("category", "dress")
                    log.info("garment set to %s", evt["url"])
                    await ws.send_json({"ok": True, "garment": evt["url"]})
    except WebSocketDisconnect:
        log.info("ws/live disconnected")


# ============================================================================
# Health + readiness
# ============================================================================
@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    return {
        "ok": True,
        "cuda": PIPES.cuda_available,
        "snapshot": PIPES.idm_vton is not None and PIPES.wan_i2v is not None,
        "live": PIPES.dm_vton is not None,
        "model_versions": {
            "idm_vton": "yisol/IDM-VTON",
            "wan_i2v": "Wan-Video/Wan2.1",
            "dm_vton": "KiseKloset/DM-VTON",
        },
    }


# Static result files (so webapp can fetch /results/*.png and /results/*.mp4).
# We need the mount BEFORE the app starts handling requests, but the directory
# might not exist yet at import time. Create it here best-effort — the real
# permission check is in lifespan(), which fails loudly with a useful message
# if writes won't work.
from fastapi.staticfiles import StaticFiles
try:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
except PermissionError:
    pass  # lifespan() will surface this with a clear error message
app.mount("/results", StaticFiles(directory=str(RESULTS_DIR), check_dir=False), name="results")
