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

# Base URL the webapp's browser uses to fetch /results/*.png and /results/*.mp4
# from this service. MUST include the host the browser can reach — using
# 'localhost' breaks split deployments (the Mac kiosk's browser interprets
# localhost as the Mac itself, not the vendor box).
# Default: best-effort detection. Override with VENDORBOX_PUBLIC_URL.
def _default_public_url() -> str:
    import socket
    try:
        # Connect-style detection: which interface routes outbound traffic?
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
        return f"http://{ip}:8000"
    except Exception:
        return "http://localhost:8000"

PUBLIC_BASE_URL = os.environ.get("VENDORBOX_PUBLIC_URL", _default_public_url()).rstrip("/")


# ============================================================================
# Pipeline holders — populated by lifespan() on startup
# ============================================================================
@dataclass
class Pipelines:
    """Lazy-loaded pipelines. None = unavailable on this box."""
    idm_vton: Any = None
    catvton: Any = None
    wan_i2v: Any = None
    dm_vton: Any = None
    device: str = "cpu"
    cuda_available: bool = False


PIPES = Pipelines()

# Which model serves /tryon/still. Set via SNAPSHOT_MODEL env var:
#   SNAPSHOT_MODEL=idm_vton  (default, photoreal but slow ~25-30s on 4060 Ti)
#   SNAPSHOT_MODEL=catvton   (fast ~4-6s, similar quality, NC license)
#   SNAPSHOT_MODEL=auto      (use catvton if loaded, else idm_vton)
SNAPSHOT_MODEL = os.environ.get("SNAPSHOT_MODEL", "auto").lower()


def _active_snapshot_pipe():
    """Return the IDMVTONPipe or CatVTONPipe currently serving /tryon/still,
    or None if no snapshot model is loaded."""
    if SNAPSHOT_MODEL == "catvton":
        return PIPES.catvton
    if SNAPSHOT_MODEL == "idm_vton":
        return PIPES.idm_vton
    # auto: prefer catvton (faster), fall back to idm_vton
    return PIPES.catvton or PIPES.idm_vton


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

        # ── IDM-VTON (snapshot still) — DISABLED by default ──────────
        # Snapshot/photo-booth UX is not in the customer page anymore. IDM-VTON
        # also conflicts with newer diffusers (it imports the now-internal
        # PositionNet class). Set LOAD_IDM_VTON=1 to opt in if you want
        # snapshot mode back; otherwise we skip and don't even import.
        if os.environ.get("LOAD_IDM_VTON") == "1":
            log.info("loading IDM-VTON…")
            PIPES.idm_vton = _try_load("IDM-VTON", lambda: (
                __import__("idm_vton_loader").load_idm_vton(MODELS_DIR / "idm-vton", device="cuda")
            ))
            if PIPES.idm_vton: log.info("IDM-VTON ready")
        else:
            log.info("IDM-VTON: skipped (set LOAD_IDM_VTON=1 to enable)")

        # ── CatVTON (ICLR 2025) — fast SD 1.5 alternative ──────────────
        # 10× faster than IDM-VTON on the same hardware (~4-6s on RTX 4060 Ti).
        # Set LOAD_CATVTON=1 to enable. License: CC BY-NC-SA 4.0 — research/
        # non-commercial only without a separately-negotiated commercial license
        # from the authors.
        if os.environ.get("LOAD_CATVTON") == "1":
            log.info("loading CatVTON…")
            PIPES.catvton = _try_load("CatVTON", lambda: (
                __import__("catvton_loader").load_catvton(MODELS_DIR / "catvton", device="cuda")
            ))
            if PIPES.catvton: log.info("CatVTON ready")
        else:
            log.info("CatVTON: skipped (set LOAD_CATVTON=1 to enable)")

        # ── Wan 2.1 i2v (snapshot video) — DISABLED by default ───────
        # Video animation only makes sense layered on top of a snapshot still.
        # Skip in the LIVE-only product. Set LOAD_WAN_I2V=1 to opt in.
        if os.environ.get("LOAD_WAN_I2V") == "1":
            log.info("loading Wan 2.1 i2v…")
            PIPES.wan_i2v = _try_load("Wan 2.1", lambda: (
                __import__("wan21_loader").load_wan_i2v(MODELS_DIR / "wan2.1", device="cuda")
            ))
            if PIPES.wan_i2v: log.info("Wan 2.1 i2v ready")
        else:
            log.info("Wan 2.1: skipped (LIVE-only mode; set LOAD_WAN_I2V=1 to enable)")

        # ── DM-VTON (live streaming) — opt-in via LOAD_DM_VTON=1 ──────
        # Default off because DM-VTON's repo has a top-level `utils` package
        # whose import side-effects collide with IDM-VTON's top-level `utils`
        # when both are on sys.path. Snapshot-only deployments (IDM-VTON) do
        # not need DM-VTON, so we leave it off by default.
        if os.environ.get("LOAD_DM_VTON") == "1":
            log.info("loading DM-VTON (TensorRT)…")
            PIPES.dm_vton = _try_load("DM-VTON", lambda: (
                __import__("dm_vton_loader").load_dm_vton_trt(MODELS_DIR / "dm-vton")
            ))
        else:
            log.info("DM-VTON: skipped (set LOAD_DM_VTON=1 to enable for LIVE mode)")
        if PIPES.dm_vton: log.info("DM-VTON live pipeline ready")

    except ImportError as e:
        log.error("Required Python deps missing: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("vendor-box starting up")
    log.info("  models dir  = %s", MODELS_DIR)
    log.info("  results dir = %s", RESULTS_DIR)
    log.info("  public URL  = %s   (override with VENDORBOX_PUBLIC_URL)", PUBLIC_BASE_URL)
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
    """Kick off a snapshot inference (IDM-VTON or CatVTON depending on
    SNAPSHOT_MODEL). Returns a prediction_id to poll."""
    pipe = _active_snapshot_pipe()
    if pipe is None:
        raise HTTPException(
            503,
            "No snapshot model loaded. Set LOAD_IDM_VTON=1 or LOAD_CATVTON=1.",
        )
    job = Job(id=f"still_{uuid.uuid4().hex}", kind="still")
    JOBS[job.id] = job
    asyncio.create_task(_run_still(job, req))
    return {"prediction_id": job.id}


async def _run_still(job: Job, req: StillReq) -> None:
    try:
        pipe = _active_snapshot_pipe()
        if pipe is None:
            raise RuntimeError("snapshot model unloaded between request + run")
        # Download inputs from the webapp (same LAN)
        import httpx
        from PIL import Image
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.get(req.selfie_url); r.raise_for_status()
            selfie = Image.open(io.BytesIO(r.content)).convert("RGB")
            r = await client.get(req.garment_url); r.raise_for_status()
            garment = Image.open(io.BytesIO(r.content)).convert("RGB")
        # Identify which model we're routing to so the log + tuning dials
        # match the right backend.
        is_catvton = pipe is PIPES.catvton
        model_name = "CatVTON" if is_catvton else "IDM-VTON"
        log.info("[%s] inputs fetched, running %s", job.id, model_name)

        # Per-backend quality dials. Defaults are FAST for both.
        #   IDM_VTON_STEPS=4 / IDM_VTON_RES=512 / IDM_VTON_GUIDANCE=2.0
        #   CATVTON_STEPS=20 / CATVTON_RES=512 / CATVTON_GUIDANCE=2.5
        if is_catvton:
            n_steps  = int(os.environ.get("CATVTON_STEPS",   "20"))
            res      = int(os.environ.get("CATVTON_RES",     "512"))
            guidance = float(os.environ.get("CATVTON_GUIDANCE", "2.5"))
        else:
            n_steps  = int(os.environ.get("IDM_VTON_STEPS",  "4"))
            res      = int(os.environ.get("IDM_VTON_RES",    "512"))
            guidance = float(os.environ.get("IDM_VTON_GUIDANCE", "2.0"))

        result = await asyncio.to_thread(
            pipe.run,
            selfie=selfie, garment=garment, category=req.category,
            n_steps=n_steps,
            guidance_scale=guidance,
            target_width=res,
        )

        # Save and serve back
        out_path = RESULTS_DIR / f"{job.id}.png"
        result.save(out_path)
        job.output_url = f"{PUBLIC_BASE_URL}/results/{job.id}.png"
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
    """LIVE try-on stream.

    Two modes, picked automatically based on which model is loaded:
      - dm_vton (preferred):  ~50-100 ms per frame → ~15-20 fps perceived
      - idm_vton (fallback):  ~3-5 sec per frame   → "quasi-live"

    Either way the customer sees a real photoreal-quality render of
    themselves wearing the garment, updated continuously as they move.
    """
    await ws.accept()
    if PIPES.dm_vton is None and PIPES.idm_vton is None:
        await ws.send_json({"error": "No live VTON model loaded on this box"})
        await ws.close()
        return

    mode = "dm_vton" if PIPES.dm_vton is not None else "idm_vton_lowstep"
    log.info("ws/live connected, mode=%s", mode)
    await ws.send_json({"ok": True, "mode": mode})

    garment_state: dict[str, Any] = {"image": None, "category": "dress"}
    busy = False

    async def _process_frame(frame: "Image.Image") -> "Image.Image":
        """Pick the right model based on what's loaded."""
        if PIPES.dm_vton is not None:
            return await asyncio.to_thread(
                PIPES.dm_vton.warp,
                frame=frame,
                garment=garment_state["image"],
                category=garment_state["category"],
            )
        # IDM-VTON fallback: run at 8 inference steps for speed (vs 30 for
        # snapshot quality). Tradeoff is slightly less crisp output but
        # 3-5s per frame instead of 15-25s.
        return await asyncio.to_thread(
            PIPES.idm_vton.run,
            selfie=frame,
            garment=garment_state["image"],
            category=garment_state["category"],
            n_steps=8,
            guidance_scale=2.0,
        )

    try:
        while True:
            msg = await ws.receive()
            if "bytes" in msg and msg["bytes"]:
                if busy:
                    continue  # drop, still working on the previous frame
                busy = True
                try:
                    from PIL import Image
                    frame = Image.open(io.BytesIO(msg["bytes"])).convert("RGB")
                    if garment_state["image"] is None:
                        out_bytes = msg["bytes"]  # echo back until garment is picked
                    else:
                        t0 = time.time()
                        out = await _process_frame(frame)
                        log.debug("frame processed in %.2fs", time.time() - t0)
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
    # `snapshot` only needs IDM-VTON — Wan 2.1 is an OPTIONAL video upgrade
    # on top of the still. The webapp orchestrator finishes snapshot jobs
    # successfully with just the still when Wan 2.1 is unavailable.
    # `live` is true if EITHER DM-VTON or IDM-VTON is loaded — the WebSocket
    # handler falls back to IDM-VTON low-step when DM-VTON isn't installed.
    active = _active_snapshot_pipe()
    snapshot_model = (
        "catvton" if active is PIPES.catvton
        else "idm_vton" if active is PIPES.idm_vton
        else "unavailable"
    )
    return {
        "ok": True,
        "cuda": PIPES.cuda_available,
        "snapshot": active is not None,
        "snapshot_model": snapshot_model,           # which one serves /tryon/still
        "snapshot_model_preference": SNAPSHOT_MODEL, # env-var setting (auto/idm_vton/catvton)
        "live": PIPES.dm_vton is not None or PIPES.idm_vton is not None,
        "live_mode": (
            "dm_vton" if PIPES.dm_vton is not None
            else "idm_vton_lowstep" if PIPES.idm_vton is not None
            else "unavailable"
        ),
        "video": PIPES.wan_i2v is not None,
        "models_loaded": {
            "idm_vton": PIPES.idm_vton is not None,
            "catvton": PIPES.catvton is not None,
            "wan_i2v": PIPES.wan_i2v is not None,
            "dm_vton": PIPES.dm_vton is not None,
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
