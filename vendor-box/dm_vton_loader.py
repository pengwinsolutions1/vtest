"""DM-VTON wrapper — real-time virtual try-on for the LIVE mode WebSocket.

DM-VTON ships a high-level `DMVTONPipeline` class that handles pose extraction
and checkpoint loading internally. We just pass person + clothes + edge mask.

On an RTX 4080 in FP16 this lands at ~50-100 ms per frame at 256×192. The
WebSocket pipeline in server.py drops incoming frames when busy so the
customer perceives ~15-20 fps live.
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image

log = logging.getLogger("dm-vton")


@dataclass
class DMVTONPipe:
    pipeline: Any        # DMVTONPipeline instance
    device: str
    tensor_dtype: Any    # torch.float16 normally

    def warp(
        self,
        frame: Image.Image,
        garment: Image.Image,
        category: Literal["top", "bottom", "dress"] = "dress",
    ) -> Image.Image:
        """Single-frame inference. ~50-100ms on RTX 4080 in FP16.

        category is unused by DM-VTON itself (it learns category from the
        garment shape), but we keep the parameter for API symmetry with
        IDM-VTON.
        """
        import torch
        from torchvision import transforms

        H, W = 256, 192  # DM-VTON's native resolution

        # Resize inputs
        frame_rgb = frame.convert("RGB").resize((W, H))
        garment_rgba = garment.convert("RGBA").resize((W, H))

        # Edge mask comes from the garment PNG's alpha channel — binary
        # mask of where the garment pixels are (vs the transparent bg).
        alpha = np.array(garment_rgba.split()[3])
        edge = (alpha > 128).astype(np.float32)

        # Normalised tensors in [-1, 1] (matches what DM-VTON was trained on)
        norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * 3, (0.5,) * 3),
        ])
        person_t  = norm(frame_rgb).unsqueeze(0).to(self.device, self.tensor_dtype)
        clothes_t = norm(garment_rgba.convert("RGB")).unsqueeze(0).to(self.device, self.tensor_dtype)
        edge_t    = torch.from_numpy(edge)[None, None, ...].to(self.device, self.tensor_dtype)

        with torch.no_grad():
            p_tryon, _warped = self.pipeline(person_t, clothes_t, edge_t, phase="test")

        # Denormalize [-1,1] → [0,1] → PIL
        result = ((p_tryon.clamp(-1, 1) + 1) / 2)
        result = result.squeeze(0).permute(1, 2, 0).float().cpu().numpy()
        result = (result * 255).astype(np.uint8)
        return Image.fromarray(result).resize(frame.size)


def load_dm_vton_trt(path: Path) -> DMVTONPipe:
    """Load DM-VTON's `DMVTONPipeline` from the cloned ../DM-VTON/ repo.

    Looks for any *warp*.{pt,pth} and *gen*.{pt,pth} in `path` — DM-VTON's
    Drive folder ships several variants (mobile, pf, fs, etc.) with varying
    filenames. The first matching pair is used.

    TensorRT optimisation is NOT applied here — pure PyTorch FP16 gives
    ~50-100 ms / frame which is good enough for the WebSocket's perceived
    15-20 fps. TRT would cut that to ~30-40 ms.
    """
    import torch

    repo_root = Path(__file__).resolve().parent / "DM-VTON"
    if not (repo_root / "pipelines").exists():
        raise FileNotFoundError(
            f"DM-VTON repo not found at {repo_root}. Run scripts/setup_dm_vton.sh first."
        )
    sys.path.insert(0, str(repo_root))

    # Match by substring — handle whatever naming the Drive variant uses
    def _find(needle: str) -> Path:
        candidates = sorted(path.glob(f"*{needle}*.pt")) + sorted(path.glob(f"*{needle}*.pth"))
        if not candidates:
            raise FileNotFoundError(
                f"No *{needle}*.pt or *{needle}*.pth in {path}. Download from "
                f"https://drive.google.com/drive/folders/1wfWGsR0vWC5LrA26xhj92ec_GoCKV80A"
            )
        return candidates[0]

    warp_ckpt = _find("warp")
    gen_ckpt  = _find("gen")
    log.info("found warp checkpoint: %s", warp_ckpt.name)
    log.info("found gen checkpoint:  %s", gen_ckpt.name)

    # Late import: only after sys.path includes the repo
    from pipelines import DMVTONPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # DM-VTON's submodules don't all auto-cast cleanly to FP16 — partial .half()
    # leaves dtype mismatches between layers (input FloatTensor vs weight
    # HalfTensor RuntimeError at the first conv). It's small enough that FP32
    # on RTX 4080 stays ~100-200 ms / frame, which is fine for live perception.
    dtype = torch.float32

    log.info("instantiating DMVTONPipeline…")
    pipeline = DMVTONPipeline(
        align_corners=True,
        checkpoints={"warp": str(warp_ckpt), "gen": str(gen_ckpt)},
    ).to(device).eval()

    log.info("DM-VTON pipeline ready (PyTorch %s on %s)", dtype, device)
    return DMVTONPipe(pipeline=pipeline, device=device, tensor_dtype=dtype)
