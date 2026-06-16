"""CatVTON wrapper — photoreal still virtual try-on, ~4-6s on RTX 4060 Ti.

CatVTON (ICLR 2025, Zheng-Chong/CatVTON) is a much lighter alternative to
IDM-VTON. The key differences that make it ~10× faster on the same hardware:

  - 899M total params (50M trainable) vs IDM-VTON's ~14 GB stack
  - SD 1.5 backbone (not SDXL) — half the resolution, ~4× less compute
  - NO text encoder, NO pose detector, NO human parser, NO IP-Adapter
  - Person + garment + agnostic mask are concatenated on the channel axis
  - Fits in ~3.3 GB VRAM at full resolution (vs IDM-VTON's 14 GB)

The only preprocessing we need is the agnostic mask. We REUSE the same
MediaPipe / OpenPose / HumanParsing setup IDM-VTON uses to generate it
(get_mask_location), since CatVTON's official inference also expects a
pre-computed mask.

NOTE on license: CatVTON ships under CC BY-NC-SA 4.0 (non-commercial).
For paid retail deployment you must email the authors for a commercial
license — see https://github.com/Zheng-Chong/CatVTON#license-and-citation.

This file is a SCAFFOLD. The integration steps (clone the repo, download
weights, wire the pipeline) require the actual code on disk; run
`scripts/setup_catvton.sh` first. Until then, the loader raises
NotImplementedError which gets reported as TODO in the startup logs and
doesn't crash the service.
"""
from __future__ import annotations

import logging
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from PIL import Image

warnings.filterwarnings("ignore", category=FutureWarning, module=r"diffusers\..*")

log = logging.getLogger("catvton")

# Same category names as IDM-VTON for API symmetry. CatVTON itself uses
# 'upper_body' / 'lower_body' / 'overall' (dresses) — map at the call site.
CATEGORY_MAP = {"top": "upper_body", "bottom": "lower_body", "dress": "overall"}


@dataclass
class CatVTONPipe:
    pipe: Any                       # CatVTONPipeline instance
    automasker: Any                 # CatVTON's AutoMasker (replaces OpenPose+Parsing for us)
    device: str
    tensor_dtype: Any               # torch.float16 normally

    def run(
        self,
        selfie: Image.Image,
        garment: Image.Image,
        category: Literal["top", "bottom", "dress"],
        garment_description: str = "garment",   # ignored — CatVTON has no text path
        n_steps: int = 20,                       # 20 is the sweet spot; 30+ is overkill
        guidance_scale: float = 2.5,
        target_width: int = 512,
    ) -> Image.Image:
        """One photoreal try-on inference. ~4-6s on RTX 4060 Ti at 20 steps.

        Mirrors gradio_app.py from the upstream repo.
        """
        # TODO: integrate against the actual CatVTON pipeline once
        # scripts/setup_catvton.sh has cloned the repo. The reference
        # inference is roughly:
        #
        #   from model.pipeline import CatVTONPipeline
        #   image = pipe(image=selfie, condition_image=garment, mask=mask,
        #                num_inference_steps=n_steps, guidance_scale=gs).images[0]
        raise NotImplementedError(
            "CatVTON pipeline not yet integrated — "
            "complete scripts/setup_catvton.sh first."
        )


def load_catvton(path: Path, device: str = "cuda") -> CatVTONPipe:
    """Load CatVTON weights + AutoMasker from the cloned CatVTON repo.

    Path layout expected (created by scripts/setup_catvton.sh):
        path/                          # data/models/catvton/
        ├── CatVTON/                   # cloned repo
        │   ├── model/
        │   ├── densepose/
        │   └── ...
        └── weights/                   # HF snapshot of zhengchong/CatVTON
            ├── automasker/
            ├── catvton/
            └── densepose/
    """
    import torch

    repo_root = path / "CatVTON"
    if not (repo_root / "model" / "pipeline.py").exists():
        raise FileNotFoundError(
            f"CatVTON repo not found at {repo_root}. "
            f"Run scripts/setup_catvton.sh first."
        )
    sys.path.insert(0, str(repo_root))

    weights = path / "weights"
    if not (weights / "catvton").exists():
        raise FileNotFoundError(
            f"CatVTON weights not found at {weights}. "
            f"Run scripts/setup_catvton.sh first."
        )

    # TODO: replace with actual loader once repo is on disk
    raise NotImplementedError(
        f"Loading CatVTON from {path} not implemented — "
        f"wire up the real loader after running scripts/setup_catvton.sh"
    )
