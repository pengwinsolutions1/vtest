"""IDM-VTON wrapper. Photoreal still try-on.

Loads the diffusers pipeline + custom modules from `yisol/IDM-VTON`. Real
integration TBD — this file defines the runtime contract server.py expects:

    pipe = load_idm_vton(path, device)
    out_image = pipe.run(selfie=PIL, garment=PIL, category="dress")

When the vendor box is set up, replace the `_TODO_` body with the actual
inference call. The IDM-VTON repo has a `gradio_demo/app.py` that's a useful
starting point — pull out the inference function and call it here.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from PIL import Image


@dataclass
class IDMVTONPipe:
    pipe: Any   # the actual diffusers pipeline (TBD on box)
    device: str

    def run(self, selfie: Image.Image, garment: Image.Image, category: str) -> Image.Image:
        """Run one inference. Should take ~15-25s on RTX 4080 at 768×1024.

        category ∈ {"top", "bottom", "dress"} — IDM-VTON uses
        {"upper_body", "lower_body", "dresses"}. Map here.
        """
        # TODO: implement against the real pipeline once weights are on disk
        # category_map = {"top": "upper_body", "bottom": "lower_body", "dress": "dresses"}
        # return self.pipe(
        #     image=selfie, garm_img=garment, category=category_map[category],
        #     num_inference_steps=30, guidance_scale=2.0,
        # )["images"][0]
        raise NotImplementedError("IDM-VTON pipeline not yet integrated — see TODO in idm_vton_loader.py")


def load_idm_vton(path: Path, device: str = "cuda") -> IDMVTONPipe:
    """Load weights and prep the pipeline. Called once at service start."""
    # TODO: implement against the real loader once the box is up
    # from diffusers import DiffusionPipeline
    # pipe = DiffusionPipeline.from_pretrained(str(path), torch_dtype=torch.float16)
    # pipe.to(device)
    # return IDMVTONPipe(pipe=pipe, device=device)
    raise NotImplementedError(
        f"Loading IDM-VTON from {path} not implemented — wire up the real loader on the vendor box"
    )
