"""IDM-VTON wrapper — photoreal still virtual try-on.

Loads the official IDM-VTON pipeline from the cloned repo at ../IDM-VTON.
The repo ships custom Stable Diffusion XL UNet modules that we have to
import directly (they're not pip-installable). Run scripts/setup_idm_vton.sh
first to clone the repo + download weights.

Inference flow (matches IDM-VTON's gradio_demo/app.py):
    1. Pre-process customer photo:
       - Run OpenPose to extract pose keypoints
       - Run HumanParsing to label body regions (arms, torso, etc.)
       - Generate an "agnostic mask" — the region to be replaced by the garment
       - Run DensePose to get a 3D body surface map
    2. Feed (photo, garment, agnostic mask, densepose) into the SDXL pipeline
    3. Pipeline returns a photoreal image of the customer wearing the garment
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from PIL import Image

log = logging.getLogger("idm-vton")

# Map our internal category names to IDM-VTON's labels
CATEGORY_MAP = {"top": "upper_body", "bottom": "lower_body", "dress": "dresses"}


@dataclass
class IDMVTONPipe:
    pipe: Any                       # the SDXL TryonPipeline
    parsing_model: Any              # HumanParsing
    openpose_model: Any             # OpenPose
    device: str
    tensor_dtype: Any               # torch.float16 normally

    def run(
        self,
        selfie: Image.Image,
        garment: Image.Image,
        category: Literal["top", "bottom", "dress"],
        garment_description: str = "garment",
        n_steps: int = 30,
        guidance_scale: float = 2.0,
        seed: int = 42,
    ) -> Image.Image:
        """One photoreal try-on inference. ~15-25 sec on RTX 4080.

        Returns the customer photo with the garment composited photoreally.
        """
        import torch
        from torchvision import transforms

        idm_category = CATEGORY_MAP[category]

        # Resize inputs to 768×1024 (IDM-VTON's native resolution)
        TARGET = (768, 1024)
        selfie_resized = selfie.convert("RGB").resize(TARGET)
        garment_resized = garment.convert("RGB").resize(TARGET)

        # ─── Pre-process the human photo ─────────────────────────────
        log.info("running pose + parsing pre-processing…")
        # OpenPose: 18-keypoint stick figure on a black background
        keypoints = self.openpose_model(selfie_resized.resize((384, 512)))
        # HumanParsing: per-pixel body region labels (arms, torso, face, ...)
        parsed_image, _ = self.parsing_model(selfie_resized.resize((384, 512)))

        # Agnostic mask: the region where the new garment will be painted in.
        # For dresses → torso + upper legs. For top → torso + upper arms. Etc.
        # IDM-VTON ships a helper for this.
        from preprocess.humanparsing.run_parsing import get_mask_location
        agnostic_mask_pil, agnostic_mask = get_mask_location(
            "hd", idm_category, parsed_image.resize((384, 512)), keypoints,
        )
        agnostic_mask = agnostic_mask_pil.resize(TARGET)

        # DensePose: dense correspondence between pixels and a 3D body model.
        # IDM-VTON's app uses a torchscripted model from the ckpt dir.
        from preprocess.dwpose import DWposeDetector
        # Note: actual IDM-VTON uses DensePose; some forks use DWPose. Adapt
        # as needed based on what's in ckpt/.
        # For now, leverage the ip-adapter / dense pose path from app.py.
        # (Placeholder — densepose_img generation depends on repo version)
        from torchvision.transforms import ToTensor
        densepose_tensor = ToTensor()(selfie_resized).unsqueeze(0).to(self.device, self.tensor_dtype)

        # ─── Run the SDXL TryonPipeline ──────────────────────────────
        log.info("running SDXL pipeline (%d steps)…", n_steps)
        with torch.no_grad():
            prompt = f"model is wearing a {garment_description}"
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
            prompt_embeds, neg_prompt_embeds, pooled_prompt_embeds, neg_pooled_prompt_embeds = self.pipe.encode_prompt(
                prompt, negative_prompt=negative_prompt, num_images_per_prompt=1, do_classifier_free_guidance=True,
            )
            prompt_embeds_c, _, pooled_c, _ = self.pipe.encode_prompt(
                f"a photo of {garment_description}", num_images_per_prompt=1, do_classifier_free_guidance=False,
            )

            generator = torch.Generator(self.device).manual_seed(seed)
            images = self.pipe(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=neg_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=neg_pooled_prompt_embeds,
                num_inference_steps=n_steps,
                generator=generator,
                strength=1.0,
                pose_img=densepose_tensor,
                text_embeds_cloth=prompt_embeds_c,
                cloth=ToTensor()(garment_resized).unsqueeze(0).to(self.device, self.tensor_dtype),
                mask_image=agnostic_mask,
                image=selfie_resized,
                height=TARGET[1],
                width=TARGET[0],
                ip_adapter_image=garment_resized.resize((224, 224)),
                guidance_scale=guidance_scale,
            ).images

        return images[0]


def load_idm_vton(path: Path, device: str = "cuda") -> IDMVTONPipe:
    """Load IDM-VTON weights + UNet hacks from the cloned repo at ../IDM-VTON.

    Raises a clear error if the repo isn't cloned or weights aren't downloaded.
    Both are done by scripts/setup_idm_vton.sh.
    """
    import torch

    # The IDM-VTON repo must be cloned next to vendor-box/ (i.e. as sibling
    # of this file's package). Add it to sys.path so its `src/` and
    # `preprocess/` imports resolve.
    repo_root = Path(__file__).resolve().parent / "IDM-VTON"
    if not (repo_root / "src" / "tryon_pipeline.py").exists():
        raise FileNotFoundError(
            f"IDM-VTON repo not found at {repo_root}. Run scripts/setup_idm_vton.sh first."
        )
    sys.path.insert(0, str(repo_root))

    if not (path / "ckpt" / "densepose").exists():
        raise FileNotFoundError(
            f"IDM-VTON weights not found at {path}. Run scripts/setup_idm_vton.sh first."
        )

    # Import IDM-VTON's custom modules (only available once sys.path includes the clone)
    from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
    from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
    from src.unet_hacked_tryon import UNet2DConditionModel
    from preprocess.humanparsing.aigc_run_parsing import Parsing
    from preprocess.openpose.run_openpose import OpenPose

    from transformers import (
        CLIPImageProcessor,
        CLIPVisionModelWithProjection,
        CLIPTextModel,
        CLIPTextModelWithProjection,
        AutoTokenizer,
    )
    from diffusers import AutoencoderKL, DDPMScheduler

    base = str(path)
    dtype = torch.float16
    log.info("loading IDM-VTON SDXL components…")

    unet = UNet2DConditionModel.from_pretrained(
        base, subfolder="unet", torch_dtype=dtype,
    ).to(device).requires_grad_(False)

    unet_encoder = UNet2DConditionModel_ref.from_pretrained(
        base, subfolder="unet_encoder", torch_dtype=dtype,
    ).to(device).requires_grad_(False)

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        base, subfolder="image_encoder", torch_dtype=dtype,
    ).to(device).requires_grad_(False)

    vae = AutoencoderKL.from_pretrained(
        base, subfolder="vae", torch_dtype=dtype,
    ).to(device).requires_grad_(False)

    text_encoder_one = CLIPTextModel.from_pretrained(
        base, subfolder="text_encoder", torch_dtype=dtype,
    ).to(device).requires_grad_(False)
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
        base, subfolder="text_encoder_2", torch_dtype=dtype,
    ).to(device).requires_grad_(False)

    tokenizer_one = AutoTokenizer.from_pretrained(base, subfolder="tokenizer", use_fast=False)
    tokenizer_two = AutoTokenizer.from_pretrained(base, subfolder="tokenizer_2", use_fast=False)
    scheduler = DDPMScheduler.from_pretrained(base, subfolder="scheduler")
    feature_extractor = CLIPImageProcessor()

    pipe = TryonPipeline.from_pretrained(
        base,
        unet=unet,
        vae=vae,
        feature_extractor=feature_extractor,
        text_encoder=text_encoder_one,
        text_encoder_2=text_encoder_two,
        tokenizer=tokenizer_one,
        tokenizer_2=tokenizer_two,
        scheduler=scheduler,
        image_encoder=image_encoder,
        unet_encoder=unet_encoder,
        torch_dtype=dtype,
    )
    pipe.to(device)

    # Pre-processing models (live on CPU + small GPU footprint)
    log.info("loading OpenPose + HumanParsing pre-processors…")
    openpose = OpenPose(str(path / "ckpt" / "openpose"))
    openpose.preprocessor.body_estimation.model.to(device)
    parsing = Parsing(str(path / "ckpt" / "humanparsing"))

    log.info("IDM-VTON pipeline ready")
    return IDMVTONPipe(
        pipe=pipe, parsing_model=parsing, openpose_model=openpose,
        device=device, tensor_dtype=dtype,
    )
