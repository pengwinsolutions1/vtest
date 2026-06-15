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
    _last_crop: Any = None          # (orig_w, orig_h, left, top, tw, th)

    def run(
        self,
        selfie: Image.Image,
        garment: Image.Image,
        category: Literal["top", "bottom", "dress"],
        garment_description: str = "garment",
        # Trade-off knobs (override via server env vars IDM_VTON_STEPS /
        # IDM_VTON_RES / IDM_VTON_GUIDANCE):
        #   steps=4,  res=512  → ~3-6s, rough but recognisable (FAST default)
        #   steps=8,  res=768  → ~15-25s, balanced
        #   steps=20, res=768  → ~30-45s, near-training quality
        #   steps=30, res=768  → ~45-60s, max
        n_steps: int = 4,
        guidance_scale: float = 2.0,
        target_width: int = 512,    # 512 = fast, 768 = quality
        seed: int = 42,
    ) -> Image.Image:
        """One photoreal try-on inference. ~15-25 sec on RTX 4080.

        Port of gradio_demo/app.py::start_tryon — same preprocessing chain
        (OpenPose + HumanParsing + DensePose via detectron2.apply_net), same
        SDXL pipeline call.
        """
        import os
        import torch
        from torchvision import transforms
        from detectron2.data.detection_utils import convert_PIL_to_numpy, _apply_exif_orientation
        from utils_mask import get_mask_location
        import apply_net

        idm_category = CATEGORY_MAP[category]
        # 3:4 aspect from the configurable width. SDXL requires both
        # dimensions divisible by 8 — naive 512 × 4/3 = 682.67 → 682,
        # which raises ValueError. Floor both to the nearest multiple of 8.
        # 512 → 512 × 680   (was 682, fixed to 680)
        # 768 → 768 × 1024
        # 600 → 600 × 800
        _w = (target_width // 8) * 8
        _h = ((target_width * 4 // 3) // 8) * 8
        TARGET = (_w, _h)

        # ─── Auto-crop the selfie to 3:4 BEFORE resizing ────────────────
        # IDM-VTON's pipeline is trained on fashion portraits (3:4). Most
        # webcams ship 16:9, and a naive .resize((768,1024)) stretches a
        # body vertically by ~33% — OpenPose then sees a squashed shape
        # and finds 0 bodies (IndexError on pose['bodies']['subset'][0]).
        # Center-crop to 3:4 first so the body looks human.
        selfie_rgb = selfie.convert("RGB")
        w, h = selfie_rgb.size
        target_w = min(w, int(h * 3 / 4))
        target_h = min(h, int(w * 4 / 3))
        left = (w - target_w) // 2
        top = (h - target_h) // 2
        selfie_cropped = selfie_rgb.crop((left, top, left + target_w, top + target_h))
        crop_size = selfie_cropped.size
        human_img = selfie_cropped.resize(TARGET)
        # Remember the crop offsets — we paste the AI result back into the
        # original-aspect frame at these coords so the customer still sees
        # the full webcam frame, not a cropped one.
        self._last_crop = (w, h, left, top, target_w, target_h)

        garm_img = garment.convert("RGB").resize(TARGET)

        log.info("running pose + parsing pre-processing…")
        keypoints = self.openpose_model(human_img.resize((384, 512)))
        model_parse, _ = self.parsing_model(human_img.resize((384, 512)))
        mask, mask_gray = get_mask_location("hd", idm_category, model_parse, keypoints)
        mask = mask.resize(TARGET)

        log.info("running DensePose (detectron2)…")
        # apply_net's parse_args uses RELATIVE paths './configs/...' and
        # './ckpt/densepose/...'. Must chdir to IDM-VTON/ before invoking.
        repo_root = Path(__file__).resolve().parent / "IDM-VTON"
        prev_cwd = os.getcwd()
        try:
            os.chdir(repo_root)
            human_img_arg = _apply_exif_orientation(human_img.resize((384, 512)))
            human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")
            args = apply_net.create_argument_parser().parse_args((
                "show",
                "./configs/densepose_rcnn_R_50_FPN_s1x.yaml",
                "./ckpt/densepose/model_final_162be9.pkl",
                "dp_segm", "-v",
                "--opts", "MODEL.DEVICE", "cuda",
            ))
            pose_img = args.func(args, human_img_arg)
        finally:
            os.chdir(prev_cwd)
        pose_img = pose_img[:, :, ::-1]   # BGR → RGB
        pose_img = Image.fromarray(pose_img).resize(TARGET)

        tensor_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        log.info("running SDXL TryonPipeline (%d steps)…", n_steps)
        device = self.device
        with torch.no_grad(), torch.amp.autocast("cuda"):
            prompt = f"model is wearing {garment_description}"
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
            (
                prompt_embeds, neg_prompt_embeds,
                pooled_prompt_embeds, neg_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                prompt, num_images_per_prompt=1,
                do_classifier_free_guidance=True, negative_prompt=negative_prompt,
            )
            prompt_c = f"a photo of {garment_description}"
            (prompt_embeds_c, _, _, _) = self.pipe.encode_prompt(
                prompt_c, num_images_per_prompt=1,
                do_classifier_free_guidance=False, negative_prompt=negative_prompt,
            )

            pose_tensor = tensor_transform(pose_img).unsqueeze(0).to(device, torch.float16)
            garm_tensor = tensor_transform(garm_img).unsqueeze(0).to(device, torch.float16)
            generator = torch.Generator(device).manual_seed(seed)

            images = self.pipe(
                prompt_embeds=prompt_embeds.to(device, torch.float16),
                negative_prompt_embeds=neg_prompt_embeds.to(device, torch.float16),
                pooled_prompt_embeds=pooled_prompt_embeds.to(device, torch.float16),
                negative_pooled_prompt_embeds=neg_pooled_prompt_embeds.to(device, torch.float16),
                num_inference_steps=n_steps,
                generator=generator,
                strength=1.0,
                pose_img=pose_tensor,
                text_embeds_cloth=prompt_embeds_c.to(device, torch.float16),
                cloth=garm_tensor,
                mask_image=mask,
                image=human_img,
                height=TARGET[1], width=TARGET[0],
                ip_adapter_image=garm_img.resize(TARGET),
                guidance_scale=guidance_scale,
            )[0]

        # Free up VRAM + Python state before returning. Without this,
        # fragments accumulate across inferences. Explicitly del every
        # intermediate tensor — Python's GC alone doesn't reliably reach
        # them during a hot loop.
        result_img = images[0]
        del images, pose_tensor, garm_tensor
        del prompt_embeds, neg_prompt_embeds, pooled_prompt_embeds
        del neg_pooled_prompt_embeds, prompt_embeds_c
        del generator
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()   # block until all queued GPU ops complete
        return result_img


def load_idm_vton(path: Path, device: str = "cuda") -> IDMVTONPipe:
    """Load IDM-VTON weights + UNet hacks from the cloned repo at ../IDM-VTON.

    Raises a clear error if the repo isn't cloned or weights aren't downloaded.
    Both are done by scripts/setup_idm_vton.sh.
    """
    import torch

    # The IDM-VTON repo must be cloned next to vendor-box/ (i.e. as sibling
    # of this file's package). Add it to sys.path so its `src/` and
    # `preprocess/` imports resolve. Also add gradio_demo/ because that's
    # where utils_mask.get_mask_location lives (the IDM-VTON authors didn't
    # promote it to a package, it's just a script-level helper).
    repo_root = Path(__file__).resolve().parent / "IDM-VTON"
    if not (repo_root / "src" / "tryon_pipeline.py").exists():
        raise FileNotFoundError(
            f"IDM-VTON repo not found at {repo_root}. Run scripts/setup_idm_vton.sh first."
        )
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "gradio_demo"))

    # IDM-VTON's preprocess code uses RELATIVE paths like
    #   Path(__file__).parents[2] / 'ckpt/humanparsing/parsing_atr.onnx'
    # which resolves to IDM-VTON/ckpt/humanparsing/...  Our weights live
    # under data/models/idm-vton/<subdir>, so we symlink IDM-VTON/ckpt
    # to that location.
    #
    # CAVEAT: the upstream IDM-VTON repo ships Git LFS stubs under ckpt/,
    # so `git clone` (without lfs) leaves a real ckpt/ directory full of
    # 24-31-byte pointer files. We have to replace the stub directory
    # with the symlink, not skip-when-present.
    import shutil
    ckpt_link = repo_root / "ckpt"
    if ckpt_link.is_symlink():
        target = ckpt_link.resolve()
        if target != path.resolve():
            ckpt_link.unlink()  # was pointing at the wrong place
    if ckpt_link.exists() and not ckpt_link.is_symlink():
        # Real directory (Git LFS stubs). Verify it's NOT real weights,
        # then nuke it. A real weights dir would have a file > 10 MB inside.
        max_size = max(
            (f.stat().st_size for f in ckpt_link.rglob("*") if f.is_file()),
            default=0,
        )
        if max_size > 10_000_000:
            log.warning(
                "ckpt/ has real weights inside (%d bytes max) — leaving it alone",
                max_size,
            )
        else:
            log.info("removing LFS-stub ckpt/ (max file %d bytes)", max_size)
            shutil.rmtree(ckpt_link)
    if not ckpt_link.exists():
        try:
            ckpt_link.symlink_to(path, target_is_directory=True)
            log.info("symlinked %s -> %s", ckpt_link, path)
        except OSError as e:
            raise RuntimeError(
                f"Couldn't create symlink {ckpt_link} -> {path}: {e}. "
                f"Manually run:  rm -rf {ckpt_link} && ln -s {path} {ckpt_link}"
            )

    # yisol/IDM-VTON on HuggingFace ships subdirs at the repo root:
    #   densepose/  humanparsing/  openpose/  unet/  unet_encoder/  vae/ ...
    # The original gradio_demo expected them under ckpt/ — adjust to the
    # HF layout, which is what scripts/setup_idm_vton.sh fetches.
    if not (path / "densepose").exists():
        raise FileNotFoundError(
            f"IDM-VTON weights not found at {path}. "
            f"Expected subdirs: densepose/, humanparsing/, openpose/, unet/, etc. "
            f"Run scripts/setup_idm_vton.sh."
        )

    # Import IDM-VTON's custom modules (only available once sys.path includes the clone)
    from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
    from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
    from src.unet_hacked_tryon import UNet2DConditionModel
    # In the cloned IDM-VTON repo:
    #   preprocess/humanparsing/run_parsing.py  → class Parsing
    #   preprocess/openpose/run_openpose.py      → class OpenPose
    from preprocess.humanparsing.run_parsing import Parsing
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
    ).requires_grad_(False)

    unet_encoder = UNet2DConditionModel_ref.from_pretrained(
        base, subfolder="unet_encoder", torch_dtype=dtype,
    ).requires_grad_(False)

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        base, subfolder="image_encoder", torch_dtype=dtype,
    ).requires_grad_(False)

    vae = AutoencoderKL.from_pretrained(
        base, subfolder="vae", torch_dtype=dtype,
    ).requires_grad_(False)

    text_encoder_one = CLIPTextModel.from_pretrained(
        base, subfolder="text_encoder", torch_dtype=dtype,
    ).requires_grad_(False)
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
        base, subfolder="text_encoder_2", torch_dtype=dtype,
    ).requires_grad_(False)

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
    # Custom partial CPU offload for 16 GB cards (RTX 4060 Ti / 4070).
    #
    # enable_model_cpu_offload() hit a device-mismatch error: the pipeline
    # calls unet.encoder_hid_proj (IP-Adapter resampler submodule) BEFORE
    # unet.__call__, so accelerate's pre-forward hook on `unet` didn't fire
    # and that submodule stayed on CPU while inputs were on CUDA. The
    # pre-move workaround didn't stick — accelerate's bookkeeping fights
    # manual .to(device) on hooked submodules.
    #
    # New strategy: keep the UNet stack (unet + unet_encoder) permanently
    # on GPU so all UNet submodules (including encoder_hid_proj) are always
    # on GPU. Only the lighter components (text_encoder × 2, vae,
    # image_encoder) offload — they swap in for their single forward each.
    #
    # VRAM budget (FP16, 16 GB card):
    #   unet                ~5 GB resident
    #   unet_encoder        ~5 GB resident
    #   vae+text+image      ~1 GB peak (one at a time, accelerate hook)
    #   activations 768x1024 ~3 GB peak
    #   total peak          ~14 GB — fits in 16 GB with headroom
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory < 20 * 1024**3:
        log.info("partial CPU offload: UNet stack resident, light components swap in on demand")
        from accelerate import cpu_offload_with_hook
        # UNet + UNet encoder stay on GPU (all submodules including
        # encoder_hid_proj are now GPU-resident).
        pipe.unet.to(device)
        if hasattr(pipe, "unet_encoder") and pipe.unet_encoder is not None:
            pipe.unet_encoder.to(device)
        # Light components: independent hooks (NO prev_module_hook chain).
        # The chained version was leaving hooks in inconsistent state after
        # the first inference completed — second call hung. Independent
        # hooks just lazily move each component to GPU when called and
        # back to CPU after, no inter-component state.
        for name in ("text_encoder", "text_encoder_2", "image_encoder", "vae"):
            component = getattr(pipe, name, None)
            if component is None:
                continue
            component.to("cpu")
            cpu_offload_with_hook(component, execution_device=device)
    else:
        pipe.to(device)

    # Pre-processing models. Both classes take gpu_id (int), NOT a path —
    # they find their weights via IDM-VTON/ckpt/<subdir>/... which we
    # symlinked above to the actual weights dir.
    log.info("loading OpenPose + HumanParsing pre-processors…")
    openpose = OpenPose(0)
    openpose.preprocessor.body_estimation.model.to(device)
    parsing = Parsing(0)

    log.info("IDM-VTON pipeline ready")
    return IDMVTONPipe(
        pipe=pipe, parsing_model=parsing, openpose_model=openpose,
        device=device, tensor_dtype=dtype,
    )
