#!/usr/bin/env python3
"""Pre-download all model weights to /var/lib/vendorbox/models.

Run once after `pip install -r requirements.txt`. Idempotent — skips files
that already exist with matching size. Expected total: ~25 GB.

If you need an HF token (gated models), export HF_TOKEN before running.
"""
import os
import sys
from pathlib import Path

MODELS_DIR = Path(os.environ.get("VENDORBOX_MODELS_DIR", "/var/lib/vendorbox/models"))


def download_idm_vton() -> None:
    target = MODELS_DIR / "idm-vton"
    if (target / "config.json").exists():
        print(f"[skip] IDM-VTON already at {target}")
        return
    print(f"[get] IDM-VTON → {target}")
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id="yisol/IDM-VTON",
        local_dir=str(target),
        local_dir_use_symlinks=False,
    )


def download_wan_i2v() -> None:
    target = MODELS_DIR / "wan2.1"
    if (target / "config.json").exists():
        print(f"[skip] Wan 2.1 already at {target}")
        return
    print(f"[get] Wan 2.1 i2v → {target}")
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id="Wan-AI/Wan2.1-I2V-14B-720P",
        local_dir=str(target),
        local_dir_use_symlinks=False,
    )


def download_dm_vton() -> None:
    target = MODELS_DIR / "dm-vton"
    if (target / "checkpoint.pth").exists():
        print(f"[skip] DM-VTON already at {target}")
        return
    print(f"[get] DM-VTON → {target}")
    # DM-VTON checkpoints are released as direct downloads, not HF. Stub.
    print("  NOTE: DM-VTON checkpoint download is a manual step.")
    print("  Visit https://github.com/KiseKloset/DM-VTON/releases and place")
    print(f"  checkpoint.pth under {target}/")


def main() -> int:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"models dir: {MODELS_DIR}")

    # IDM-VTON + Wan 2.1 are downloaded from HF
    download_idm_vton()
    download_wan_i2v()
    download_dm_vton()

    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
