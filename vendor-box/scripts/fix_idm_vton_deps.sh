#!/usr/bin/env bash
# Downgrade diffusers + transformers to versions IDM-VTON's custom UNet code
# expects. Without this, IDM-VTON load fails with:
#   ImportError: cannot import name 'PositionNet' from 'diffusers.models.embeddings'
#
# IDM-VTON was pinned against diffusers==0.25.0 / transformers==4.36.2 at
# release time. Their custom unet_hacked_garmnet.py imports an internal
# class (PositionNet) that was renamed in diffusers >=0.27.
#
# DM-VTON does NOT use diffusers, so this downgrade is safe for it.
set -euo pipefail

say() { printf "\n\033[1;34m[fix-idm-vton]\033[0m %s\n" "$*"; }
ok()  { printf "\033[1;32m  ✓ %s\033[0m\n" "$*"; }
die() { printf "\033[1;31m  ✗ %s\033[0m\n" "$*"; exit 1; }

[[ "${VIRTUAL_ENV:-}" ]] || die "activate the venv first: source venv/bin/activate"

say "downgrading diffusers + transformers + huggingface_hub for IDM-VTON"
# IDM-VTON pins:
#   diffusers==0.25.0   transformers==4.36.2   accelerate==0.25.0
# diffusers 0.25 imports `huggingface_hub.cached_download`, removed in
# huggingface_hub 0.27. Pin hub <0.26 to keep that import alive.
pip install \
  'diffusers==0.25.0' \
  'transformers==4.36.2' \
  'accelerate==0.25.0' \
  'huggingface_hub<0.26'

ok "done — restart uvicorn; IDM-VTON should now load"
