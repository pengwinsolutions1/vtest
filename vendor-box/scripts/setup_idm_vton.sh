#!/usr/bin/env bash
# Install IDM-VTON on the vendor box.
#
# IDM-VTON is not a pure pip package — it ships custom Stable Diffusion XL
# UNet modules that have to be imported from the cloned repo. This script:
#   1. Clones the official IDM-VTON repo into ./IDM-VTON
#   2. Downloads model weights (~10 GB) from HuggingFace into ./data/models/idm-vton
#   3. Installs IDM-VTON's extra Python deps into the active venv
#
# Run from vendor-box/ after activating venv. Idempotent — safe to re-run.
set -euo pipefail

say() { printf "\n\033[1;34m[idm-vton]\033[0m %s\n" "$*"; }
ok()  { printf "\033[1;32m  ✓ %s\033[0m\n" "$*"; }
die() { printf "\033[1;31m  ✗ %s\033[0m\n" "$*"; exit 1; }

[[ "${VIRTUAL_ENV:-}" ]] || die "activate the venv first: source venv/bin/activate"

HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"

# ── 1. Clone the IDM-VTON repo (or pull latest if already cloned) ────────
say "1/3 — clone IDM-VTON repo"
if [[ -d IDM-VTON/.git ]]; then
  (cd IDM-VTON && git pull --ff-only) || true
  ok "IDM-VTON already cloned (pulled latest)"
else
  git clone --depth 1 https://github.com/yisol/IDM-VTON.git
  ok "cloned IDM-VTON"
fi

# ── 2. Install IDM-VTON's extra Python deps ──────────────────────────────
say "2/3 — install IDM-VTON extras"
# IDM-VTON's environment.yaml lists conda packages. The pip-equivalents we
# need on top of our base requirements:
pip install \
  'config==0.5.1' \
  'einops>=0.7' \
  'ninja' \
  'cloudpickle' \
  'omegaconf' \
  'av' \
  'fvcore' \
  'iopath' \
  'matplotlib' \
  'opencv-python-headless>=4.9' \
  'scikit-image' \
  'scipy' \
  'tqdm'
ok "IDM-VTON extras installed"

# ── 3. Download weights from HuggingFace ──────────────────────────────────
say "3/3 — download model weights (~10 GB, can take 10-20 min)"
TARGET="${VENDORBOX_MODELS_DIR:-$HERE/data/models}/idm-vton"
mkdir -p "$TARGET"

# HF rate-limits unauthenticated downloads aggressively (esp. for large
# repos). Check the token is set up — either via HF_TOKEN env var or a
# cached login in ~/.cache/huggingface/token.
# Test connectivity to HuggingFace. Some networks (regional firewalls, ISP
# throttling, etc.) block huggingface.co outright — fall back to the public
# hf-mirror.com which proxies all repos and is reachable from most regions.
if [[ -z "${HF_ENDPOINT:-}" ]]; then
  if ! curl -sf --max-time 5 -o /dev/null https://huggingface.co/; then
    if curl -sf --max-time 5 -o /dev/null https://hf-mirror.com/; then
      warn "huggingface.co unreachable from this box — falling back to hf-mirror.com"
      export HF_ENDPOINT=https://hf-mirror.com
    else
      die "Neither huggingface.co nor hf-mirror.com is reachable. Check network/firewall/DNS on this box."
    fi
  fi
fi

TOKEN_CACHE="$HOME/.cache/huggingface/token"
if [[ -z "${HF_TOKEN:-}" && ! -s "$TOKEN_CACHE" ]]; then
  cat <<EOF

HuggingFace authentication required for the IDM-VTON download. Without a
token, requests hit aggressive rate limits.

To authenticate, either:

  Option A — login once (recommended):
    hf auth login
    # paste a token from https://huggingface.co/settings/tokens
    # (Type: 'Read' is sufficient. No special scopes needed.)

  Option B — pass the token as an env var each run:
    export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
    ./scripts/setup_idm_vton.sh

If you don't have an HF account, create one at https://huggingface.co/join
then generate a read token at https://huggingface.co/settings/tokens.

EOF
  die "no HF token found"
fi

# Prefer `hf` (new CLI). Fall back to legacy 'huggingface-cli', then
# Python's snapshot_download. All three pick up HF_TOKEN automatically.
if command -v hf > /dev/null; then
  hf download yisol/IDM-VTON --local-dir "$TARGET"
elif command -v huggingface-cli > /dev/null; then
  huggingface-cli download yisol/IDM-VTON --local-dir "$TARGET"
else
  python - <<PY
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="yisol/IDM-VTON",
    local_dir="$TARGET",
    token=os.environ.get("HF_TOKEN"),
)
PY
fi

# IDM-VTON also needs DensePose + a couple of pre-processing models, all
# inside the same HF repo under ckpt/. snapshot_download grabs them.
[[ -d "$TARGET/ckpt/densepose" ]] || say "WARN: densepose checkpoints not found — IDM-VTON pre-processing will fail"

ok "weights at $TARGET"

say "DONE — IDM-VTON ready. Restart uvicorn and check /healthz."
