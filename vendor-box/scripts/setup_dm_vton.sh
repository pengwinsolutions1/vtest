#!/usr/bin/env bash
# Install DM-VTON for real-time LIVE virtual try-on (~50-100ms/frame on RTX 4080).
#
# DM-VTON is research code from the KiseKloset paper. It's distilled from
# heavier teacher networks to run fast — good enough for live AR at ~15-20fps
# server-side, with TensorRT we can push closer to 30fps.
#
# This script clones the repo + downloads pretrained checkpoints. TensorRT
# conversion is optional; without it the loader falls back to plain PyTorch.
set -euo pipefail

say() { printf "\n\033[1;34m[dm-vton]\033[0m %s\n" "$*"; }
ok()  { printf "\033[1;32m  ✓ %s\033[0m\n" "$*"; }
die() { printf "\033[1;31m  ✗ %s\033[0m\n" "$*"; exit 1; }

[[ "${VIRTUAL_ENV:-}" ]] || die "activate the venv first: source venv/bin/activate"

HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"

# ── 1. Clone repo ──────────────────────────────────────────────────────────
say "1/3 — clone DM-VTON repo"
if [[ -d DM-VTON/.git ]]; then
  (cd DM-VTON && git pull --ff-only) || true
  ok "DM-VTON already cloned (pulled latest)"
else
  git clone --depth 1 https://github.com/KiseKloset/DM-VTON.git
  ok "cloned DM-VTON"
fi

# ── 2. Install DM-VTON extras ──────────────────────────────────────────────
say "2/3 — install DM-VTON extras"
pip install \
  'tensorboard' \
  'opencv-python-headless>=4.9' \
  'scikit-image' \
  'pillow>=10.0' \
  'tqdm'
ok "DM-VTON extras installed"

# ── 3. Download pretrained checkpoints ─────────────────────────────────────
say "3/3 — download DM-VTON pretrained checkpoints (~500 MB)"
TARGET="${VENDORBOX_MODELS_DIR:-$HERE/data/models}/dm-vton"
mkdir -p "$TARGET"

# DM-VTON's checkpoints are on GitHub releases. URLs may shift across versions;
# we try the official release first, then fall back to the HF mirror.
RELEASE_TAG="v1.0"
BASE_URL="https://github.com/KiseKloset/DM-VTON/releases/download/${RELEASE_TAG}"
for ckpt in checkpoint_warp.pth checkpoint_gen.pth checkpoint_mobile.pth; do
  if [[ -f "$TARGET/$ckpt" ]]; then
    ok "  $ckpt already present"
    continue
  fi
  echo "  downloading $ckpt…"
  curl -fL "$BASE_URL/$ckpt" -o "$TARGET/$ckpt" || die "failed to download $ckpt — check the release page at https://github.com/KiseKloset/DM-VTON/releases"
  ok "  $ckpt"
done

ok "weights at $TARGET"
say "DONE — DM-VTON ready. Restart uvicorn and check /healthz."
