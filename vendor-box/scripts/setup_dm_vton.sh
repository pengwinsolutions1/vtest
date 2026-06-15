#!/usr/bin/env bash
# Install DM-VTON for real-time LIVE virtual try-on (~50-100ms/frame on RTX 4080).
#
# DM-VTON's pretrained weights are distributed via Google Drive links in the
# upstream README — there's no scriptable download URL. We clone the repo +
# install deps automatically; weights are a manual one-time step.
set -euo pipefail

say() { printf "\n\033[1;34m[dm-vton]\033[0m %s\n" "$*"; }
ok()  { printf "\033[1;32m  ✓ %s\033[0m\n" "$*"; }
warn(){ printf "\033[1;33m  ⚠ %s\033[0m\n" "$*"; }
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

# ── 3. Pretrained checkpoints (MANUAL STEP) ────────────────────────────────
say "3/3 — pretrained checkpoints"
TARGET="${VENDORBOX_MODELS_DIR:-$HERE/data/models}/dm-vton"
mkdir -p "$TARGET"

REQUIRED=(checkpoint_warp.pth checkpoint_gen.pth)
MISSING=()
for ckpt in "${REQUIRED[@]}"; do
  if [[ -f "$TARGET/$ckpt" ]]; then
    ok "  $ckpt present"
  else
    MISSING+=("$ckpt")
  fi
done

if [[ ${#MISSING[@]} -eq 0 ]]; then
  ok "all weights present"
  say "DONE — DM-VTON ready. Restart uvicorn and check /healthz."
  exit 0
fi

warn "missing weight files: ${MISSING[*]}"
cat <<EOF

DM-VTON's authors distribute their pretrained checkpoints via Google Drive
links in their README — there's no scriptable URL we can curl.

To finish setup:

  1. Open https://github.com/KiseKloset/DM-VTON#-pretrained-models
  2. Download the checkpoints linked there. The ones you need are:
       ${MISSING[*]}
     (also any 'warp' and 'gen' / 'tryon' model the README mentions for
      their best-quality config)
  3. Move the downloaded files to:
       $TARGET
  4. Re-run this script — it will skip the download and proceed.

If the README has renamed the checkpoints (the project files shift across
releases), put them in the target dir under the names listed above (or
adjust dm_vton_loader.py to match the new names).

Alternative: a HuggingFace community mirror of DM-VTON may exist. If you
find one, you can also do:
  huggingface-cli download <repo-id> --local-dir "$TARGET"

EOF
die "manual download step required"
