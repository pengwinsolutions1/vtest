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
# cupy is a CuPy-CUDA GPU array library that DM-VTON's correlation kernel
# uses internally for the warping math. Wheel name depends on the CUDA
# major version (cu11 vs cu12).
#
# IMPORTANT: pin to <13. DM-VTON uses cupy.cuda.compile_with_cache, which
# was deprecated in CuPy 11.0 and REMOVED in CuPy 13.0. With CuPy 13+ you
# get an AttributeError at the first frame. CuPy 12.x still has the old
# API (deprecation warnings, but functional).
CUPY_SPEC="cupy-cuda12x<13"
if command -v nvidia-smi > /dev/null; then
  CUDA_MAJ=$(nvidia-smi 2>/dev/null | grep -oE 'CUDA Version: [0-9]+' | grep -oE '[0-9]+' || true)
  case "$CUDA_MAJ" in
    11) CUPY_SPEC="cupy-cuda11x<13" ;;
    12) CUPY_SPEC="cupy-cuda12x<13" ;;
    *)  echo "  (couldn't detect CUDA major version, defaulting to $CUPY_SPEC)" ;;
  esac
fi
echo "  installing $CUPY_SPEC"

# opencv-python-headless 4.13+ requires numpy>=2. We pin numpy<2 elsewhere
# in this stack because mediapipe + DM-VTON's own ABI expect 1.x. Cap opencv
# to <4.13 to keep the numpy 1 environment consistent.
# tifffile (pulled in by scikit-image) has the same story — pin <2025 wheel.
pip install \
  "$CUPY_SPEC" \
  'tensorboard' \
  'opencv-python-headless>=4.9,<4.13' \
  'scikit-image>=0.22,<0.25' \
  'tifffile<2025' \
  'pillow>=10.0' \
  'tqdm'
ok "DM-VTON extras installed (incl. $CUPY_SPEC)"

# ── 3. Pretrained checkpoints (MANUAL STEP) ────────────────────────────────
say "3/3 — pretrained checkpoints"
TARGET="${VENDORBOX_MODELS_DIR:-$HERE/data/models}/dm-vton"
mkdir -p "$TARGET"

# Match by substring — DM-VTON ships multiple checkpoint variants under
# varying names (mobile_warp.pt, pf_warp.pt, etc.). The loader picks the
# first file matching *warp*.{pt,pth} and *gen*.{pt,pth}.
shopt -s nullglob nocaseglob   # nocaseglob so "Warp.pt" matches too
WARP_FILES=( "$TARGET"/*warp*.pt "$TARGET"/*warp*.pth )
GEN_FILES=(  "$TARGET"/*gen*.pt  "$TARGET"/*gen*.pth  )
shopt -u nocaseglob nullglob

echo "  searching in: $TARGET"
if [[ -d "$TARGET" ]]; then
  ls -lh "$TARGET" | sed 's/^/    /'
else
  echo "    (directory does not exist yet — creating)"
  mkdir -p "$TARGET"
fi

if (( ${#WARP_FILES[@]} > 0 && ${#GEN_FILES[@]} > 0 )); then
  ok "warp + gen checkpoints found:"
  for f in "${WARP_FILES[@]}" "${GEN_FILES[@]}"; do
    echo "    $(basename "$f")"
  done
  say "DONE — DM-VTON ready. Restart uvicorn and check /healthz."
  exit 0
fi

MISSING=()
(( ${#WARP_FILES[@]} == 0 )) && MISSING+=("*warp*.{pt,pth}")
(( ${#GEN_FILES[@]}  == 0 )) && MISSING+=("*gen*.{pt,pth}")

warn "missing checkpoint patterns: ${MISSING[*]}"
cat <<EOF

DM-VTON is an OPTIONAL quality upgrade for LIVE mode (~50ms/frame instead
of the IDM-VTON fallback at ~3-5s/frame). The service runs fine without it.

If you want the full DM-VTON speed later:
  1. Open https://drive.google.com/drive/folders/1wfWGsR0vWC5LrA26xhj92ec_GoCKV80A
  2. Download the .pt files (e.g. mobile_warp.pt + mobile_gen.pt, or whatever
     the latest variant is — the loader matches by name substring)
  3. Place them at: $TARGET
  4. Restart uvicorn — /healthz will then report "dm_vton": true

For now, this script is exiting successfully. The /ws/live endpoint will
use IDM-VTON at low-step config as a quasi-live fallback.

EOF
ok "DM-VTON skipped (optional) — using IDM-VTON live fallback"
exit 0
