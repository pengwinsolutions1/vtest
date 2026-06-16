#!/usr/bin/env bash
# Install CatVTON (Zheng-Chong/CatVTON, ICLR 2025) for fast photoreal try-on.
#
# CatVTON is ~10x faster than IDM-VTON on the same hardware: 899M params,
# SD 1.5 backbone, no text encoder / pose detector / human parser needed.
# Expect ~4-6s per inference at 20 steps on RTX 4060 Ti 16 GB.
#
# License: CC BY-NC-SA 4.0 (non-commercial). For paid deployment contact
# the authors via https://github.com/Zheng-Chong/CatVTON#license-and-citation.
set -euo pipefail

say() { printf "\n\033[1;34m[catvton]\033[0m %s\n" "$*"; }
ok()  { printf "\033[1;32m  ✓ %s\033[0m\n" "$*"; }
die() { printf "\033[1;31m  ✗ %s\033[0m\n" "$*"; exit 1; }

[[ "${VIRTUAL_ENV:-}" ]] || die "activate the venv first: source venv/bin/activate"

HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"

TARGET="${VENDORBOX_MODELS_DIR:-$HERE/data/models}/catvton"
mkdir -p "$TARGET"

# ── 1. Clone the repo ────────────────────────────────────────────────────
say "1/3 — clone CatVTON repo"
if [[ -d "$TARGET/CatVTON/.git" ]]; then
  (cd "$TARGET/CatVTON" && git pull --ff-only) || true
  ok "CatVTON already cloned (pulled latest)"
else
  git clone --depth 1 https://github.com/Zheng-Chong/CatVTON.git "$TARGET/CatVTON"
  ok "cloned CatVTON"
fi

# ── 2. Install Python deps ───────────────────────────────────────────────
say "2/3 — install CatVTON extras"
# CatVTON's environment.yaml lists conda packages; pip equivalents:
pip install \
  'diffusers>=0.29,<0.32' \
  'accelerate>=0.30' \
  'transformers>=4.40' \
  'fvcore' \
  'omegaconf'
ok "CatVTON deps installed"

# ── 3. Download weights from HuggingFace ─────────────────────────────────
say "3/3 — download CatVTON weights (~1.5 GB)"
WEIGHTS="$TARGET/weights"
mkdir -p "$WEIGHTS"

# Honour the same HF mirror fallback we use for IDM-VTON
if [[ -z "${HF_ENDPOINT:-}" ]]; then
  if ! curl -sf --max-time 5 -o /dev/null https://huggingface.co/; then
    if curl -sf --max-time 5 -o /dev/null https://hf-mirror.com/; then
      warn "huggingface.co unreachable — using hf-mirror.com"
      export HF_ENDPOINT=https://hf-mirror.com
    fi
  fi
fi

if command -v hf > /dev/null; then
  hf download zhengchong/CatVTON --local-dir "$WEIGHTS"
elif command -v huggingface-cli > /dev/null; then
  huggingface-cli download zhengchong/CatVTON --local-dir "$WEIGHTS"
else
  python - <<PY
from huggingface_hub import snapshot_download
snapshot_download(repo_id="zhengchong/CatVTON", local_dir="$WEIGHTS")
PY
fi
ok "weights at $WEIGHTS"

say "DONE — CatVTON ready. Restart the server:"
echo "  LOAD_CATVTON=1 SNAPSHOT_MODEL=catvton ./scripts/run-server.sh"
