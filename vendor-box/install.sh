#!/usr/bin/env bash
# Stage-by-stage installer for the vendor-box service. Failing fast and
# clearly is better than one giant `pip install -r requirements.txt` that
# blows up halfway and is impossible to debug.
#
# Usage:
#   source venv/bin/activate
#   ./install.sh
#
# Each `pip install` is its own resolver pass — if a stage fails, you see
# which one and can iterate without re-running the successful stages.
set -euo pipefail

say() { printf "\n\033[1;34m[install]\033[0m %s\n" "$*"; }
ok()  { printf "\033[1;32m  ✓ %s\033[0m\n" "$*"; }
die() { printf "\033[1;31m  ✗ %s\033[0m\n" "$*"; exit 1; }

[[ "${VIRTUAL_ENV:-}" ]] || die "activate the venv first: source venv/bin/activate"

say "0/5 — upgrade pip + build tools"
pip install --upgrade pip wheel setuptools
ok "pip $(pip --version | awk '{print $2}')"

say "1/5 — web service deps"
pip install \
  'fastapi>=0.110,<1' \
  'uvicorn[standard]>=0.30' \
  'pydantic>=2.6,<3' \
  'httpx>=0.27' \
  'python-multipart>=0.0.9' \
  'pillow>=10.0'
ok "web stack installed"

say "2/5 — numpy (pin <2 for mediapipe + diffusers ABI compatibility)"
pip install 'numpy>=1.24,<2'
ok "numpy installed"

say "3/5 — PyTorch + torchvision (CUDA 12.1 wheels)"
pip install --index-url https://download.pytorch.org/whl/cu121 \
  'torch>=2.4,<3' \
  'torchvision>=0.19,<1'
# Verify torch sees CUDA
python -c 'import torch; assert torch.cuda.is_available(), "torch installed but cuda.is_available()=False"; print(f"  torch={torch.__version__} CUDA={torch.version.cuda} device={torch.cuda.get_device_name(0)}")'
ok "PyTorch CUDA available"

say "4/5 — diffusers / transformers / accelerate"
pip install \
  'diffusers>=0.30' \
  'transformers>=4.45' \
  'accelerate>=1.0' \
  'safetensors>=0.4' \
  'einops>=0.7'
ok "diffusers stack installed"

say "5/5 — extras (mediapipe for body parsing, hub for downloads)"
pip install \
  'mediapipe>=0.10.14' \
  'huggingface_hub>=0.25' \
  'opencv-python-headless>=4.9'
ok "extras installed"

say "DONE — try: uvicorn server:app --host 0.0.0.0 --port 8000"
