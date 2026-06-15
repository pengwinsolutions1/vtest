#!/usr/bin/env bash
# Wrapper that starts the vendor-box service with the right LD_LIBRARY_PATH
# so CuPy can dlopen the pip-installed libnvrtc.so.12.
#
# Usage:
#   ./scripts/run-server.sh                       # default :8000
#   ./scripts/run-server.sh --port 8080 --workers 1
set -euo pipefail

[[ "${VIRTUAL_ENV:-}" ]] || { echo "activate the venv first: source venv/bin/activate"; exit 1; }

# Locate every nvidia/* package that ships .so files in its lib/ subdir and
# add all of them to LD_LIBRARY_PATH. This catches nvrtc, cublas, cudnn, etc.
# that pip wheels distribute self-contained.
NVIDIA_LIBS=$(python - <<'PY'
import importlib.util, os, pkgutil
roots = set()
try:
    import nvidia
    for finder, name, ispkg in pkgutil.iter_modules(nvidia.__path__, prefix="nvidia."):
        try:
            spec = importlib.util.find_spec(name)
            if not spec or not spec.origin:
                continue
            lib = os.path.join(os.path.dirname(spec.origin), "lib")
            if os.path.isdir(lib):
                roots.add(lib)
        except Exception:
            pass
except ImportError:
    pass
print(":".join(sorted(roots)))
PY
)

if [[ -n "$NVIDIA_LIBS" ]]; then
  export LD_LIBRARY_PATH="$NVIDIA_LIBS:${LD_LIBRARY_PATH:-}"
  echo "[run-server] LD_LIBRARY_PATH adds: $NVIDIA_LIBS"
fi

# PyTorch's default CUDA allocator fragments easily under model-cpu-offload
# (lots of allocs/frees as components swap). expandable_segments cuts the
# fragmentation cost dramatically — required to fit IDM-VTON on a 16 GB GPU.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

exec uvicorn server:app --host 0.0.0.0 --port "${PORT:-8000}" "$@"
