#!/usr/bin/env bash
# Run in an activated Python 3.12 environment; cpu is useful for import checks.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
BUILD="${1:-cu126}"
case "$BUILD" in cpu|cu126) ;; *) echo 'Usage: bash scripts/install_environment.sh [cu126|cpu]' >&2; exit 2;; esac
if [[ $# -gt 1 ]]; then echo 'Expected at most one build argument' >&2; exit 2; fi
"$PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Activate a Python 3.12 environment"'
"$PYTHON" -m pip install --upgrade pip
"$PYTHON" -m pip install "torch==2.7.0+$BUILD" --index-url "https://download.pytorch.org/whl/$BUILD"
# Never fall back to an isolated source build without Torch.
"$PYTHON" -m pip install --only-binary=:all: --no-index \
  "torch-scatter==2.1.2+pt27$BUILD" -f "https://data.pyg.org/whl/torch-2.7.0+$BUILD.html"
"$PYTHON" -m pip install -r "$ROOT/requirements.txt"
"$PYTHON" -m pip check
"$PYTHON" -c 'import torch, torch_scatter, torch_geometric; from torch_scatter import scatter_sum; x = torch.tensor([1., 2.]); assert scatter_sum(x, torch.tensor([0, 0])).item() == 3.; print("torch", torch.__version__, "scatter", torch_scatter.__version__, "pyg", torch_geometric.__version__)'
PYTHONPATH="$ROOT/baselines/external/recbole${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" -c 'from recbole.quick_start import run; from recbole.utils import get_model; [get_model(name) for name in ("GCMC", "CKE", "KTUP", "KGCN")]; print("Vendored RecBole imports OK")'
