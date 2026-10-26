#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Clean Linux GPU environment; avoid reusing the Windows .venv.
python3 -m venv .venv-cloud
source .venv-cloud/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip check
nvidia-smi
python scripts/download_model.py
python scripts/preflight.py
printf '\nReady. Activate with: source .venv-cloud/bin/activate\n'
