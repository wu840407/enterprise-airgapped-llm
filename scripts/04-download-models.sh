#!/bin/bash
# scripts/04-download-models.sh
#
# Phase 4 — Pull foundation model weights from Hugging Face into the data tier.
#
# Default model: Qwen3-Coder-30B-A3B-Instruct (AWQ quantization).
#   - Apache 2.0 licensed, 30B-parameter MoE coding model
#   - ~17 GB on disk after AWQ INT4 quantization
#   - Fits within 48 GB total VRAM on 2× 24 GB Turing cards
#
# This script only runs in the ONLINE phase. Once the model is on disk and
# verified, the host can be migrated to the air-gapped network.

set -euo pipefail

MODEL_REPO="${MODEL_REPO:-stelterlab/Qwen3-Coder-30B-A3B-Instruct-AWQ}"
MODEL_DIR_NAME="${MODEL_DIR_NAME:-Qwen3-Coder-30B-A3B-Instruct-AWQ}"
TARGET_DIR="/opt/ai/models"
VENV_DIR="${HOME}/hf-env"

# -----------------------------------------------------------------
# 1. Pre-flight
# -----------------------------------------------------------------
if [[ ! -d "$TARGET_DIR" ]]; then
  echo "ERROR: $TARGET_DIR does not exist. Run 02-raid-setup.sh first."
  exit 1
fi

if ! ping -c1 -W3 huggingface.co >/dev/null 2>&1; then
  echo "ERROR: Cannot reach huggingface.co. Models can only be downloaded in"
  echo "       the online phase, before the host is moved to the air-gapped"
  echo "       network."
  exit 1
fi

# -----------------------------------------------------------------
# 2. Python virtualenv with the Hugging Face CLI
# -----------------------------------------------------------------
echo "==> [1/3] Preparing Python virtualenv at $VENV_DIR"
if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
pip install --upgrade pip
pip install --upgrade "huggingface_hub[cli]" hf_transfer

# Enable the multi-threaded transfer accelerator. Significantly faster on
# fast networks; falls back gracefully if unavailable.
export HF_HUB_ENABLE_HF_TRANSFER=1

# -----------------------------------------------------------------
# 3. Download
# -----------------------------------------------------------------
echo "==> [2/3] Downloading $MODEL_REPO → $TARGET_DIR/$MODEL_DIR_NAME"
echo "         (typical size: 17–18 GiB; 5–30 min depending on bandwidth)"

cd "$TARGET_DIR"
hf download "$MODEL_REPO" \
  --local-dir "$MODEL_DIR_NAME" \
  --max-workers 8

# -----------------------------------------------------------------
# 4. Sanity check
# -----------------------------------------------------------------
echo "==> [3/3] Verifying download"
SIZE=$(du -sh "$TARGET_DIR/$MODEL_DIR_NAME" | awk '{print $1}')
echo "Final size: $SIZE"
ls -la "$TARGET_DIR/$MODEL_DIR_NAME" | head -20

if [[ ! -f "$TARGET_DIR/$MODEL_DIR_NAME/config.json" ]]; then
  echo "ERROR: config.json missing. Download incomplete."
  exit 1
fi

cat <<'EOM'

==============================================================================
Model download complete.

The model directory is referenced by docker-compose.yml as:
  /models/Qwen3-Coder-30B-A3B-Instruct-AWQ

If you used a different MODEL_DIR_NAME, update the command: section in
docker-compose.yml accordingly.

You can now proceed to 05-deploy.sh, or migrate the host to the air-gapped
network before bringing the stack up.
==============================================================================
EOM
