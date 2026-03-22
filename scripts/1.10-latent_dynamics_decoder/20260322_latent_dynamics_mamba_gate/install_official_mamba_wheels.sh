#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
WHEEL_DIR="${WHEEL_DIR:-/root/autodl-tmp/wheels}"

CAUSAL_WHEEL_URL="https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.1.post4/causal_conv1d-1.6.1%2Bcu12torch2.10cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"
CAUSAL_WHEEL_NAME="causal_conv1d-1.6.1+cu12torch2.10cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"
MAMBA_WHEEL_URL="https://github.com/state-spaces/mamba/releases/download/v2.3.1/mamba_ssm-2.3.1%2Bcu12torch2.10cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"
MAMBA_WHEEL_NAME="mamba_ssm-2.3.1+cu12torch2.10cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"

mkdir -p "$WHEEL_DIR"

wget -nv -O "$WHEEL_DIR/$CAUSAL_WHEEL_NAME" "$CAUSAL_WHEEL_URL"
wget -nv -O "$WHEEL_DIR/$MAMBA_WHEEL_NAME" "$MAMBA_WHEEL_URL"

"$PYTHON" -m pip install --no-deps --force-reinstall \
  "$WHEEL_DIR/$CAUSAL_WHEEL_NAME" \
  "$WHEEL_DIR/$MAMBA_WHEEL_NAME"
