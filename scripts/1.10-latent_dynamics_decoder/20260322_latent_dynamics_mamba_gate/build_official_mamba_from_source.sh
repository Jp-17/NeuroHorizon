#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
BUILD_DIR="${BUILD_DIR:-/root/autodl-tmp/tmp_mamba_build}"
TMPDIR="${TMPDIR:-/root/autodl-tmp/tmp_pip_build}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.4}"

export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export TMPDIR
export MAX_JOBS="${MAX_JOBS:-8}"

mkdir -p "$BUILD_DIR" "$TMPDIR"
cd "$BUILD_DIR"

wget -nv -O causal_conv1d-1.6.1.tar.gz \
  https://files.pythonhosted.org/packages/63/15/ec51d77a2df03ee93410f8ee97fceeb7181da213813c51243e9dd6d7e144/causal_conv1d-1.6.1.tar.gz
wget -nv -O mamba_ssm-2.3.1.tar.gz \
  https://files.pythonhosted.org/packages/34/67/ec89aa703da194a813e35d2ea2de8f74a7ce6991a120a29f3a0c5e30d4b9/mamba_ssm-2.3.1.tar.gz

tar -xf causal_conv1d-1.6.1.tar.gz
tar -xf mamba_ssm-2.3.1.tar.gz

"$PYTHON" - <<'PY'
from pathlib import Path
import re

paths = [
    Path('/root/autodl-tmp/tmp_mamba_build/causal_conv1d-1.6.1/setup.py'),
    Path('/root/autodl-tmp/tmp_mamba_build/mamba_ssm-2.3.1/setup.py'),
]
pattern = re.compile(
    r'\n\s*if bare_metal_version <= Version\(\"12\\.9\"\):.*?\n\n\s*# HACK:',
    re.S,
)
replacement = (
    '\n        cc_flag.extend([\n'
    '            "-gencode",\n'
    '            "arch=compute_89,code=sm_89",\n'
    '        ])\n\n'
    '    # HACK:'
)
for path in paths:
    text = path.read_text()
    new_text, count = pattern.subn(replacement, text, count=1)
    if count != 1:
        raise RuntimeError(f'failed to patch {path}')
    path.write_text(new_text)
    print(f'patched {path}')
PY

"$PYTHON" -m pip install ninja

export CAUSAL_CONV1D_FORCE_BUILD=TRUE
"$PYTHON" -m pip install --no-build-isolation --no-deps "$BUILD_DIR/causal_conv1d-1.6.1"
unset CAUSAL_CONV1D_FORCE_BUILD

export MAMBA_FORCE_BUILD=TRUE
"$PYTHON" -m pip install --no-build-isolation --no-deps "$BUILD_DIR/mamba_ssm-2.3.1"
