#!/usr/bin/env bash
# One-shot environment setup for xarm-teleop on a fresh machine.
#
#   bash scripts/install_env.sh [env_name]      # default env name: xarm-teleop
#
# Requires: conda + an NVIDIA driver supporting CUDA 11.8. Validated on Python 3.13.
# The order matters (see the notes in requirements.txt): torch(cu118) -> numpy -> chumpy
# (no build isolation) -> project deps -> WiLoR-mini (no-deps).
set -euo pipefail

ENV="${1:-xarm-teleop}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo ">>> creating conda env '$ENV' (python 3.13)"
conda create -n "$ENV" python=3.13 pip setuptools -y
run() { conda run --no-capture-output -n "$ENV" "$@"; }

echo ">>> [1/5] PyTorch (CUDA 11.8)"
run pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118

echo ">>> [2/5] numpy (pinned so chumpy and opencv agree on one ABI)"
run pip install "numpy==2.2.6"

echo ">>> [3/5] chumpy (--no-build-isolation: its setup.py imports pip, absent in isolated builds)"
run pip install --no-build-isolation "chumpy @ git+https://github.com/mattloper/chumpy"

echo ">>> [4/5] project dependencies"
run pip install -r "$HERE/requirements.txt"

echo ">>> [5/5] WiLoR-mini (--no-deps; its deps were installed above)"
run pip install --no-deps "git+https://github.com/warmshao/WiLoR-mini"

echo ">>> verifying imports"
run python -c "import torch, numpy, pyrealsense2, mujoco, cv2, chumpy, smplx, timm, ultralytics, serial; \
from xarm.wrapper import XArmAPI; \
from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline; \
print('xarm-teleop env OK -- torch', torch.__version__, '| cuda', torch.cuda.is_available())"

echo ">>> done. Activate with:  conda activate $ENV"
echo ">>> smoke test:            python scripts/teleop.py wilor-image"
