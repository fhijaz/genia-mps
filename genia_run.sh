#!/usr/bin/env bash
# GenIA (MPS port) runner: single-image setting on the Mac Studio.
#   ~/alt3d/genia_run.sh image [--scene NAME] [hydra overrides...]
# Wraps demo/run_demo.py with the layered genia_env and the fork-parity env.
set -e
export SPARSE_BACKEND=native
export SAM3D_CONV_IMPL=vec
export SAM3D_MPS_USE_SDPA=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTORCH_ENABLE_MPS_FALLBACK=1
export GENIA_SAM3D_WEIGHTS=/Users/fhijaz/alt3d/sam3d-objects-mac/checkpoints/hf/checkpoints
exec /Users/fhijaz/alt3d/genia_env/bin/python /Users/fhijaz/alt3d/GenIA/demo/run_demo.py "$@"
