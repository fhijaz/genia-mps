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
# Relax the AGX context-store deadline that kills long MPS command buffers
# as kIOGPUCommandBufferCallbackErrorImpactingInteractivity (the macOS GPU
# watchdog that originally forced the SLAT mesh decode onto CPU). Confirmed
# workaround (mlx #3267; llama.cpp sets it by default in its Metal backend).
# It must be set before the process creates its Metal device, so it lives
# here rather than in Python. Harmless on this M1 machine. A value provided
# by the caller is respected.
export AGX_RELAX_CDM_CTXSTORE_TIMEOUT="${AGX_RELAX_CDM_CTXSTORE_TIMEOUT:-1}"
# Default GenIA runs to the mtlattn varlen flash kernel for the SLAT
# decoder windowed attention (adopted Nov 25, 2026; segment A/B marginal
# pass, finetune block ~4.0x). The fork windowed_attn.py gate reads
# SAM3D_ATTN_IMPL. A value provided by the caller is respected: set
# SAM3D_ATTN_IMPL=flash32 for the fp32 arm, or any other value (e.g.
# off) to skip the gate and use the original masked_sdpa path.
export SAM3D_ATTN_IMPL="${SAM3D_ATTN_IMPL:-flash32}"
export GENIA_SAM3D_WEIGHTS=/Users/fhijaz/alt3d/sam3d-objects-mac/checkpoints/hf/checkpoints
exec /Users/fhijaz/alt3d/genia_env/bin/python /Users/fhijaz/alt3d/GenIA/demo/run_demo.py "$@"
