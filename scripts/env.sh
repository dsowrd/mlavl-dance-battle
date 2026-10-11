#!/usr/bin/env bash
# Common environment for every run on Server 1. Source this first:
#   source scripts/env.sh
#   CUDA_VISIBLE_DEVICES=0 python main.py ...      # allowed GPUs: 0, 1, 3 only (GPU 2 is faulty, never use it)

# nvidia-smi numbering == PyTorch numbering
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# deterministic cuBLAS (main.py also sets this; kept here for scripts that call other entry points)
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# conda env (do NOT rely on `conda activate` inside non-interactive shells; use the explicit python)
export MLAVL_PY=/home/irl/anaconda3/envs/mlavl/bin/python

# data roots (large files live on the HDD; adjust once the HDD is mounted)
export MLAVL_DATA=${MLAVL_DATA:-/data/mlavl}

# refuse GPU 2
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && ",${CUDA_VISIBLE_DEVICES}," == *",2,"* ]]; then
  echo "ERROR: GPU 2 is banned on Server 1 (hardware error). Use 0, 1 or 3." >&2
  return 1 2>/dev/null || exit 1
fi
