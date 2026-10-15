#!/usr/bin/env bash
# 在仓库根目录 source env.sh；不改变用户的 Python 虚拟环境。
ACT_DEMO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export LP_NUM_THREADS="${LP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$ACT_DEMO_ROOT/.cache}"
export HF_HOME="${HF_HOME:-$XDG_CACHE_HOME/huggingface}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$HF_HOME/lerobot}"
export TORCH_HOME="${TORCH_HOME:-$XDG_CACHE_HOME/torch}"
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
