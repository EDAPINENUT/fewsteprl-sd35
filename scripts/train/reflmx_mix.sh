#!/usr/bin/env bash
# Launch the `reflmx_mix` experiment.
#
# The ReFL corner of few-step DiffusionOPSD: keep the ascent square alone
# (opsd_n_ascent=1, opsd_dual_neg=false, no proximal spring), with the reward
# gradient taken from the weighted training reward (opsd_reward_kind=mixed).
# opsd_sigma_mode=lognormal centres the state on the low-noise end
# (sigma* = 0.278), the analogue of ReFL's last-25% draw.
# disable_cd=true skips the central difference (S := 0); S cancels in the
# differentiable channel but not in the stop-grad coefficients, so this is
# close to, not bit-equivalent with, the with-CD objective.
#
# Script name, config filename and checkpoint directory all use the SAME
# experiment name `reflmx_mix`.
#
#   config     : scripts/train/configs/reflmx_mix.yaml
#   trainer    : meanflowrefl_rl
#   released   : checkpoints/reflmx_mix/checkpoint-400/generator_ema.pt
#
# Usage:
#   bash scripts/train/reflmx_mix.sh [NUM_GPUS] [EXTRA_OVERRIDES...]
#   bash scripts/train/reflmx_mix.sh 8 fewstep_rl.num_epochs=10     # 10-iter smoke
#
# TRAINER_KEY is a CLI argument and is NOT written into config.yaml -- reading
# the config alone cannot tell you which algorithm ran. That is why it is
# pinned here rather than left to a default.

set -euo pipefail

ARM='reflmx_mix'
TRAINER='meanflowrefl_rl'
PORT=29621

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO}"

NGPU="${1:-8}"; shift || true

export PRETRAIN_LORA="${PRETRAIN_LORA:-${REPO}/models/pretrained/meanflownft_loras/anyflow_pretrain/generator_ema.pt}"
export ONPOLICY_LORA="${ONPOLICY_LORA:-${REPO}/models/pretrained/meanflownft_loras/anyflow_onpolicy/generator_ema.pt}"
export TRAINER_KEY="${TRAINER}"
export CONFIG="scripts/train/configs/${ARM}.yaml"
export NNODES="${NNODES:-1}" NODE_RANK="${NODE_RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-localhost}" MASTER_PORT="${MASTER_PORT:-${PORT}}"
# Without network / a wandb api_key, an unset WANDB_MODE makes rank 0 raise
# inside wandb.init while the other ranks block in NCCL: the job hangs instead
# of failing. Export WANDB_MODE=online if you are logged in.
export WANDB_MODE="${WANDB_MODE:-offline}"

echo "[${ARM}] trainer=${TRAINER} config=${CONFIG} gpus=${NGPU} port=${MASTER_PORT}"
exec bash scripts/train/_launcher_opsd.sh "${NGPU}" "$@"
