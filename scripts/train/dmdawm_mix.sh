#!/usr/bin/env bash
# Launch the `dmdawm_mix` experiment.
#
# DMD-style AWM: no central difference (x_anchor_2 = x_t - t*u_old, dF/dt is
# never computed). Aux v-predictor warm-started from checkpoints/aux_vel/mix.
#
# Script name, config filename and checkpoint directory all use the SAME
# experiment name `dmdawm_mix`.
#
#   config     : scripts/train/configs/dmdawm_mix.yaml
#   trainer    : dmdawm_rl
#   released   : checkpoints/dmdawm_mix/checkpoint-400/generator_ema.pt
#
# Usage:
#   bash scripts/train/dmdawm_mix.sh [NUM_GPUS] [EXTRA_OVERRIDES...]
#   bash scripts/train/dmdawm_mix.sh 8 fewstep_rl.num_epochs=10     # 10-iter smoke
#
# TRAINER_KEY is a CLI argument and is NOT written into config.yaml -- reading
# the config alone cannot tell you which algorithm ran. That is why it is
# pinned here rather than left to a default.

set -euo pipefail

TRIAL='dmdawm_mix'
TRAINER='dmdawm_rl'
PORT=29615

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO}"

NGPU="${1:-8}"; shift || true

export PRETRAIN_LORA="${PRETRAIN_LORA:-${REPO}/models/pretrained/meanflownft_loras/anyflow_pretrain/generator_ema.pt}"
export ONPOLICY_LORA="${ONPOLICY_LORA:-${REPO}/models/pretrained/meanflownft_loras/anyflow_onpolicy/generator_ema.pt}"
export TRAINER_KEY="${TRAINER}"
export CONFIG="scripts/train/configs/${TRIAL}.yaml"
export NNODES="${NNODES:-1}" NODE_RANK="${NODE_RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-localhost}" MASTER_PORT="${MASTER_PORT:-${PORT}}"
# Without network / a wandb api_key, an unset WANDB_MODE makes rank 0 raise
# inside wandb.init while the other ranks block in NCCL: the job hangs instead
# of failing. Export WANDB_MODE=online if you are logged in.
export WANDB_MODE="${WANDB_MODE:-offline}"

echo "[${TRIAL}] trainer=${TRAINER} config=${CONFIG} gpus=${NGPU} port=${MASTER_PORT}"
exec bash scripts/train/_launcher_decoupled.sh "${NGPU}" "$@"
