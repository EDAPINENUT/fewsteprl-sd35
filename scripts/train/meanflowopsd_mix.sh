#!/usr/bin/env bash
# Launch the `meanflowopsd_mix` experiment.
#
# DiffusionOPSD under MeanFlow, with the query noise level drawn per sample
# around sigma* = 0.278 (opsd_sigma_mode=lognormal), r drawn given t
# (opsd_r_mode=given_t), and the target direction taken from the weighted
# training reward (opsd_reward_kind=mixed).
# disable_cd=true skips the central difference (S := 0): S = (t - r) * dF/dt
# cancels between the OPSD prediction and its anchor, but not in the stop-grad
# target coefficients, so this is close to, not bit-equivalent with, the
# with-CD objective (see FewstepRLTrainConfig.disable_cd).
#
# Script name, config filename and checkpoint directory all use the SAME
# experiment name `meanflowopsd_mix`.
#
#   config     : scripts/train/configs/meanflowopsd_mix.yaml
#   trainer    : meanflowopsd_rl
#   released   : checkpoints/meanflowopsd_mix/checkpoint-400/generator_ema.pt
#
# Usage:
#   bash scripts/train/meanflowopsd_mix.sh [NUM_GPUS] [EXTRA_OVERRIDES...]
#   bash scripts/train/meanflowopsd_mix.sh 8 fewstep_rl.num_epochs=10     # 10-iter smoke
#
# TRAINER_KEY is a CLI argument and is NOT written into config.yaml -- reading
# the config alone cannot tell you which algorithm ran. That is why it is
# pinned here rather than left to a default.

set -euo pipefail

ARM='meanflowopsd_mix'
TRAINER='meanflowopsd_rl'
PORT=29619

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
