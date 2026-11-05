#!/bin/bash
# Shared launcher for the decoupled-loss family (every trial except the two OPSD
# ones, which use _launcher_opsd.sh). Not meant to be called directly: the
# scripts/train/<trial>.sh wrappers export CONFIG / TRAINER_KEY / the Stage-1+2
# LoRAs and exec this file.
#
# Prerequisites (see README.md, "Setup"):
#   bash download_scripts/download_assets.sh
#
# Usage (via a wrapper):
#   bash scripts/train/<trial>.sh [NUM_GPUS] [EXTRA_OVERRIDES...]
#
# Environment overrides:
#   FEWSTEPRL_SD35_PATH           SD3.5-Medium diffusers pipeline dir
#   FEWSTEPRL_REWARD_CKPT_PATH    reward checkpoint root (hpsv2 / aesthetic files)
#   FEWSTEPRL_PADDLEOCR_V5_DIR    PaddleOCR 3.x PP-OCRv5 model dirs
#   HF_HOME                       HF hub cache holding PickScore / CLIP
#   NNODES NODE_RANK MASTER_ADDR MASTER_PORT TORCHRUN_BIN

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

NUM_GPUS="${1:-8}"
EXTRA_ARGS=("${@:2}")
set --

[[ -n "${CONFIG:-}" ]] || { echo "ERROR: CONFIG is not set; launch through scripts/train/<trial>.sh" >&2; exit 1; }
[[ -n "${TRAINER_KEY:-}" ]] || { echo "ERROR: TRAINER_KEY is not set; launch through scripts/train/<trial>.sh" >&2; exit 1; }

# --- Local asset defaults (populated by download_scripts/download_assets.sh) --
SD35_PATH="${FEWSTEPRL_SD35_PATH:-${REPO_ROOT}/models/stable-diffusion-3.5-medium}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/models/hf_cache}"
export FEWSTEPRL_REWARD_CKPT_PATH="${FEWSTEPRL_REWARD_CKPT_PATH:-${REPO_ROOT}/models/reward_ckpts}"
export FEWSTEPRL_PADDLEOCR_V5_DIR="${FEWSTEPRL_PADDLEOCR_V5_DIR:-${REPO_ROOT}/models/paddleocr_v5}"

# --- Preflight: fail loudly BEFORE torchrun spawns N ranks -------------------
fail() { echo "ERROR: $*" >&2; exit 1; }

[[ -n "${PRETRAIN_LORA:-}" ]] || fail "PRETRAIN_LORA is not set (Stage 1 anyflow_pretrain/generator_ema.pt)."
[[ -n "${ONPOLICY_LORA:-}" ]] || fail "ONPOLICY_LORA is not set (Stage 2 anyflow_onpolicy/generator_ema.pt)."
[[ -r "${PRETRAIN_LORA}" ]] || fail "PRETRAIN_LORA not readable: ${PRETRAIN_LORA}"
[[ -r "${ONPOLICY_LORA}" ]] || fail "ONPOLICY_LORA not readable: ${ONPOLICY_LORA}"
[[ -f "${SD35_PATH}/model_index.json" ]] || fail "SD3.5 pipeline not found at ${SD35_PATH}"
for f in open_clip_pytorch_model.bin HPS_v2.1_compressed.pt sac+logos+ava1-l14-linearMSE.pth; do
    [[ -e "${FEWSTEPRL_REWARD_CKPT_PATH}/${f}" ]] \
        || fail "reward checkpoint missing: ${FEWSTEPRL_REWARD_CKPT_PATH}/${f}
  Run: bash download_scripts/download_assets.sh"
done
for repo in yuvalkirstain--PickScore_v1 laion--CLIP-ViT-H-14-laion2B-s32B-b79K openai--clip-vit-large-patch14; do
    [[ -d "${HF_HOME}/hub/models--${repo}" ]] \
        || fail "${repo/--//} not in ${HF_HOME}/hub (pickscore / clipscore / aesthetic).
  Run: bash download_scripts/download_assets.sh"
done
[[ -d "${FEWSTEPRL_PADDLEOCR_V5_DIR}/en_PP-OCRv5_mobile_rec" && ( -d "${FEWSTEPRL_PADDLEOCR_V5_DIR}/PP-OCRv5_mobile_det" || -d "${FEWSTEPRL_PADDLEOCR_V5_DIR}/PP-OCRv5_server_det" ) ]] \
    || fail "PaddleOCR PP-OCRv5 models not found under ${FEWSTEPRL_PADDLEOCR_V5_DIR} (needed by the ocr reward/metric).
  Run: bash download_scripts/download_assets.sh"

# The algorithm is selected by TRAINER_KEY, not by a config key, so only assert
# the section this config must carry: a wrong yaml then fails loudly instead of
# silently loading every fewstep_rl default. Also print the reward recipe the
# config trains on.
python3 - "${CONFIG}" <<'PYEOF' || exit 1
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
if "fewstep_rl" not in cfg:
    print(f"ERROR: {sys.argv[1]} has no 'fewstep_rl' section.", file=sys.stderr)
    sys.exit(1)
print("  Preflight OK: fewstep_rl section present")
print("  Train reward: " + " + ".join(
    f"{k} {v}" for k, v in cfg["fewstep_rl"].get("reward_fn", {}).items()))
PYEOF

# K-repeat constraint: batch_size * world_size % K == 0.
NNODES="${NNODES:-1}"
python3 - "${CONFIG}" "${NUM_GPUS}" "${NNODES}" <<'PYEOF' || exit 1
import sys, yaml
cfg_path, num_gpus, nnodes = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
cfg = yaml.safe_load(open(cfg_path))
bs = int(cfg["train"]["batch_size"])
K = int(cfg["fewstep_rl"]["num_image_per_prompt"])
total = bs * num_gpus * nnodes
if total % K:
    print(f"ERROR: train.batch_size * total_gpus = {bs} * {num_gpus * nnodes} = {total}", file=sys.stderr)
    print(f"       is not divisible by num_image_per_prompt = {K}.", file=sys.stderr)
    print("       Override one of them, e.g. add:", file=sys.stderr)
    for cand in (24, 12, 8, 6, 4, 3, 2):
        if total % cand == 0:
            print(f"         fewstep_rl.num_image_per_prompt={cand}"
                  f"   ({total // cand} prompts/step)", file=sys.stderr)
            break
    sys.exit(1)
print(f"  Preflight OK: {total} samples/step / K={K} = {total // K} prompts/step")
PYEOF

# --- Distributed / NCCL env ---------------------------------------------------
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29501}"

export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-2}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export TORCH_NCCL_AVOID_RECORD_STREAMS="${TORCH_NCCL_AVOID_RECORD_STREAMS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:256}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export GLOG_minloglevel="${GLOG_minloglevel:-2}"
# ``WandbLogger`` calls ``wandb.init`` without a mode argument
# (fewsteprl/utils/logging.py), so with no network / api_key an unset
# WANDB_MODE makes rank 0 raise inside ``train()`` while ranks 1..N block in
# NCCL -- the job HANGS until the watchdog fires instead of failing. Set
# WANDB_MODE=online explicitly if you are logged in.
export WANDB_MODE="${WANDB_MODE:-offline}"

TOTAL_GPUS=$((NUM_GPUS * NNODES))

# Validate the key against the registry HERE, before torchrun. argparse catches
# a bad key too, but only once every rank has booted and imported torch.
python3 - "${TRAINER_KEY}" <<'PYEOF' || exit 1
import sys
sys.path.insert(0, ".")
from main import TRAINER_REGISTRY
key = sys.argv[1]
if key not in TRAINER_REGISTRY:
    print(f"ERROR: TRAINER_KEY={key!r} is not in main.py's TRAINER_REGISTRY.",
          file=sys.stderr)
    print(f"       Valid keys: {sorted(TRAINER_REGISTRY)}", file=sys.stderr)
    sys.exit(1)
print(f"  Preflight OK: trainer {key} -> {TRAINER_REGISTRY[key].__module__}")
PYEOF

echo "=============================================="
echo "FewStepRL Stage-3 -- decoupled loss"
echo "  Nodes:       ${NNODES} (this node: ${NODE_RANK})"
echo "  GPUs/node:   ${NUM_GPUS}    Total: ${TOTAL_GPUS}"
echo "  Trainer:     ${TRAINER_KEY}"
echo "  Config:      ${CONFIG}"
echo "  SD3.5:       ${SD35_PATH}"
echo "  Stage 1:     ${PRETRAIN_LORA}"
echo "  Stage 2:     ${ONPOLICY_LORA}"
echo "  Time:        $(date)"
echo "=============================================="

"${TORCHRUN_BIN:-torchrun}" \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --nproc_per_node="${NUM_GPUS}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    main.py \
    "${CONFIG}" \
    --trainer "${TRAINER_KEY}" \
    --override \
    model.pretrained_path="${SD35_PATH}" \
    "model.generator_lora.pre_merge_paths=[${PRETRAIN_LORA}]" \
    model.generator_lora.load_path="${ONPOLICY_LORA}" \
    "${EXTRA_ARGS[@]}"
