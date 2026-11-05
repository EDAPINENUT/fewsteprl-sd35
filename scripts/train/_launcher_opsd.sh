#!/bin/bash
# Shared launcher for the OPSD family (meanflowopsd_mix, reflmx_mix). Not
# meant to be called directly: the scripts/train/<trial>.sh wrappers export
# CONFIG / TRAINER_KEY / the Stage-1+2 LoRAs and exec this file.
#
# Prerequisites (see README.md, "Setup"):
#   bash download_scripts/download_assets.sh
#
# Usage (via a wrapper):
#   bash scripts/train/<trial>.sh [NUM_GPUS] [EXTRA_OVERRIDES...]
#
# Smoke (1 GPU, tiny, no eval):
#   bash scripts/train/meanflowopsd_mix.sh 1 \
#       fewstep_rl.num_image_per_prompt=4 fewstep_rl.num_batches_per_epoch=1 \
#       train.batch_size=4 fewstep_rl.num_epochs=2 eval.enabled=false \
#       fewstep_rl.log_sample_images=0
#
# loss_policy / loss_unscaled are NOT comparable across trials (different
# objectives). Compare rl_reward/* and the eval metrics. Within an OPSD trial,
# watch:
#   rl_diagnostic/opsd_delta_res  -- mean ||xp - x_anchor|| / ||x_anchor||; pinned
#                                    at rho means the trust region saturates every step
#   rl_loss/opsd_pull_pos vs opsd_push_neg -- the two reward forces
#   rl_loss/reward_augmentation_loss vs self_regression_loss

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

# Same two-stage init as every other Stage-3 trial, so comparisons start from an
# identical policy.
LORA_DIR="${REPO_ROOT}/models/pretrained/meanflownft_loras"
export PRETRAIN_LORA="${PRETRAIN_LORA:-${LORA_DIR}/anyflow_pretrain/generator_ema.pt}"
export ONPOLICY_LORA="${ONPOLICY_LORA:-${LORA_DIR}/anyflow_onpolicy/generator_ema.pt}"

fail() { echo "ERROR: $*" >&2; exit 1; }

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
    || fail "PaddleOCR PP-OCRv5 models not found under ${FEWSTEPRL_PADDLEOCR_V5_DIR} (needed by the ocr metric).
  Run: bash download_scripts/download_assets.sh"

# --- Preflight: config sanity + registry ------------------------------------
python3 - "${CONFIG}" "${NUM_GPUS}" "${NNODES:-1}" "${TRAINER_KEY}" <<'PYEOF' || exit 1
import sys, yaml
sys.path.insert(0, ".")
cfg_path, num_gpus, nnodes, key = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
cfg = yaml.safe_load(open(cfg_path))

if "fewstep_rl" not in cfg:
    print(f"ERROR: {cfg_path} has no 'fewstep_rl' section.", file=sys.stderr)
    sys.exit(1)
fs = cfg["fewstep_rl"]

if not fs.get("opsd_enabled", False):
    print(f"ERROR: {cfg_path} has fewstep_rl.opsd_enabled=false. This trainer "
          "builds reward-gradient targets and has nothing to fit without it.",
          file=sys.stderr)
    sys.exit(1)

# The reward that supplies the target DIRECTION must be differentiable.
DIFFABLE = {"hpsv2", "clipscore", "pickscore", "aesthetic", "mixed"}
kind = fs.get("opsd_reward_kind", "hpsv2")
if fs.get("opsd_dir_mode", "grad") == "grad" and kind not in DIFFABLE:
    print(f"ERROR: opsd_reward_kind={kind!r} is not differentiable, so "
          "opsd_dir_mode='grad' cannot build a target direction. In the OCR "
          "(non-differentiable) setting OPSD degenerates to MeanFlowNFT -- run "
          "scripts/train/meanflownft_ocr.sh instead.", file=sys.stderr)
    sys.exit(1)

# The config validator catches this too, but only after N ranks have booted.
if fs.get("opsd_state_mode", "forward") == "rollout" and not fs.get("reuse_x_t", False):
    print("ERROR: opsd_state_mode='rollout' requires fewstep_rl.reuse_x_t=true.",
          file=sys.stderr)
    sys.exit(1)

# K-repeat constraint: batch_size * total_gpus % K == 0.
bs, K = int(cfg["train"]["batch_size"]), int(fs["num_image_per_prompt"])
total = bs * num_gpus * nnodes
if total % K:
    print(f"ERROR: train.batch_size * total_gpus = {total} is not divisible by "
          f"num_image_per_prompt = {K}.", file=sys.stderr)
    for cand in (24, 12, 8, 6, 4, 3, 2, 1):
        if total % cand == 0:
            print(f"       try fewstep_rl.num_image_per_prompt={cand}"
                  f"   ({total // cand} prompts/step)", file=sys.stderr)
            break
    sys.exit(1)

from main import TRAINER_REGISTRY
if key not in TRAINER_REGISTRY:
    print(f"ERROR: TRAINER_KEY={key!r} is not in main.py's TRAINER_REGISTRY.\n"
          f"       Valid keys: {sorted(TRAINER_REGISTRY)}", file=sys.stderr)
    sys.exit(1)

print(f"  Preflight OK: trainer {key} -> {TRAINER_REGISTRY[key].__module__}")
print(f"  Preflight OK: {total} samples/step / K={K} = {total // K} prompts/step")
print(f"  Preflight OK: OPSD rho={fs.get('opsd_rho')} "
      f"sigma*={fs.get('opsd_query_sigma')} M_tgt={fs.get('opsd_n_ascent')} "
      f"dir={fs.get('opsd_dir_mode')} state={fs.get('opsd_state_mode')} "
      f"grad_reward={kind}")
print("  Train reward: " + " + ".join(
    f"{k} {v}" for k, v in fs.get("reward_fn", {}).items()))
PYEOF

# --- Distributed / NCCL env ---------------------------------------------------
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29507}"

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
# See _launcher_decoupled.sh: an unset WANDB_MODE without network/api_key turns
# into a multi-minute NCCL hang instead of an error.
export WANDB_MODE="${WANDB_MODE:-offline}"

TOTAL_GPUS=$((NUM_GPUS * NNODES))

echo "=============================================="
echo "FewStepRL Stage-3 -- DiffusionOPSD family"
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
