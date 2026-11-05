#!/usr/bin/env bash
# Generate eval samples for one checkpoint on one prompt set.
#
# Usage:
#   bash scripts/eval/sample.sh <TARGET> <PROMPT_SET> [NUM_GPUS] [EXTRA_OVERRIDES...]
#
#   TARGET      an experiment name under checkpoints/ (e.g. meanflownft_mix),
#               `base` for the step-0 model (SD3.5-M + Stage 1 + Stage 2 only),
#               or a path to any Stage-3 generator_ema.pt
#   PROMPT_SET  drawbench | ocr | pickscore | hpsv3 | geneval2
#               (reads dataset/<PROMPT_SET>/test.txt)
#   NUM_GPUS    default 1; >1 shards prompts over GPUs with torchrun
#
# Output: outputs/eval/<TAG>/<PROMPT_SET>/{*.png, metadata.json}, where TAG is
# the experiment name, `base`, or (for a path) $TAG / the checkpoint's parent
# directory names.
#
# Environment:
#   MAX_PROMPTS          0 (default) = the whole test split
#   TAG                  override the output tag
#   FORCE=1              regenerate even if metadata.json is complete
#   FEWSTEPRL_SD35_PATH  SD3.5-Medium pipeline (default models/stable-diffusion-3.5-medium)
#   OUT_ROOT             default outputs/eval
#
# Sampling uses seed 12345 + rank, so results depend on NUM_GPUS; keep it
# fixed across the checkpoints you compare.
#
# Examples:
#   bash scripts/eval/sample.sh meanflownft_mix drawbench 8
#   bash scripts/eval/sample.sh base ocr 8
#   MAX_PROMPTS=64 bash scripts/eval/sample.sh outputs/my_run/checkpoint-100/generator_ema.pt drawbench

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO}"

TARGET="${1:?usage: sample.sh <TARGET> <PROMPT_SET> [NUM_GPUS] [EXTRA_OVERRIDES...]}"
PROMPT_SET="${2:?usage: sample.sh <TARGET> <PROMPT_SET> [NUM_GPUS] [EXTRA_OVERRIDES...]}"
NGPU="${3:-1}"
EXTRA=("${@:4}")

CONFIG="scripts/eval/configs/sample_sd35m_fewstep.yaml"
OUT_ROOT="${OUT_ROOT:-outputs/eval}"
MAX_PROMPTS="${MAX_PROMPTS:-0}"
SD35_PATH="${FEWSTEPRL_SD35_PATH:-models/stable-diffusion-3.5-medium}"

fail() { echo "ERROR: $*" >&2; exit 1; }

PROMPT_FILE="dataset/${PROMPT_SET}/test.txt"
[[ -f "${PROMPT_FILE}" ]] || fail "unknown prompt set '${PROMPT_SET}': ${PROMPT_FILE} not found"

# --- Resolve TARGET -> (num_stages, stage3 path, tag) -------------------------
STAGE_ARGS=()
if [[ "${TARGET}" == "base" ]]; then
    TAG="${TAG:-base}"
    STAGE_ARGS=("num_stages=2")
elif [[ -f "${TARGET}" ]]; then
    CKPT="$(cd "$(dirname "${TARGET}")" && pwd)/$(basename "${TARGET}")"
    TAG="${TAG:-$(basename "$(dirname "$(dirname "${CKPT}")")")_$(basename "$(dirname "${CKPT}")")}"
    STAGE_ARGS=("num_stages=3" "stage3_lora_path=${CKPT}")
elif [[ -d "checkpoints/${TARGET}" ]]; then
    mapfile -t found < <(ls checkpoints/"${TARGET}"/checkpoint-*/generator_ema.pt 2>/dev/null)
    [[ ${#found[@]} -eq 1 ]] || fail "expected exactly one checkpoints/${TARGET}/checkpoint-*/generator_ema.pt, found ${#found[@]}"
    CKPT="${REPO}/${found[0]}"
    TAG="${TAG:-${TARGET}}"
    STAGE_ARGS=("num_stages=3" "stage3_lora_path=${CKPT}")
else
    fail "TARGET '${TARGET}' is not 'base', a file, or a directory under checkpoints/"
fi

OUT_DIR="${OUT_ROOT}/${TAG}/${PROMPT_SET}"

# --- Skip if already complete -------------------------------------------------
N_EXPECTED="$(grep -cve '^\s*$' "${PROMPT_FILE}")"
if [[ "${MAX_PROMPTS}" -gt 0 && "${MAX_PROMPTS}" -lt "${N_EXPECTED}" ]]; then
    N_EXPECTED="${MAX_PROMPTS}"
fi
if [[ -z "${FORCE:-}" && -f "${OUT_DIR}/metadata.json" ]]; then
    N_DONE="$(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["images"]))' "${OUT_DIR}/metadata.json")"
    if [[ "${N_DONE}" -ge "${N_EXPECTED}" ]]; then
        echo "[sample] skip ${TAG}/${PROMPT_SET}: ${N_DONE} images already in ${OUT_DIR}"
        exit 0
    fi
fi

# --- Preflight ----------------------------------------------------------------
[[ -f "${SD35_PATH}/model_index.json" ]] || fail "SD3.5 pipeline not found at ${SD35_PATH} (bash download_scripts/download_assets.sh)"
for s in anyflow_pretrain anyflow_onpolicy; do
    [[ -f "models/pretrained/meanflownft_loras/${s}/generator_ema.pt" ]] \
        || fail "missing Stage LoRA models/pretrained/meanflownft_loras/${s}/generator_ema.pt (bash download_scripts/download_assets.sh)"
done

export HF_HOME="${HF_HOME:-${REPO}/models/hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false

mkdir -p "${OUT_DIR}"
OVERRIDES=(
    "pretrained_path=${SD35_PATH}"
    "${STAGE_ARGS[@]}"
    "prompt_file=${PROMPT_FILE}"
    "max_prompts=${MAX_PROMPTS}"
    "output_dir=${OUT_DIR}"
    "${EXTRA[@]}"
)

echo "[sample] ${TAG} / ${PROMPT_SET}: ${N_EXPECTED} prompts on ${NGPU} GPU(s) -> ${OUT_DIR}"
if [[ "${NGPU}" -gt 1 ]]; then
    "${TORCHRUN_BIN:-torchrun}" --standalone --nproc_per_node="${NGPU}" \
        inference.py "${CONFIG}" --override "${OVERRIDES[@]}" distributed=true
else
    python3 inference.py "${CONFIG}" --override "${OVERRIDES[@]}"
fi
