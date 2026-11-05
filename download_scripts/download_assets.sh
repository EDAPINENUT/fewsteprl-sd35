#!/bin/bash
# Download every external asset needed to train (scripts/train) and evaluate
# (scripts/eval). Needs internet; run once, then everything runs offline.
#
# Always fetched (training + the default eval protocol):
#   models/stable-diffusion-3.5-medium/            SD3.5-Medium (GATED: accept the license first)
#   models/pretrained/meanflownft_loras/           AnyFlow Stage-1/2 LoRAs (Harahan/MeanFlowNFT)
#   models/reward_ckpts/open_clip_pytorch_model.bin  hpsv2 backbone (CLIP-ViT-H-14)
#   models/reward_ckpts/HPS_v2.1_compressed.pt       hpsv2 head
#   models/reward_ckpts/sac+logos+ava1-l14-linearMSE.pth  aesthetic head
#   models/hf_cache/   PickScore_v1, CLIP-ViT-H-14 processor, openai/clip-vit-large-patch14
#   models/paddleocr_v5/{en_PP-OCRv5_mobile_rec,PP-OCRv5_mobile_det}   ocr reward
#
# Optional (extra eval metrics, large):
#   WITH_IMAGEREWARD=1  models/reward_ckpts/ImageReward/   (THUDM/ImageReward, ~1.7 GB)
#   WITH_HPSV3=1        models/reward_ckpts/hpsv3/ + models/qwen2vl/Qwen2-VL-7B-Instruct (~32 GB)
#   WITH_GENEVAL2=1     models/reward_ckpts/geneval2/Qwen3-VL-8B-Instruct (~16 GB)
#
# Usage:
#   bash download_scripts/download_assets.sh
#   SKIP_SD35=1 bash download_scripts/download_assets.sh          # base model already present
#   WITH_IMAGEREWARD=1 WITH_HPSV3=1 WITH_GENEVAL2=1 bash download_scripts/download_assets.sh
#
# Override any location with FEWSTEPRL_SD35_PATH / FEWSTEPRL_REWARD_CKPT_PATH /
# FEWSTEPRL_PADDLEOCR_V5_DIR / HF_HOME (the train and eval scripts read the same
# variables).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

SD35_DIR="${FEWSTEPRL_SD35_PATH:-${REPO_ROOT}/models/stable-diffusion-3.5-medium}"
LORA_DIR="${REPO_ROOT}/models/pretrained/meanflownft_loras"
REWARD_DIR="${FEWSTEPRL_REWARD_CKPT_PATH:-${REPO_ROOT}/models/reward_ckpts}"
OCR_DIR="${FEWSTEPRL_PADDLEOCR_V5_DIR:-${REPO_ROOT}/models/paddleocr_v5}"
QWEN2VL_DIR="${REPO_ROOT}/models/qwen2vl/Qwen2-VL-7B-Instruct"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/models/hf_cache}"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE

mkdir -p "${SD35_DIR}" "${LORA_DIR}" "${REWARD_DIR}" "${OCR_DIR}" "${HF_HOME}"

if command -v hf >/dev/null 2>&1; then
    HF=(hf)
elif command -v huggingface-cli >/dev/null 2>&1; then
    HF=(huggingface-cli)
else
    echo "ERROR: install the HF CLI first:  pip install -U 'huggingface_hub[cli]'" >&2
    exit 1
fi

echo "=============================================="
echo "FewStepRL-SD35 asset download"
echo "  SD3.5:        ${SD35_DIR}"
echo "  Stage LoRAs:  ${LORA_DIR}"
echo "  Reward ckpts: ${REWARD_DIR}"
echo "  PaddleOCR:    ${OCR_DIR}"
echo "  HF_HOME:      ${HF_HOME}"
echo "=============================================="

# --- 1. SD3.5-Medium (gated) ---------------------------------------------------
if [[ -n "${SKIP_SD35:-}" ]]; then
    echo "[1/6] SD3.5-Medium: SKIPPED (SKIP_SD35 set)"
elif ! "${HF[@]}" auth whoami >/dev/null 2>&1; then
    echo "[1/6] SD3.5-Medium is gated: accept the license at"
    echo "      https://huggingface.co/stabilityai/stable-diffusion-3.5-medium"
    echo "      then run '${HF[*]} auth login' and re-run this script (or set SKIP_SD35=1)." >&2
    exit 1
else
    echo "[1/6] stabilityai/stable-diffusion-3.5-medium"
    "${HF[@]}" download stabilityai/stable-diffusion-3.5-medium --local-dir "${SD35_DIR}"
fi

# --- 2. AnyFlow Stage-1 / Stage-2 LoRAs -----------------------------------------
# Every Stage-3 trial starts from SD3.5-M + Stage 1 (merged) + Stage 2 (merged).
echo "[2/6] Harahan/MeanFlowNFT: anyflow_pretrain + anyflow_onpolicy"
"${HF[@]}" download Harahan/MeanFlowNFT \
    anyflow_pretrain/generator_ema.pt \
    anyflow_onpolicy/generator_ema.pt \
    --local-dir "${LORA_DIR}"

# --- 3. Reward files whose names and location are fixed in the scorers ----------
echo "[3/6] reward checkpoints -> ${REWARD_DIR}"
"${HF[@]}" download laion/CLIP-ViT-H-14-laion2B-s32B-b79K open_clip_pytorch_model.bin \
    --local-dir "${REWARD_DIR}"
"${HF[@]}" download xswu/HPSv2 HPS_v2.1_compressed.pt --local-dir "${REWARD_DIR}"
curl -fL --retry 3 --retry-delay 2 \
    "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/refs/heads/main/sac+logos+ava1-l14-linearMSE.pth" \
    -o "${REWARD_DIR}/sac+logos+ava1-l14-linearMSE.pth"

# --- 4. Hub-loaded reward models, cached into HF_HOME -----------------------------
# pickscore_scorer.py: AutoModel("yuvalkirstain/PickScore_v1") +
#   AutoProcessor("laion/CLIP-ViT-H-14-laion2B-s32B-b79K") (processor files only).
# clip_scorer.py / aesthetic_scorer.py: openai/clip-vit-large-patch14.
echo "[4/6] HF hub reward models -> ${HF_HOME}"
"${HF[@]}" download yuvalkirstain/PickScore_v1
"${HF[@]}" download laion/CLIP-ViT-H-14-laion2B-s32B-b79K \
    --exclude "*.bin" "*.safetensors" "*.pt" "*.msgpack" "*.h5"
"${HF[@]}" download openai/clip-vit-large-patch14

# --- 5. PaddleOCR 3.x PP-OCRv5 (ocr reward) ---------------------------------------
# ocr_scorer.py loads <OCR_DIR>/en_PP-OCRv5_mobile_rec and the first present of
# PP-OCRv5_mobile_det / PP-OCRv5_server_det, each an extracted PaddleX
# inference dir (inference.json / inference.pdiparams / inference.yml).
echo "[5/6] PaddleOCR PP-OCRv5 -> ${OCR_DIR}"
for m in en_PP-OCRv5_mobile_rec PP-OCRv5_mobile_det; do
    "${HF[@]}" download "PaddlePaddle/${m}" --local-dir "${OCR_DIR}/${m}"
done

# --- 6. Optional extra eval metrics ----------------------------------------------
echo "[6/6] optional eval metrics"
if [[ "${WITH_IMAGEREWARD:-0}" == "1" ]]; then
    echo "  -> THUDM/ImageReward"
    "${HF[@]}" download THUDM/ImageReward ImageReward.pt med_config.json \
        --local-dir "${REWARD_DIR}/ImageReward"
fi
if [[ "${WITH_HPSV3:-0}" == "1" ]]; then
    # HPSv3 = reward head (MizzenAI/HPSv3) on top of Qwen2-VL-7B-Instruct; its
    # inferencer yaml names the base model, so point it at the local copy.
    echo "  -> MizzenAI/HPSv3 + Qwen/Qwen2-VL-7B-Instruct"
    mkdir -p "${REWARD_DIR}/hpsv3"
    "${HF[@]}" download MizzenAI/HPSv3 HPSv3.safetensors --local-dir "${REWARD_DIR}/hpsv3"
    "${HF[@]}" download Qwen/Qwen2-VL-7B-Instruct --local-dir "${QWEN2VL_DIR}" \
        --exclude "*.bin" "*.pth" "*.msgpack" "*.h5"
    curl -fsSL -o "${REWARD_DIR}/hpsv3/HPSv3_7B.yaml" \
        https://raw.githubusercontent.com/MizzenAI/HPSv3/main/hpsv3/config/HPSv3_7B.yaml
    sed -i "s#^model_name_or_path:.*#model_name_or_path: \"${QWEN2VL_DIR}\"#" \
        "${REWARD_DIR}/hpsv3/HPSv3_7B.yaml"
fi
if [[ "${WITH_GENEVAL2:-0}" == "1" ]]; then
    echo "  -> Qwen/Qwen3-VL-8B-Instruct"
    "${HF[@]}" download Qwen/Qwen3-VL-8B-Instruct \
        --local-dir "${REWARD_DIR}/geneval2/Qwen3-VL-8B-Instruct"
fi

# --- Verify --------------------------------------------------------------------
echo "---------- verification ----------"
status=0
check() {  # check <path> <label>
    if [[ -e "$1" ]]; then printf '  OK    %s\n' "$2"; else printf '  MISS  %s (%s)\n' "$2" "$1"; status=1; fi
}
[[ -n "${SKIP_SD35:-}" ]] || check "${SD35_DIR}/model_index.json" "SD3.5-Medium"
check "${LORA_DIR}/anyflow_pretrain/generator_ema.pt" "Stage-1 LoRA"
check "${LORA_DIR}/anyflow_onpolicy/generator_ema.pt" "Stage-2 LoRA"
check "${REWARD_DIR}/open_clip_pytorch_model.bin" "hpsv2 backbone"
check "${REWARD_DIR}/HPS_v2.1_compressed.pt" "hpsv2 head"
check "${REWARD_DIR}/sac+logos+ava1-l14-linearMSE.pth" "aesthetic head"
check "${HF_HOME}/hub/models--yuvalkirstain--PickScore_v1" "PickScore"
check "${HF_HOME}/hub/models--openai--clip-vit-large-patch14" "CLIP ViT-L/14"
check "${HF_HOME}/hub/models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K" "CLIP-H processor"
check "${OCR_DIR}/en_PP-OCRv5_mobile_rec/inference.pdiparams" "PP-OCRv5 rec"
check "${OCR_DIR}/PP-OCRv5_mobile_det/inference.pdiparams" "PP-OCRv5 det"
[[ "${WITH_IMAGEREWARD:-0}" != "1" ]] || check "${REWARD_DIR}/ImageReward/ImageReward.pt" "ImageReward"
[[ "${WITH_HPSV3:-0}" != "1" ]] || check "${REWARD_DIR}/hpsv3/HPSv3.safetensors" "HPSv3"
[[ "${WITH_GENEVAL2:-0}" != "1" ]] || check "${REWARD_DIR}/geneval2/Qwen3-VL-8B-Instruct/config.json" "GenEval2 judge"

if [[ "${status}" -eq 0 ]]; then
    echo "All requested assets are present. Train:  bash scripts/train/meanflownft_mix.sh 8"
    echo "                                  Eval:   bash scripts/eval/eval_ckpt.sh meanflownft_mix 8"
fi
exit "${status}"
