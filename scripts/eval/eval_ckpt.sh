#!/usr/bin/env bash
# Sample + score one checkpoint under the protocol its training run used.
#
# Usage:
#   bash scripts/eval/eval_ckpt.sh <TARGET> [NUM_GPUS]
#
#   TARGET   experiment name under checkpoints/, `base`, or a generator_ema.pt
#            path (then set PROTOCOL=mix|ocr)
#
# Protocol = the training config's eval.reward_dataset_map:
#   mix  (*_mix):  drawbench -> aesthetic clipscore hpsv2 pickscore
#                  ocr       -> ocr                    (held-out probe)
#   ocr  (*_ocr):  ocr       -> aesthetic clipscore ocr
#   base:          both protocols (the step-0 row of each table)
#
# Environment:
#   PROTOCOL       mix | ocr | both   (default: from the TARGET suffix)
#   EXTRA_METRICS  space-separated, any of: imagereward hpsv3 geneval2
#                    imagereward -> drawbench prompts
#                    hpsv3       -> hpsv3 prompts, scored with $HPSV3_PYTHON
#                    geneval2    -> geneval2 prompts (Qwen3-VL-8B judge)
#   HPSV3_PYTHON   interpreter of the separate hpsv3 env (required for hpsv3)
#   MAX_PROMPTS, TAG, FORCE, OUT_ROOT   passed through to sample.sh
#
# Scoring runs on the first visible GPU after sampling finishes.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO}"

TARGET="${1:?usage: eval_ckpt.sh <TARGET> [NUM_GPUS]}"
NGPU="${2:-1}"
OUT_ROOT="${OUT_ROOT:-outputs/eval}"
export OUT_ROOT

if [[ -z "${PROTOCOL:-}" ]]; then
    case "${TARGET}" in
        base)  PROTOCOL=both ;;
        *_mix) PROTOCOL=mix ;;
        *_ocr) PROTOCOL=ocr ;;
        *) echo "ERROR: cannot infer PROTOCOL from '${TARGET}'; set PROTOCOL=mix|ocr" >&2; exit 1 ;;
    esac
fi

# Resolve the tag sample.sh will use, so score.py reads the same directory.
if [[ -n "${TAG:-}" ]]; then
    :
elif [[ "${TARGET}" == "base" || -d "checkpoints/${TARGET}" ]]; then
    TAG="${TARGET}"
else
    CKPT="$(cd "$(dirname "${TARGET}")" && pwd)/$(basename "${TARGET}")"
    TAG="$(basename "$(dirname "$(dirname "${CKPT}")")")_$(basename "$(dirname "${CKPT}")")"
fi
export TAG

run() {  # run <prompt_set> <metrics-csv> [python]
    local set="$1" metrics="$2" py="${3:-python3}"
    bash scripts/eval/sample.sh "${TARGET}" "${set}" "${NGPU}"
    "${py}" scripts/eval/score.py --sample_dir "${OUT_ROOT}/${TAG}/${set}" --metrics "${metrics}"
}

if [[ "${PROTOCOL}" == "mix" || "${PROTOCOL}" == "both" ]]; then
    run drawbench aesthetic,clipscore,hpsv2,pickscore
    run ocr ocr
fi
if [[ "${PROTOCOL}" == "ocr" || "${PROTOCOL}" == "both" ]]; then
    run ocr aesthetic,clipscore,ocr
fi

for m in ${EXTRA_METRICS:-}; do
    case "${m}" in
        imagereward) run drawbench imagereward ;;
        geneval2)    run geneval2 geneval2 ;;
        hpsv3)
            [[ -n "${HPSV3_PYTHON:-}" ]] || { echo "ERROR: hpsv3 needs HPSV3_PYTHON (see scripts/eval/README.md)" >&2; exit 1; }
            run hpsv3 hpsv3 "${HPSV3_PYTHON}" ;;
        *) echo "ERROR: unknown EXTRA_METRICS entry '${m}'" >&2; exit 1 ;;
    esac
done

echo "[eval] ${TAG} done; per-set summaries under ${OUT_ROOT}/${TAG}/*/summary.json"
