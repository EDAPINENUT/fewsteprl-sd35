# Evaluation: sampling + reward scoring

Generate images from a released (or your own) Stage-3 checkpoint and score
them with the reward models used during training. Run every command from the
repository root.

```
scripts/eval/
  configs/sample_sd35m_fewstep.yaml   sampling protocol (4-step, 512 px, seed 12345)
  sample.sh      <target> <prompt_set> [gpus]   -> outputs/eval/<tag>/<set>/*.png + metadata.json
  score.py       --sample_dir ... --metrics ...  -> score_<metric>.json + summary.json
  eval_ckpt.sh   <target> [gpus]                 sample + score under the trial's protocol
  eval_all.sh    [gpus] [targets...]             base + every checkpoints/<trial>, then collect
  collect.py     [--out_root outputs/eval]       -> outputs/eval/summary.{csv,md}
```

## 1. Environment

- Python 3.10, CUDA 12.4, PyTorch 2.6.0. From the repo root:

  ```bash
  pip install -r requirements.txt
  ```

  This covers sampling and the aesthetic / clipscore / hpsv2 / pickscore / ocr /
  imagereward / geneval2 scorers. OCR uses `paddleocr==3.7.0` +
  `paddlepaddle==3.3.1` (CPU wheel; det+rec run on CPU by design).

- **hpsv3 needs a separate environment**: `hpsv3` pins `transformers==4.45`,
  while training and the other scorers need `transformers==4.57`. Create a
  second env with the same torch, then install `hpsv3` (e.g.
  `pip install hpsv3` or from <https://github.com/MizzenAI/HPSv3>) together with
  this repo's other requirements except `transformers`. Pass that interpreter
  as `HPSV3_PYTHON` (for `eval_ckpt.sh`) or `--python` (for `score.py`).

## 2. Models and scorers

`bash download_scripts/download_assets.sh` fetches everything below into the
layout the code expects (`WITH_IMAGEREWARD=1 WITH_HPSV3=1 WITH_GENEVAL2=1` for
the optional metrics). Download sources:

| what | source | local path |
|---|---|---|
| SD3.5-Medium (gated) | [stabilityai/stable-diffusion-3.5-medium](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium) | `models/stable-diffusion-3.5-medium/` |
| AnyFlow Stage-1 / Stage-2 LoRAs | [Harahan/MeanFlowNFT](https://huggingface.co/Harahan/MeanFlowNFT) (`anyflow_pretrain/`, `anyflow_onpolicy/`) | `models/pretrained/meanflownft_loras/` |
| Stage-3 checkpoints | [Delcher/fewsteprl-sd35](https://huggingface.co/Delcher/fewsteprl-sd35) | `checkpoints/<trial>/checkpoint-<step>/generator_ema.pt` |
| **pickscore** | [yuvalkirstain/PickScore_v1](https://huggingface.co/yuvalkirstain/PickScore_v1) + processor of [laion/CLIP-ViT-H-14-laion2B-s32B-b79K](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K) | `models/hf_cache/` |
| **hpsv2** (v2.1) | [xswu/HPSv2](https://huggingface.co/xswu/HPSv2) `HPS_v2.1_compressed.pt` + `open_clip_pytorch_model.bin` from [laion/CLIP-ViT-H-14-laion2B-s32B-b79K](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K) | `models/reward_ckpts/` |
| **clipscore** | [openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) | `models/hf_cache/` |
| **aesthetic** | `sac+logos+ava1-l14-linearMSE.pth` from [christophschuhmann/improved-aesthetic-predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor) (on CLIP ViT-L/14) | `models/reward_ckpts/` |
| **ocr** | PaddleOCR 3.x PP-OCRv5: [PaddlePaddle/en_PP-OCRv5_mobile_rec](https://huggingface.co/PaddlePaddle/en_PP-OCRv5_mobile_rec), [PaddlePaddle/PP-OCRv5_mobile_det](https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_det) | `models/paddleocr_v5/` |
| imagereward (optional) | [THUDM/ImageReward](https://huggingface.co/THUDM/ImageReward) `ImageReward.pt`, `med_config.json` | `models/reward_ckpts/ImageReward/` |
| hpsv3 (optional) | [MizzenAI/HPSv3](https://huggingface.co/MizzenAI/HPSv3) `HPSv3.safetensors` + `HPSv3_7B.yaml` ([GitHub](https://github.com/MizzenAI/HPSv3)), base [Qwen/Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) | `models/reward_ckpts/hpsv3/`, `models/qwen2vl/` |
| geneval2 (optional) | [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) as the Soft-TIFA judge | `models/reward_ckpts/geneval2/Qwen3-VL-8B-Instruct/` |

Environment variables (all optional; defaults are the paths above):

| variable | meaning |
|---|---|
| `FEWSTEPRL_SD35_PATH` | SD3.5-Medium pipeline dir |
| `FEWSTEPRL_REWARD_CKPT_PATH` | reward checkpoint root (`models/reward_ckpts`) |
| `FEWSTEPRL_PADDLEOCR_V5_DIR` | dir holding the PP-OCRv5 model dirs |
| `FEWSTEPRL_PADDLEOCR_HOME` | writable dir for the OCR init lock (default `<reward root>/.paddleocr`) |
| `FEWSTEPRL_HPSV3_CONFIG_PATH`, `FEWSTEPRL_HPSV3_CHECKPOINT_PATH` | explicit HPSv3 yaml / weights |
| `FEWSTEPRL_GENEVAL2_MODEL_NAME`, `FEWSTEPRL_GENEVAL2_DATA_PATH` | GenEval2 judge / VQA jsonl (default `dataset/geneval2/test.jsonl`) |
| `HF_HOME` | HF cache (default `models/hf_cache`) |
| `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE` | set to 1 by the scripts: everything loads from local files |

## 3. Protocol

Sampling matches the eval block of every training config: Stage 1 + Stage 2
merged into SD3.5-M, the Stage-3 LoRA on top, **4-step flow-map ODE, guidance
1.0, 512 px, seed 12345 (+ rank), `text_max_sequence_length` 128**, the full
test split of each prompt set. `base` = Stage 1 + Stage 2 only, i.e. the step-0
model every trial starts from.

Which metric is scored on which prompts follows each config's
`eval.reward_dataset_map`:

| protocol | prompts -> metrics | `mean` |
|---|---|---|
| mix (`*_mix`) | `drawbench` (1000) -> aesthetic, clipscore, hpsv2, pickscore; `ocr` (1018) -> ocr | sum of the five |
| ocr (`*_ocr`) | `ocr` (1018) -> aesthetic, clipscore, ocr | sum of the three |
| extras | imagereward -> `drawbench`; hpsv3 -> `hpsv3` (1024); geneval2 -> `geneval2` (800) | not in `mean` |

On the mix side, ocr is a held-out probe (not trained on). The OCR scorer reads
the target string from the double-quoted span of the prompt, so it is only
defined on the `ocr` prompt set; `score.py` refuses to run it elsewhere.

## 4. Usage

```bash
# one released checkpoint (protocol inferred from the name), 8 GPUs for sampling
bash scripts/eval/eval_ckpt.sh meanflownft_mix 8
bash scripts/eval/eval_ckpt.sh dmdawm_ocr 8

# the step-0 baseline, both protocols
bash scripts/eval/eval_ckpt.sh base 8

# everything + summary table (outputs/eval/summary.{csv,md})
bash scripts/eval/eval_all.sh 8

# extra metrics
EXTRA_METRICS="imagereward geneval2" bash scripts/eval/eval_ckpt.sh reflmx_mix 8
EXTRA_METRICS=hpsv3 HPSV3_PYTHON=/path/to/hpsv3_env/bin/python \
    bash scripts/eval/eval_ckpt.sh reflmx_mix 8

# your own checkpoint from a training run
PROTOCOL=mix bash scripts/eval/eval_ckpt.sh outputs/meanflownft_mix/<timestamp>/checkpoint-100/generator_ema.pt 8

# the two steps by hand
bash scripts/eval/sample.sh meanflownft_mix drawbench 8
python3 scripts/eval/score.py --sample_dir outputs/eval/meanflownft_mix/drawbench \
    --metrics aesthetic,clipscore,hpsv2,pickscore
```

Quick smoke test: `MAX_PROMPTS=32 bash scripts/eval/eval_ckpt.sh meanflownft_mix 1`
(use a separate `OUT_ROOT` so the truncated sets are not mistaken for full ones).

Finished sample sets (complete `metadata.json`) and scored metrics
(`score_<metric>.json` with one score per image) are skipped on re-runs; set
`FORCE=1` (sampling) or pass `--force` (scoring) to redo them.

## 5. Reading the numbers

- **Offline re-sampling is not the online eval.** The `checkpoints/README.md`
  tables are what the runs logged during training (or, where marked offline,
  re-sampled after training). This pipeline draws its own samples from
  `generator_ema.pt` with a fixed seed; expect small differences from the
  archived rows, and compare checkpoints only against each other under the same
  `NUM_GPUS` (the per-rank seed is `12345 + rank`).
- **OCR scores depend on the PaddleOCR generation.** All numbers in this repo
  are PaddleOCR 3.x + PP-OCRv5. PaddleOCR 2.x + PP-OCRv3/v4 scores the same
  images ~17% higher (0.5355 vs 0.4420 on 64 baseline images); never compare
  across the two.
