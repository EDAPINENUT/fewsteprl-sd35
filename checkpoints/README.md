# Released checkpoints

One Stage-3 LoRA per experiment, plus the two auxiliary v-predictors that the
`dmd*` trials warm-start from. Every experiment name matches its launcher
`scripts/train/<trial>.sh` and config `scripts/train/configs/<trial>.yaml`.

```
checkpoints/<trial>/
  config.yaml                          # the config the run saved (paths made relative)
  checkpoint-<step>/generator_ema.pt   # EMA generator: Stage-3 LoRA + delta_embedder
checkpoints/aux_vel/{mix,ocr}/
  config.yaml                          # the aux_vel distillation run's config
  aux_v_transformer/                   # diffusers SD3Transformer2DModel (~9.9 GB)
```

The weights (`generator_ema.pt` and `aux_v_transformer/`) are not in this
repository; they will be released on Hugging Face soon. Only the runs'
`config.yaml` and the aux transformers' `config.json` are tracked. The four `dmd*` launchers need them in place at
`checkpoints/aux_vel/{mix,ocr}/aux_v_transformer/diffusion_pytorch_model.safetensors`.

`generator_ema.pt` is a plain tensor state dict. It is applied **on top of**
SD3.5-Medium + the AnyFlow Stage-1 and Stage-2 LoRAs
(`models/pretrained/meanflownft_loras/{anyflow_pretrain,anyflow_onpolicy}`),
never alone -- `inference.py` with `num_stages=3` does this, and
`scripts/eval/sample.sh <trial>` wraps it.

## Index

`mean` is the sum of the eval dimensions (mix: aesthetic + clipscore + hpsv2 +
pickscore + ocr on drawbench/ocr prompts; ocr: aesthetic + clipscore + ocr on
ocr prompts), 4-step sampling, seed 12345, 512 px. Reproduce them with
`scripts/eval/`; re-sampling is not bit-identical, so expect small deviations.

### Mixed reward (train: clipscore + hpsv2 + pickscore)

| trial | trainer | step | aesthetic | clipscore | hpsv2 | pickscore | ocr (held out) | mean |
|---|---|---|---|---|---|---|---|---|
| step 0 (Stage 1+2 only) | -- | 0 | 5.363 | 0.2886 | 0.2974 | 0.8638 | 0.5071 | 7.320 |
| meanflownft_mix | `meanflow_nft` | 400 | 6.043 | 0.2814 | 0.3289 | 0.8989 | 0.4607 | 8.012 |
| nft_mix | `nft_rl` | 400 | -- | -- | -- | -- | -- | -- |
| dmdnft_mix | `dmdnft_rl` | 400 | 5.98 | 0.285 | 0.332 | 0.902 | 0.448 | 7.947 |
| dmdawm_mix | `dmdawm_rl` | 400 | 5.86 | 0.281 | 0.326 | 0.891 | 0.424 | 7.782 |
| fewstepgrpo_mix | `meanflowgrpo_rl` | 300 | 5.792 | 0.2811 | 0.3137 | 0.8733 | 0.4009 | 7.661 |
| flowgrpo_mix | `flowgrpo_rl` | 400 | 5.564 | 0.2837 | 0.2968 | 0.8713 | 0.4843 | 7.500 |
| meanflowawm_mix | `meanflowawm_rl` | 400 | 5.915 | 0.2786 | 0.3079 | 0.8902 | 0.4131 | 7.805 |
| awm_mix | `awm_rl` | 400 | 6.046 | 0.2828 | 0.3332 | 0.8945 | 0.4278 | 7.984 |
| meanflowopsd_mix | `meanflowopsd_rl` | 400 | 5.906 | 0.3217 | 0.3734 | 0.9458 | 0.4928 | 8.040 |
| reflmx_mix | `meanflowrefl_rl` | 400 | 5.891 | 0.3172 | 0.3675 | 0.9378 | 0.5459 | 8.060 |

### OCR reward (train: ocr)

| trial | trainer | step | aesthetic | clipscore | ocr | mean |
|---|---|---|---|---|---|---|
| step 0 (Stage 1+2 only) | -- | 0 | 5.236 | 0.3196 | 0.4983 | 6.054 |
| meanflownft_ocr | `meanflow_nft` | 200 | 4.394 | 0.2894 | 0.934 | 5.617 |
| dmdnft_ocr | `dmdnft_rl` | 200 | 4.406 | 0.2946 | 0.8973 | 5.598 |
| dmdawm_ocr | `dmdawm_rl` | 400 | 3.889 | 0.2703 | 0.9661 | 5.125 |
| fewstepgrpo_ocr | `meanflowgrpo_rl` | 200 | 4.877 | 0.3251 | 0.7763 | 5.978 |
| meanflowawm_ocr | `meanflowawm_rl` | 400 | 5.011 | 0.3112 | 0.9271 | 6.249 |

OCR numbers are PaddleOCR 3.x + PP-OCRv5 throughout; they are not comparable
with PaddleOCR 2.x scores (see `requirements.txt`).

## Notes

- `fewstepgrpo_mix` and `fewstepgrpo_ocr` are released at their peak step (300
  and 200); every other trial at its final step.
- `dmdnft_ocr` uses the aux network from `aux_vel/mix` (recorded in its
  `config.yaml`); `scripts/train/configs/dmdnft_ocr.yaml` points at
  `aux_vel/ocr`.
- `meanflowopsd_mix` and `reflmx_mix` run with `fewstep_rl.disable_cd=true`.
- `dmdawm_*` are the no-central-difference trials (`x_anchor_2 = x_t - t*u_old`).

`SHA256SUMS` lists every weight file in this directory.
