# Released checkpoints

One Stage-3 LoRA per experiment, plus the two auxiliary v-predictors that the
`dmd*` arms warm-start from. Every experiment name matches its launcher
`scripts/train/<arm>.sh` and config `scripts/train/configs/<arm>.yaml`.

```
checkpoints/<arm>/
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
`scripts/eval/sample.sh <arm>` wraps it.

## Index

`mean` is the sum of the eval dimensions logged during training (mix: aesthetic
+ clipscore + hpsv2 + pickscore + ocr on drawbench/ocr prompts; ocr: aesthetic +
clipscore + ocr on ocr prompts), 4-step sampling, seed 12345, 512 px.
`source` = online (eval inside training) or offline (re-sampled from the saved
checkpoint after training, because `eval_interval` does not fire on the last
step). These are archived training-time numbers; reproduce them with
`scripts/eval/` (offline re-sampling is not bit-identical to online eval).
Where a released step is an iteration the paper reports, the values follow the
paper (Tab. 6); rows at other steps (`fewstepgrpo_mix` 300, `dmdawm_ocr` and
`meanflowawm_ocr` 400) are archived logs at steps the paper does not report.

### Mixed reward (train: clipscore + hpsv2 + pickscore)

| arm | trainer | step | aesthetic | clipscore | hpsv2 | pickscore | ocr (held out) | mean | source |
|---|---|---|---|---|---|---|---|---|---|
| step 0 (Stage 1+2 only) | -- | 0 | 5.363 | 0.2886 | 0.2974 | 0.8638 | 0.5071 | 7.320 | online |
| meanflownft_mix | `meanflow_nft` | 400 | 6.043 | 0.2814 | 0.3289 | 0.8989 | 0.4607 | 8.012 | online |
| nft_mix | `nft_rl` | 400 | -- | -- | -- | -- | -- | -- | not logged (last online eval: 300; peak 7.969 @250) |
| dmdnft_mix | `dmdnft_rl` | 400 | 5.98 | 0.285 | 0.332 | 0.902 | 0.448 | 7.947 | paper Tab. 6(b) |
| dmdawm_mix | `dmdawm_rl` | 400 | 5.86 | 0.281 | 0.326 | 0.891 | 0.424 | 7.782 | paper Tab. 6(b) |
| fewstepgrpo_mix | `meanflowgrpo_rl` | 300 (peak) | 5.792 | 0.2811 | 0.3137 | 0.8733 | 0.4009 | 7.661 | online |
| flowgrpo_mix | `flowgrpo_rl` | 400 | 5.564 | 0.2837 | 0.2968 | 0.8713 | 0.4843 | 7.500 | online |
| meanflowawm_mix | `meanflowawm_rl` | 400 | 5.915 | 0.2786 | 0.3079 | 0.8902 | 0.4131 | 7.805 | offline |
| awm_mix | `awm_rl` | 400 | 6.046 | 0.2828 | 0.3332 | 0.8945 | 0.4278 | 7.984 | online |
| meanflowopsd_mix | `meanflowopsd_rl` | 400 | 5.906 | 0.3217 | 0.3734 | 0.9458 | 0.4928 | 8.040 | offline |
| reflmx_mix | `meanflowrefl_rl` | 400 | 5.891 | 0.3172 | 0.3675 | 0.9378 | 0.5459 | 8.060 | offline |

### OCR reward (train: ocr)

| arm | trainer | step | aesthetic | clipscore | ocr | mean | source |
|---|---|---|---|---|---|---|---|
| step 0 (Stage 1+2 only) | -- | 0 | 5.236 | 0.3196 | 0.4983 | 6.054 | online |
| meanflownft_ocr | `meanflow_nft` | 200 | 4.394 | 0.2894 | 0.934 | 5.617 | offline; ocr from paper Tab. 6(a) |
| dmdnft_ocr | `dmdnft_rl` | 200 | 4.406 | 0.2946 | 0.8973 | 5.598 | online |
| dmdawm_ocr | `dmdawm_rl` | 400 | 3.889 | 0.2703 | 0.9661 | 5.125 | online |
| fewstepgrpo_ocr | `meanflowgrpo_rl` | 200 (peak ocr) | 4.877 | 0.3251 | 0.7763 | 5.978 | online |
| meanflowawm_ocr | `meanflowawm_rl` | 400 | 5.011 | 0.3112 | 0.9271 | 6.249 | offline |

OCR numbers are PaddleOCR 3.x + PP-OCRv5 throughout; they are not comparable
with PaddleOCR 2.x scores (see `requirements.txt`).

## Notes per checkpoint

- **Released step.** The final saved step of each run, except the two
  `fewstepgrpo_*` arms, which release their peak step: `fewstepgrpo_mix` peaks
  at 300 (mean 7.661), and `fewstepgrpo_ocr` collapses after step 200 (ocr
  0.776 -> 0.03 by step 400). Every mix arm is released at or before step 400,
  the standard length (`dmdnft_mix` continued past 400 in its run; only its
  step-400 checkpoint is released).
- **dmdnft_ocr aux init.** The released `dmdnft_ocr` checkpoint was trained with
  the aux network warm-started from `aux_vel/mix` (its `config.yaml` records
  `aux_v_init_path: ./checkpoints/aux_vel/mix`). The training config
  `scripts/train/configs/dmdnft_ocr.yaml` points at `aux_vel/ocr`, the aux
  network distilled on OCR prompts, to match `dmdawm_ocr`. To reproduce the
  released checkpoint exactly, override
  `model.aux_v_init_path=./checkpoints/aux_vel/mix`.
- **Rollout type.** `meanflownft_mix`, `dmdnft_mix` and `dmdnft_ocr` were
  trained with the deterministic flow-map (ODE) rollout before
  `fewstep_rl.rollout_type` existed, so their `config.yaml` has no
  `rollout_type` key; their training configs pin `rollout_type: ode`. Every
  other saved config records its rollout. `nft_mix` and `meanflownft_ocr` used
  `sde` (the config default), so the `meanflownft_mix` / `nft_mix` ablation
  pair also differs in rollout type, not only in the S term.
- **reflmx_mix** (0-150, resumed 150-400) and **meanflownft_ocr** (0-150,
  resumed 150-200) were trained in two legs; their `config.yaml` is the second
  leg's with `train.resume_from` cleared.
- **meanflowopsd_mix / reflmx_mix** run with `fewstep_rl.disable_cd=true`.
- **dmdawm_*** are the no-central-difference arms (`x_anchor_2 = x_t - t*u_old`).
- Saved configs carry `fewstep_rl.num_epochs` = the step the run was stopped at
  (the shipped training configs keep 10000 and are stopped by hand), and the
  `max_aux_sample` value the config class derived at load time for `dmd*` arms.

## Provenance

| arm | source run (`outputs/` of the original repo) |
|---|---|
| meanflownft_mix | `sd35m_mfnft_klu_mix/20260826_194032` |
| nft_mix | `nft_mix/20261023_211056` |
| dmdnft_mix | `sd35m_dmdnft_klu_mix/20260828_150833` |
| dmdawm_mix | `dmdawm_mix/20261010_164630` |
| fewstepgrpo_mix | `sd35m_mfgrpoV_mix_clip3e3/20260908_164900` |
| flowgrpo_mix | `flowgrpo_mix/20261023_211059` |
| meanflowawm_mix | `sd35m_mfawm_klu_mix/20260909_144932` |
| awm_mix | `awm_mix/20261023_211058` |
| meanflowopsd_mix | `sd35m_mfopsd_vtnocd_mix/20260925_150906` |
| reflmx_mix | `meanflowreflmx_nocd_mix/20261004_150254` + `20261005_142542` |
| meanflownft_ocr | `meanflownft_klu_ocr_to200/20261012_182626` (resumed from step 150) |
| dmdnft_ocr | `dmdnft_klu_auxv_ocr/20260827_222412` |
| dmdawm_ocr | `dmdawm_nocd_u_ocr/20261008_211854` |
| fewstepgrpo_ocr | `sd35m_mfgrpoV_ocr_eta025_clip004/20260904_134338` |
| meanflowawm_ocr | `sd35m_mfawm_klu_ocr/20260907_185510` |
| aux_vel/mix | `sd35m_aux_vel/20260827_190542` (final `aux_v_transformer`, step 1000) |
| aux_vel/ocr | `sd35m_aux_vel_ocr/20260920_194139` (final `aux_v_transformer`, step 1000) |

`SHA256SUMS` lists every weight file in this directory.
