"""Dataclass configuration for few-step RL on SD3.5-Medium.

:class:`FewStepRLConfig` is the root config that ``main.py`` loads from YAML;
its ``fewstep_rl`` section (:class:`FewstepRLTrainConfig`) holds the settings
shared by every RL trainer. Unknown keys are rejected at load time (see
:func:`_dataclass_from_dict`), so every field of a saved ``config.yaml`` must
exist here.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields
from typing import Any

import yaml


def _apply_flat_overrides(data: dict, overrides: dict[str, Any]) -> dict:
    """Apply dot-separated command-line overrides to a nested dictionary."""
    for key, value in overrides.items():
        target = data
        parts = key.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return data


def _dataclass_from_dict(cls, data: Any):
    """Instantiate ``cls`` from ``data``, recursing into nested dataclasses.

    Raises on keys that are not fields of ``cls``, so a misspelled key fails
    instead of being ignored.
    """
    if not isinstance(data, dict):
        return data
    field_names = {item.name for item in fields(cls)}
    unknown = sorted(set(data).difference(field_names))
    if unknown:
        raise ValueError(
            f"Unknown {cls.__name__} field(s): {', '.join(unknown)}"
        )
    kwargs = {}
    for item in fields(cls):
        if item.name not in data:
            continue
        value = data[item.name]
        item_type = item.type
        if isinstance(item_type, str):
            item_type = eval(item_type, globals())
        if isinstance(value, dict) and hasattr(item_type, "__dataclass_fields__"):
            value = _dataclass_from_dict(item_type, value)
        kwargs[item.name] = value
    return cls(**kwargs)


@dataclass
class LoRAConfig:
    """LoRA configuration for one trainable transformer role."""

    enabled: bool = False
    rank: int = 32
    lora_alpha: int = 8
    target_modules: list = field(
        default_factory=lambda: [
            "attn.to_q",
            "attn.to_k",
            "attn.to_v",
            "attn.to_out.0",
        ]
    )
    init_lora_weights: str = "gaussian"
    load_path: str = ""
    # Earlier-stage adapters, merged in order before load_path.
    pre_merge_paths: list = field(default_factory=list)
    merge_before_training: bool = False


@dataclass
class ModelConfig:
    """SD3.5 model and role-specific initialization."""

    pretrained_path: str = ""
    generator_init_path: str = ""
    fake_score_init_path: str = ""
    teacher_init_path: str = ""
    # Init for the RL trainers' aux v-predictor: a previous run's output_dir
    # (its ``aux_v_transformer`` subdir is used), a diffusers transformer dir,
    # or a state-dict file. Empty = pretrained SD3.5 (same as ``ref_v_model``).
    aux_v_init_path: str = ""
    model_type: str = "sd35"
    gradient_checkpointing: bool = True
    dtype: str = "bf16"
    image_resolution: int = 1024
    generator_lora: LoRAConfig = field(default_factory=LoRAConfig)
    fake_score_lora: LoRAConfig = field(default_factory=LoRAConfig)


@dataclass
class FlowMapModelConfig:
    """Two-time ``(t, r)`` embedding wrapper that turns the SD3.5 transformer
    into a flow map (see ``fewsteprl/models/sd35_flowmap.py``)."""

    enabled: bool = True
    gate_value: float = 0.25
    deltatime_type: str = "r"

    def __post_init__(self) -> None:
        if self.deltatime_type not in {"r", "t-r"}:
            raise ValueError(
                "flowmap_model.deltatime_type must be 'r' or 't-r'; "
                f"got {self.deltatime_type!r}."
            )


@dataclass
class AnyFlowPretrainConfig:
    """Stage 1 forward-training parameters."""

    diffusion_ratio: float = 0.25
    consistency_ratio: float = 0.25
    epsilon: float = 0.5
    cd_velocity_source: str = "noise_minus_x0"
    v_target_source: str = "noise_minus_x0"
    forward_guidance_scale: float = 1.0
    weight_type: str = "gaussian"
    negative_embedding_path: str = ""
    drop_text_ratio: float = 0.1
    ema_warmup_steps: int = 0

    def __post_init__(self) -> None:
        total = self.diffusion_ratio + self.consistency_ratio
        if self.diffusion_ratio < 0 or self.consistency_ratio < 0 or total > 1.0 + 1e-6:
            raise ValueError(
                "anyflow_pretrain diffusion_ratio and consistency_ratio must be "
                f"non-negative and sum to at most 1; got {total}."
            )
        valid_sources = {"noise_minus_x0", "u_self"}
        if self.cd_velocity_source not in valid_sources:
            raise ValueError(
                "anyflow_pretrain.cd_velocity_source must be 'noise_minus_x0' "
                f"or 'u_self'; got {self.cd_velocity_source!r}."
            )
        if self.v_target_source not in valid_sources:
            raise ValueError(
                "anyflow_pretrain.v_target_source must be 'noise_minus_x0' "
                f"or 'u_self'; got {self.v_target_source!r}."
            )
        if not 0.0 <= self.drop_text_ratio <= 1.0:
            raise ValueError(
                "anyflow_pretrain.drop_text_ratio must be in [0, 1]; "
                f"got {self.drop_text_ratio}."
            )


@dataclass
class AnyFlowOnPolicyConfig:
    """Stage 2 on-policy flow-map distribution matching."""

    num_inference_steps_list: list = field(default_factory=lambda: [4, 8, 16, 32])
    dmd_weight: float = 1.0
    dmd_batch_size: int = 0
    dmd_min_timestep: int = 0
    dmd_max_timestep: int = 1000
    real_guidance_scale: float = 0.0
    gradient_normalization: bool = True
    cotrain_forward_kl: bool = True
    discriminator_update_ratio: int = 1
    rollout_detach_between_jumps: bool = False
    rollout_full_steps: bool = False

    def __post_init__(self) -> None:
        if not self.num_inference_steps_list or any(
            int(step) <= 0 for step in self.num_inference_steps_list
        ):
            raise ValueError(
                "anyflow_onpolicy.num_inference_steps_list must contain "
                "positive integers."
            )
        if self.dmd_min_timestep > self.dmd_max_timestep:
            raise ValueError(
                "anyflow_onpolicy.dmd_min_timestep must not exceed "
                "dmd_max_timestep."
            )
        if self.discriminator_update_ratio < 1:
            raise ValueError(
                "anyflow_onpolicy.discriminator_update_ratio must be at least 1."
            )
        if self.rollout_full_steps and not self.rollout_detach_between_jumps:
            raise ValueError(
                "anyflow_onpolicy.rollout_full_steps=True requires "
                "rollout_detach_between_jumps=True."
            )


@dataclass
class AnyFlowDMDConfig:
    """Stage 2 in the DMD loop shape.

    The losses are configured by :class:`AnyFlowOnPolicyConfig`; this section
    only sets the per-``train_step`` loop:

        1. one no-grad rollout into a shared buffer,
        2. ``num_inner_epochs_fake`` sweeps of fake-score flow-matching
           updates over that buffer,
        3. ``freq_generator_update`` generator DMD updates, each with its
           own fresh differentiable rollout.

    ``anyflow_onpolicy`` instead runs a generator update followed by N
    discriminator updates, each with its own rollout.
    """

    # Phase 1: size of the no-grad rollout buffer. 0 => the whole dataloader
    # batch. Clamped to ``train.batch_size`` because prompts come from the
    # batch; repeating prompts to fill a larger buffer would change the
    # conditioning distribution.
    rollout_batch_size: int = 0

    # Phase 2: fake-score inner loop. ``fm_inner_batchsize=0`` means one step
    # over the whole buffer. The loop runs
    # ``num_inner_epochs_fake * ceil(n / fm_inner_batchsize)`` optimizer steps.
    fm_inner_batchsize: int = 0
    num_inner_epochs_fake: int = 1

    # Phase 3: generator DMD updates per train_step. Each pass draws its own
    # rank-synced (sample_step, grad_timestep) and its own fresh noise, so the
    # passes are independent rollouts over the same prompts.
    freq_generator_update: int = 1

    # Generator sub-batch for the DMD branch, scoped to the buffer.
    # 0 => the whole buffer.
    dmd_batch_size: int = 0

    # A constraint, not a tuning knob: "fresh_with_grad" is the only accepted
    # value. DMD needs a differentiable sample -- the loss
    # ``mse(pred_video, (pred_video - grad).detach())`` is a
    # distribution-matching gradient only when ``pred_video`` depends on the
    # generator parameters through the sampler. Feeding the no-grad Phase-1
    # buffer instead optimises a reconstruction, a different objective, so the
    # buffer only feeds the fake score's flow-matching loss (which needs just
    # a detached ``x0``).
    generator_rollout_source: str = "fresh_with_grad"

    def __post_init__(self) -> None:
        for name in (
            "rollout_batch_size", "fm_inner_batchsize", "dmd_batch_size",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"anyflow_dmd.{name} must be >= 0.")
        if self.num_inner_epochs_fake < 1:
            raise ValueError(
                "anyflow_dmd.num_inner_epochs_fake must be at least 1."
            )
        if self.freq_generator_update < 1:
            raise ValueError(
                "anyflow_dmd.freq_generator_update must be at least 1."
            )
        if self.generator_rollout_source != "fresh_with_grad":
            raise ValueError(
                "anyflow_dmd.generator_rollout_source must be "
                "'fresh_with_grad'; got "
                f"{self.generator_rollout_source!r}. DMD requires a "
                "differentiable sample. The Phase-1 rollout buffer is no-grad, "
                "so using it as the generator's pred_video optimises a "
                "reconstruction rather than the sampling distribution — a "
                "different objective. The buffer may only feed the fake-score "
                "FM loss."
            )

@dataclass
class FewstepRLTrainConfig:
    """``fewstep_rl`` section: settings shared by every RL trainer."""


    # Epoch loop and sampling.
    num_epochs: int = 1000
    num_batches_per_epoch: int = 1
    num_inner_epochs: int = 1
    rl_inner_batch_size: int = 0
    rl_start_epoch: int = 0
    num_image_per_prompt: int = 4
    rl_sample_use_ema: bool = False
    sampling_num_steps: int = 4
    sampling_guidance_scale: float = 1.0
    # T5 max sequence length for ``encode_prompts_sd35``. 128 is the length the
    # MeanFlowNFT checkpoints were trained and evaluated with.
    text_max_sequence_length: int = 128

    # Advantage and policy loss.
    per_prompt_stat_tracking: bool = True
    per_prompt_global_std: bool = True
    weight_advantages: bool = False
    adv_clip_max: float = 5.0
    beta: float = 0.1
    kl_weight: float = 0.0
    #: PPO likelihood-ratio clip epsilon, read by the GRPO and AWM trainers
    #: (MeanFlowGRPOTrainer, FlowGRPOTrainer, MeanFlowAWMTrainer, AWMTrainer,
    #: DMDAWMTrainer). The NFT, DMD-NFT, OPSD and ReFL trainers have no
    #: likelihood ratio to clip.
    #:
    #: 1e-3, not the 0.2 familiar from LLM PPO: the log-density is a
    #: per-element mean over the latent (as in flow-GRPO's
    #: ``log_prob.mean(dim=...)``), so ``rho`` stays within ~1e-3 of 1 and a
    #: 0.2 clip would never bind. The default is flow-GRPO's base
    #: ``config.train.clip_range``. Its ``*_fast`` recipes use 1e-5, but their
    #: ``cps`` log-prob drops the ``1/(2 tau^2)`` normalizer that this kernel
    #: keeps, so that value is on a different scale.
    clip_range: float = 1e-3
    #: Noise level of the GRPO transition kernel (flow-GRPO's
    #: ``config.sample.noise_level``), read by MeanFlowGRPOTrainer and
    #: FlowGRPOTrainer. It sets
    #: ``std_dev_t = noise_level * sqrt(sigma / (1 - sigma))`` and hence both
    #: the kernel std ``tau = std_dev_t * sqrt(h)`` and the drift coefficients
    #: ``a`` / ``b`` (flow-GRPO's ``transition_coeffs``). 0.7 is flow-GRPO's
    #: default for the ``sde`` kernel.
    grpo_noise_level: float = 0.7

    # Old-policy update.
    decay_type: int = 1
    rl_decay_interval: int = 1

    # Reward scoring and logging.
    reward_fn: dict = field(default_factory=dict)
    reward_ckpt_path: str = ""
    log_sample_images: int = 0

    central_diff_epsilon: float = 5.0
    # Direction the central difference perturbs ``x_t`` along:
    # "noise_minus_x0" (``eps - x_0``, with ``eps`` recovered from x_t, x_0, t)
    # or "u_self" (the CD model's own ``u(x_t, t, t)``).
    cd_velocity_source: str = "noise_minus_x0"
    share_cd_with_old: bool = True

    #: OPSD only: skip the central difference and define ``dF/dt := 0``
    #: (S = 0), saving the two ``old_model`` forwards it costs per step. Unlike
    #: ``diffusion_ratio=1.0`` -- the other way to get no ``dF/dt`` -- this
    #: leaves the (t, r) sampler untouched, so r stays random.
    #:
    #: S = (t - r) * dF/dt cancels exactly in the differentiable channel: both
    #: ``_opsd_clean_output`` and ``_compute_x_anchor_1`` add the same S, so
    #: ``x_0_pred_v - x_anchor_1 == -sigma*(u_theta - u_old)`` either way. It
    #: does not cancel in the stop-grad coefficients built from anchor
    #: positions (the trust-region budget ``rho*||x_start||``,
    #: ``W = beta^2 (a_+ + a_-)`` and the reward force), which all move by
    #: ``-sigma*S``. So this is a slightly different objective, not an exact
    #: fast path; it stays consistent because target building and fitting both
    #: drop S. Rejected in ``__post_init__`` unless ``opsd_enabled``: in the
    #: other trainers S enters the target directly.
    disable_cd: bool = False

    # Fresh MeanFlowNFT (t, r) pairs used by every inner update.
    num_training_timesteps_per_sample: int = 4
    rl_min_timestep: int = 0
    rl_max_timestep: int = 1000
    diffusion_ratio: float = 0.25
    consistency_ratio: float = 0.25

    # KL reference (built only when kl_weight > 0). False: a frozen snapshot
    # of the starting generator, KL in u-space. True: the pretrained SD3.5
    # v-predictor, KL in v-space, queried with CFG scale ``guidance_scale``.
    use_ref_v_model: bool = False
    guidance_scale: float = 1.0

    # Auxiliary v-predictor (``aux_v_model``), trained by _run_aux_epoch with a
    # plain flow-matching loss on per-GPU rollouts. Required by dmdnft_rl and
    # dmdawm_rl. Off by default: enabling it costs a second trainable
    # transformer plus its AdamW state for the whole run.
    # Every count here is PER GPU. The aux model is DDP/FSDP-wrapped, so one
    # optimizer step sees ``aux_v_batch_size * world_size`` samples globally.
    aux_v_enabled: bool = False
    # 0 = derive as renoise_per_epoch * aux_v_batch_size, i.e. roll out exactly
    # enough for every sample to be trained on once and no more.
    max_aux_sample: int = 0          # rollout samples per GPU in the aux pack
    renoise_per_epoch: int = 10       # fm-loss evaluations over that pack
    aux_v_gradient_accum: int = 1     # backwards per aux optimizer step
    aux_v_batch_size: int = 0         # microbatch per fm-loss; 0 = whole pack
    rollout_type: str = "sde"   # "sde" or "ode" for rollout method
    reuse_x_t: bool = False   # whether to reuse the x_t from the rollout process

    # ==================================================================
    # DiffusionOPSD (on-policy self-distillation) -- read by
    # MeanFlowOPSDTrainer (meanflowopsd.py) and MeanFlowReFLTrainer
    # (meanflowrefl.py). With ``opsd_enabled`` False (the default) the block
    # is unread and its validation in __post_init__ is skipped.
    #
    # Names mirror DiffusionOPSD's own config keys (``opa_*``) with an
    # ``opsd_`` prefix.
    # ==================================================================

    #: Master switch; the OPSD and ReFL trainers raise without it.
    opsd_enabled: bool = False

    #: Trust-region radius factor rho (paper App. C.1, ``opa_rho``): each
    #: reward target moves at most ``rho * ||anchor||`` away from its anchor.
    opsd_rho: float = 0.10

    #: Query noise level sigma* (the paper's SD3.5-M value, ``opa_query_sigma``).
    #:
    #: Not on the rollout grid: with shift=3.0 and sampling_num_steps=4 the
    #: grid sigmas are {1.0, 0.9, 0.75, 0.5, 0.0}, whose nearest point is 0.5.
    #: ``opsd_state_mode="forward"`` (the default) therefore forward-noises to
    #: this exact sigma instead of snapping to the grid.
    opsd_query_sigma: float = 0.278

    #: M_tgt: number of normalized reward-gradient steps per target
    #: (``opa_n_ascent``); each step is one reward backward. 0 means no
    #: displacement (the targets equal the anchor).
    opsd_n_ascent: int = 2

    #: eta_tgt, the target-step multiplier (``opa_eta``). At eta <= 1 the
    #: trust-region projection is inactive in exact arithmetic; > 1 stress-tests
    #: the projection.
    opsd_eta_tgt: float = 1.0

    #: Which differentiable reward the target ascent follows
    #: (``opa_reward_kind``): "hpsv2", "clipscore", "pickscore", "aesthetic",
    #: or "mixed". Independent of ``reward_fn``, which still scores the rollout
    #: and therefore still sets the advantages: this is only the direction the
    #: targets move in. "hpsv2" is one CLIP-H backward per step; "mixed"
    #: follows the full weighted ``reward_fn`` sum (every term must be
    #: differentiable) at N times the cost.
    opsd_reward_kind: str = "hpsv2"

    #: Samples per reward-gradient chunk (``opa_mb``). Bounds peak memory of
    #: the VAE+reward backward, which is the tallest allocation in the phase.
    opsd_target_chunk: int = 6

    #: eps_g: gradient-normalization stabilizer (paper App. C.1).
    opsd_eps_grad: float = 1e-12

    #: eps_gamma: floor on the adaptive normalizers gamma_pm (paper App. C.1).
    #: Matches the released kernel's ``.clip(min=1e-5)``.
    opsd_eps_gamma: float = 1e-5

    #: Dual-negative branch (``opa_dual_neg``). False = positive-branch only,
    #: the paper's dual-negative ablation.
    opsd_dual_neg: bool = True

    #: Query-state provenance (``opa_state_mode``):
    #:   "forward" -- forward-noise the rollout endpoint to the exact
    #:       ``opsd_query_sigma``. Needs neither reuse_x_t nor an SDE rollout.
    #:   "rollout" -- take the rollout's own state at the nearest on-grid
    #:       sigma (strictly on-policy, but see ``opsd_query_sigma``: on a
    #:       4-step grid the nearest is 0.5). Requires reuse_x_t=True and
    #:       ``opsd_sigma_mode="fixed"``.
    opsd_state_mode: str = "forward"

    #: Target direction (``opa_dir_mode``). "grad" is the method; the other
    #: three are the paper's attribution controls, separating the effect of
    #: the reward direction from that of a trust-region perturbation:
    #:   "grad"     -- normalized reward-gradient steps (the method)
    #:   "rand"     -- a random unit direction, same radius, no reward
    #:   "residual" -- the denoising residual direction (x_0 - anchor)
    #:   "noop"     -- no displacement (both targets equal the anchor), so the
    #:                 loss reduces to the proximal anchor term
    opsd_dir_mode: str = "grad"

    # ---- (t, r) coverage of the OPSD fit ------------------------------
    #
    # With the defaults ("fixed", "zero") every sample is fitted at the single
    # query pair (t = opsd_query_sigma, r = 0). On a 4-step flow map that t is
    # on no inference grid point ({1.0, 0.9, 0.75, 0.5, 0.0} at shift 3.0), and
    # with deltatime_type="r" the delta embedder only ever sees r = 0 while
    # inference uses four values. The two fields below spread t and r instead.
    #:
    #: Query noise level law:
    #:   "fixed"        -- sigma_q = opsd_query_sigma for every sample.
    #:   "base_sampler" -- per-sample t from the three-mode law the other
    #:       trainers fit on (``_sample_training_tr_three_mode``). This gives up
    #:       OPSD's low-noise query: under the default uniform renoise law
    #:       only ~1.3% of its mass lies below 0.278.
    #:   "lognormal"    -- per-sample t01 ~ sigmoid(opsd_logit_mean +
    #:       opsd_logit_std * N(0,1)), then SD3.5's static shift: the
    #:       logit-normal family of the pretraining stages, centred on
    #:       ``opsd_query_sigma``. With the defaults the shifted sigma has
    #:       median 0.278 and 5-95% range [0.125, 0.508].
    #: Both per-sample modes require ``opsd_state_mode="forward"``.
    opsd_sigma_mode: str = "fixed"

    #: Only read when ``opsd_sigma_mode == "lognormal"``. The default mean
    #: logit(inverse_shift(0.278)) = logit(0.1137) = -2.053 centres the shifted
    #: sigma on ``opsd_query_sigma`` at SD3.5's shift = 3.0; a different shift
    #: needs a different mean.
    opsd_logit_mean: float = -2.053
    opsd_logit_std: float = 0.6

    # ------------------------------------------------------------------
    # Renoise distribution: the law of the training t
    # ------------------------------------------------------------------
    #: Law ``GeneralRLTrainer._sample_training_tr_three_mode`` draws the
    #: training ``t`` from. That sampler serves every trainer that re-noises
    #: x_0 at a fresh (t, r) each inner iteration (the NFT, DMD-NFT, AWM and
    #: DMD-AWM trainers), plus ``opsd_sigma_mode="base_sampler"``.
    #:
    #:   "uniform"   -- t01 = max(t1, t2), r01 = min(t1, t2), t1, t2 ~ U(0,1).
    #:       The default, used by every shipped config.
    #:   "lognormal" -- t01 = sigmoid(mean + std * N(0,1)) and r01 ~ U(0, t01),
    #:       the same conditional law of r given t as above. This is the
    #:       logit-normal family of the pretraining stages.
    #:
    #: Both then apply SD3.5's static shift and the three-mode partition.
    renoise_distribution: str = "uniform"

    #: Only read when ``renoise_distribution == "lognormal"``. Same
    #: parameterisation as ``opsd_logit_mean`` / ``opsd_logit_std``, which
    #: govern a different quantity (the OPSD query noise level, not the
    #: training t). The default mean -2.053 = logit(inverse_shift(0.278)) at
    #: shift = 3.0 puts the median of the shifted t at 0.278; a different shift
    #: needs a different mean.
    renoise_logit_mean: float = -2.053
    renoise_logit_std: float = 0.6

    #: Flow-map second argument r during fitting:
    #:   "zero"    -- r = 0, the clean-output query.
    #:   "given_t" -- r drawn by ``_draw_training_s_given_t``, i.e. r's own
    #:       conditional law given t (r01 | t01 ~ U(0, t01) pre-shift), with the
    #:       three-mode masks pinning r = t on the diffusion slice and r = 0 on
    #:       the consistency slice (so some samples keep the exact inference
    #:       pair). The fitting phase replays the r each anchor was built at.
    #:       Relies on the v-form clean output
    #:       ``x_0_pred_v = z_q - sigma_q * (u + (t - r) * sg(dF/dt))``, which
    #:       the MeanFlow identity makes invariant to r, so r becomes a free
    #:       training axis that does not move the OPSD target space (a u-form
    #:       ``z_q - sigma_q * u`` would shift by (t - r) * dF/dt). Requires
    #:       ``opsd_state_mode="forward"``.
    opsd_r_mode: str = "zero"

    #: Gain on the advantage-dependent (odd) part of the OPSD reward
    #: coefficient, ``Lambda = a_+ - a_-``, at fixed ``Sigma = a_+ + a_-``;
    #: 1.0 is inert. Not read by any trainer in this repo; validated to be > 0
    #: when ``opsd_enabled``.
    opsd_adv_gain: float = 1.0

    # ---- ReFL corner (MeanFlowReFLTrainer, meanflowrefl.py) ------------
    #
    #: Whether MeanFlowReFLTrainer weights its single reward square by the
    #: advantage. Other trainers ignore it. The two settings are different
    #: objectives:
    #:
    #:   False (default) -- ``r_hat == 1``: every sample is pushed up the reward
    #:       with the same weight, as in ReFL, which uses no advantage.
    #:   True -- use the OPSD trainers' Eq. 5 ``r_hat`` (advantage-weighted
    #:       ReFL). In the one-square loss ``r_hat`` then survives as the
    #:       per-sample multiplier ``a_+ = r_hat/gamma_+``, so a sample whose
    #:       advantage sits at the bottom of the clip range contributes almost
    #:       no force -- unlike the two-square OPSD loss, where ``r_hat`` only
    #:       splits a fixed total (``omega_1 + omega_2 == 1/beta``).
    refl_use_advantage: bool = False

    def __post_init__(self) -> None:
        if self.sampling_num_steps < 1:
            raise ValueError("fewstep_rl.sampling_num_steps must be at least 1.")
        # Validated for every trainer, not only OPSD: the sampler treats any
        # value other than "uniform" as "lognormal", so a typo must fail here.
        if self.renoise_distribution not in {"uniform", "lognormal"}:
            raise ValueError(
                "fewstep_rl.renoise_distribution must be 'uniform' or "
                f"'lognormal'; got {self.renoise_distribution!r}."
            )
        if self.renoise_distribution == "lognormal" and self.renoise_logit_std <= 0.0:
            raise ValueError(
                "fewstep_rl.renoise_logit_std must be > 0 for "
                f"renoise_distribution='lognormal'; got {self.renoise_logit_std!r}."
            )
        if self.renoise_per_epoch < 1:
            raise ValueError("fewstep_rl.renoise_per_epoch must be at least 1.")
        if self.aux_v_gradient_accum < 1:
            raise ValueError("fewstep_rl.aux_v_gradient_accum must be at least 1.")
        if self.aux_v_batch_size < 0:
            raise ValueError(
                "fewstep_rl.aux_v_batch_size must be >= 0 (0 means the whole pack)."
            )
        if self.max_aux_sample < 0:
            raise ValueError(
                "fewstep_rl.max_aux_sample must be >= 0 (0 derives it from "
                "renoise_per_epoch * aux_v_batch_size)."
            )
        # Everything below is gated on aux_v_enabled, including the derivation:
        # both fields default to 0 and each derives from the other, so an
        # ungated check would reject every config that has no aux network and
        # never names either one.
        if self.aux_v_enabled:
            if self.max_aux_sample == 0:
                # Resolved here rather than at read time so that every consumer
                # -- _aux_sampling_phase, the coverage check below, and the
                # config.yaml written into the checkpoint -- sees one concrete
                # number.
                if self.aux_v_batch_size < 1:
                    raise ValueError(
                        "fewstep_rl.max_aux_sample=0 derives the pack size from "
                        "aux_v_batch_size, but aux_v_batch_size=0 derives the "
                        "microbatch from the pack size. Set one of the two to a "
                        "positive value."
                    )
                self.max_aux_sample = (
                    self.renoise_per_epoch * self.aux_v_batch_size
                )
            # The aux loop consumes the pack as a cycling permutation, so a
            # sample is guaranteed to be seen only once a full sweep completes.
            # Under-running the sweep silently throws away rollouts, which are
            # the most expensive thing in the whole aux epoch.
            micro = self.aux_v_batch_size or self.max_aux_sample
            sweep_len = (self.max_aux_sample + micro - 1) // micro
            if self.renoise_per_epoch < sweep_len:
                raise ValueError(
                    "fewstep_rl.renoise_per_epoch="
                    f"{self.renoise_per_epoch} is too small to consume "
                    f"max_aux_sample={self.max_aux_sample} at a microbatch of "
                    f"{micro}: {sweep_len} evaluations are needed for every "
                    "rolled-out sample to be trained on at least once. Raise "
                    "renoise_per_epoch to at least that, or lower "
                    "max_aux_sample."
                )
            if self.renoise_per_epoch < self.aux_v_gradient_accum:
                raise ValueError(
                    "fewstep_rl.renoise_per_epoch="
                    f"{self.renoise_per_epoch} is below aux_v_gradient_accum="
                    f"{self.aux_v_gradient_accum}, so the accumulation window "
                    "never fills and the aux model would take ZERO optimizer "
                    "steps per epoch."
                )
        if self.num_training_timesteps_per_sample < 1:
            raise ValueError(
                "fewstep_rl.num_training_timesteps_per_sample must be at least 1."
            )
        if self.text_max_sequence_length < 1:
            raise ValueError(
                "fewstep_rl.text_max_sequence_length must be at least 1."
            )
        if self.cd_velocity_source not in {"noise_minus_x0", "u_self"}:
            raise ValueError(
                "fewstep_rl.cd_velocity_source must be 'noise_minus_x0' or "
                f"'u_self'; got {self.cd_velocity_source!r}."
            )
        total = self.diffusion_ratio + self.consistency_ratio
        if self.diffusion_ratio < 0 or self.consistency_ratio < 0 or total > 1.0 + 1e-6:
            raise ValueError(
                "fewstep_rl.diffusion_ratio and consistency_ratio must be "
                f"non-negative and sum to at most 1; got {total}."
            )
        if self.rl_min_timestep > self.rl_max_timestep:
            raise ValueError(
                "fewstep_rl.rl_min_timestep must not exceed rl_max_timestep."
            )
        if self.adv_clip_max <= 0:
            raise ValueError("fewstep_rl.adv_clip_max must be positive.")
        if self.beta <= 0:
            raise ValueError("fewstep_rl.beta must be positive.")
        # Upper bound as well as lower: with a per-element-mean log-density the
        # ratio stays within ~1e-3 of 1, so an LLM-PPO-scale clip (0.2) would
        # never fire and silently disable the trust region instead of erroring.
        # 0.1 is far above any value flow-GRPO ships (1e-3 .. 1e-5).
        if not 0.0 < self.clip_range <= 0.1:
            raise ValueError(
                "fewstep_rl.clip_range must be in (0, 0.1]; got "
                f"{self.clip_range!r}. The log-density is a per-element mean "
                "over the latent, so rho sits within ~1e-3 of 1 and a "
                "0.2-style epsilon never binds. Official flow-GRPO uses 1e-3 "
                "for the sde kernel."
            )
        # Upper bound because std_dev_t = noise_level * sqrt(sigma/(1-sigma))
        # already diverges as sigma -> 1; flow-GRPO ships 0.7-1.5 across its
        # recipes, so anything past ~3 is a typo rather than a choice.
        if not 0.0 < self.grpo_noise_level <= 3.0:
            raise ValueError(
                "fewstep_rl.grpo_noise_level must be in (0, 3]; got "
                f"{self.grpo_noise_level!r}. flow-GRPO's sde default is 0.7."
            )

        if self.rollout_type not in {"sde", "ode"}:
            raise ValueError(
                "fewstep_rl.rollout_type must be 'sde' or 'ode'; "
                f"got {self.rollout_type!r}."
            )

        # ------------------------------------------------------------------
        # DiffusionOPSD block, validated only when ``opsd_enabled`` (as the
        # aux block above is gated on ``aux_v_enabled``), so configs that do
        # not use OPSD are unaffected.
        #
        # Every predicate below is a config constant, hence rank-uniform:
        # these raise identically on all ranks and cannot desync a collective.
        # ------------------------------------------------------------------
        # ``disable_cd`` relies on the OPSD loss's S cancellation (see the
        # field's own note). In the other trainers S enters the target
        # directly, so honouring it there would silently change their
        # objective.
        if self.disable_cd and not self.opsd_enabled:
            raise ValueError(
                "fewstep_rl.disable_cd is OPSD-specific (the cd term cancels "
                "in the OPSD loss's differentiable channel) but "
                "opsd_enabled=false. Drop disable_cd, or use "
                "diffusion_ratio=1.0 if you want an r == t run without the "
                "central difference on a non-OPSD trainer."
            )

        if self.opsd_enabled:
            if not 0.0 < self.opsd_rho <= 1.0:
                raise ValueError(
                    "fewstep_rl.opsd_rho is the trust-region radius as a "
                    f"fraction of the anchor norm, so it must lie in (0, 1]; got "
                    f"{self.opsd_rho!r}. The paper value is 0.10."
                )
            # Strictly interior: sigma=0 is a clean sample (no query to make)
            # and sigma=1 is pure noise (the anchor carries no signal).
            if not 0.0 < self.opsd_query_sigma < 1.0:
                raise ValueError(
                    "fewstep_rl.opsd_query_sigma must be in (0, 1) -- it is a "
                    f"NORMALIZED noise level, not a raw timestep; got "
                    f"{self.opsd_query_sigma!r}. The SD3.5-M paper value is "
                    "0.278 (raw 278.0)."
                )
            if self.opsd_n_ascent < 0:
                raise ValueError(
                    "fewstep_rl.opsd_n_ascent must be >= 0 (0 means the targets "
                    f"equal the anchor, i.e. no reward displacement); got "
                    f"{self.opsd_n_ascent!r}."
                )
            if self.opsd_eta_tgt <= 0:
                raise ValueError(
                    "fewstep_rl.opsd_eta_tgt must be positive; got "
                    f"{self.opsd_eta_tgt!r}."
                )
            if self.opsd_target_chunk < 1:
                raise ValueError(
                    "fewstep_rl.opsd_target_chunk is a microbatch size and "
                    f"must be >= 1; got {self.opsd_target_chunk!r}."
                )
            if self.opsd_eps_grad <= 0 or self.opsd_eps_gamma <= 0:
                raise ValueError(
                    "fewstep_rl.opsd_eps_grad and opsd_eps_gamma are "
                    "stabilizers and must be positive; got "
                    f"{self.opsd_eps_grad!r} / {self.opsd_eps_gamma!r}."
                )
            if self.opsd_state_mode not in {"forward", "rollout"}:
                raise ValueError(
                    "fewstep_rl.opsd_state_mode must be 'forward' or "
                    f"'rollout'; got {self.opsd_state_mode!r}."
                )
            if self.opsd_dir_mode not in {"grad", "rand", "residual", "noop"}:
                raise ValueError(
                    "fewstep_rl.opsd_dir_mode must be one of 'grad', 'rand', "
                    f"'residual', 'noop'; got {self.opsd_dir_mode!r}."
                )
            if self.opsd_sigma_mode not in {"fixed", "base_sampler", "lognormal"}:
                raise ValueError(
                    "fewstep_rl.opsd_sigma_mode must be 'fixed', "
                    f"'base_sampler' or 'lognormal'; got {self.opsd_sigma_mode!r}."
                )
            if self.opsd_sigma_mode == "lognormal" and self.opsd_logit_std <= 0.0:
                raise ValueError(
                    "fewstep_rl.opsd_logit_std must be > 0 for "
                    f"opsd_sigma_mode='lognormal'; got {self.opsd_logit_std!r}."
                )
            if self.opsd_r_mode not in {"zero", "given_t"}:
                raise ValueError(
                    "fewstep_rl.opsd_r_mode must be 'zero' or 'given_t'; got "
                    f"{self.opsd_r_mode!r}."
                )
            if self.opsd_adv_gain <= 0:
                raise ValueError(
                    "fewstep_rl.opsd_adv_gain must be > 0 (1.0 is the inert "
                    f"default); got {self.opsd_adv_gain!r}. A non-positive gain "
                    "would flip the sign of the advantage-dependent half of the "
                    "reward coefficient, training the policy to prefer the "
                    "LOW-advantage branch."
                )
            # Caught here rather than as a wrong-but-running experiment.
            if self.opsd_r_mode == "given_t" and self.opsd_state_mode != "forward":
                raise ValueError(
                    "fewstep_rl.opsd_r_mode='given_t' requires "
                    "opsd_state_mode='forward': the on-grid 'rollout' state is "
                    "tied to the rollout's own (t, r) pair, so redrawing r "
                    "would decouple the query state from its own transition."
                )
            # "rollout" reads the per-step trajectory out of the sample pack,
            # which only the return_intermediate path populates. Caught here
            # rather than as a KeyError three phases later.
            if self.opsd_state_mode == "rollout" and not self.reuse_x_t:
                raise ValueError(
                    "fewstep_rl.opsd_state_mode='rollout' takes the query "
                    "state off the rollout trajectory, so it requires "
                    "reuse_x_t=true. Use opsd_state_mode='forward' to "
                    "forward-noise the endpoint instead (the default, and the "
                    "only way to hit opsd_query_sigma exactly on a short grid)."
                )


@dataclass
class MeanFlowNFTRLConfig:
    """MeanFlowNFT reward fine-tuning configuration (``meanflow_nft`` section
    of :class:`MeanFlowNFTConfig`)."""

    # Epoch loop and sampling.
    num_epochs: int = 1000
    num_batches_per_epoch: int = 1
    num_inner_epochs: int = 1
    nft_inner_batch_size: int = 0
    nft_start_epoch: int = 0
    num_image_per_prompt: int = 4
    nft_sample_use_ema: bool = False
    sampling_num_steps: int = 4
    sampling_guidance_scale: float = 1.0
    # T5 max sequence length for ``encode_prompts_sd35``. 128 is the length the
    # MeanFlowNFT checkpoints were trained and evaluated with.
    text_max_sequence_length: int = 128

    # Advantage and policy loss.
    per_prompt_stat_tracking: bool = True
    per_prompt_global_std: bool = True
    weight_advantages: bool = False
    adv_clip_max: float = 5.0
    beta: float = 0.1
    kl_weight: float = 0.0
    # Which algebraic form of the (identical) policy objective to evaluate.
    # 'posneg' -- the DiffusionNFT reference form: two implicit policies
    #   x0_plus/x0_minus, each regressed onto the rollout sample x_0 with its
    #   own adaptive weight.
    # 'affine' -- one regression onto a single stop-grad affine target
    #   x_target = x_ema + adv_coeff*(x_0 - x_ema), scaled by coeff = beta*(a+b).
    # The two share the same policy gradient; their loss values differ by a
    # theta-independent constant, so the logged loss scalar is NOT comparable
    # across the two forms.
    nft_loss_form: str = "posneg"

    # Old-policy update.
    decay_type: int = 1
    nft_decay_interval: int = 1

    # Reward scoring and logging.
    reward_fn: dict = field(default_factory=dict)
    reward_ckpt_path: str = ""
    log_sample_images: int = 0

    # MeanFlowNFT velocity construction.
    nft_velocity_mode: str = "meanflow_v"
    central_diff_epsilon: float = 5.0
    cd_velocity_source: str = "noise_minus_x0"
    share_cd_with_old: bool = True

    # Fresh MeanFlowNFT (t, r) pairs used by every inner update.
    num_training_timesteps_per_sample: int = 4
    nft_min_timestep: int = 0
    nft_max_timestep: int = 1000
    diffusion_ratio: float = 0.25
    consistency_ratio: float = 0.25

    def __post_init__(self) -> None:
        if self.sampling_num_steps < 1:
            raise ValueError("meanflow_nft.sampling_num_steps must be at least 1.")
        if self.num_training_timesteps_per_sample < 1:
            raise ValueError(
                "meanflow_nft.num_training_timesteps_per_sample must be at least 1."
            )
        if self.text_max_sequence_length < 1:
            raise ValueError(
                "meanflow_nft.text_max_sequence_length must be at least 1."
            )
        if self.nft_velocity_mode not in {"meanflow_v", "direct_u"}:
            raise ValueError(
                "meanflow_nft.nft_velocity_mode must be 'meanflow_v' or "
                f"'direct_u'; got {self.nft_velocity_mode!r}."
            )
        if self.cd_velocity_source not in {"noise_minus_x0", "u_self"}:
            raise ValueError(
                "meanflow_nft.cd_velocity_source must be 'noise_minus_x0' or "
                f"'u_self'; got {self.cd_velocity_source!r}."
            )
        total = self.diffusion_ratio + self.consistency_ratio
        if self.diffusion_ratio < 0 or self.consistency_ratio < 0 or total > 1.0 + 1e-6:
            raise ValueError(
                "meanflow_nft diffusion_ratio and consistency_ratio must be "
                f"non-negative and sum to at most 1; got {total}."
            )
        if self.nft_min_timestep > self.nft_max_timestep:
            raise ValueError(
                "meanflow_nft.nft_min_timestep must not exceed nft_max_timestep."
            )
        if self.adv_clip_max <= 0:
            raise ValueError("meanflow_nft.adv_clip_max must be positive.")
        if self.beta <= 0:
            raise ValueError("meanflow_nft.beta must be positive.")
        if self.nft_loss_form not in {"posneg", "affine"}:
            raise ValueError(
                "meanflow_nft.nft_loss_form must be 'posneg' or 'affine'; got "
                f"{self.nft_loss_form!r}."
            )


@dataclass
class GeneratorSolverConfig:
    lr: float = 1e-6
    warmup_steps: int = 0
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0


@dataclass
class FakeSolverConfig:
    lr: float = 1e-6
    warmup_steps: int = 500
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0


@dataclass
class AuxSolverConfig:
    lr: float = 1e-5
    warmup_steps: int = 0
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0


@dataclass
class SolverConfig:
    generator: GeneratorSolverConfig = field(default_factory=GeneratorSolverConfig)
    fake_score: FakeSolverConfig = field(default_factory=FakeSolverConfig)
    aux_v: AuxSolverConfig = field(default_factory=AuxSolverConfig)


@dataclass
class TrainConfig:
    total_steps: int = 100000
    save_interval: int = 1000
    log_interval: int = 10
    seed: int = 42
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    autocast_dtype: str = "bf16"
    resume_from: str = ""
    use_ema: bool = False
    ema_decay: float = 0.9999
    ema_every: int = 1


@dataclass
class DistributedConfig:
    """FSDP/HSDP/DDP settings."""

    strategy: str = "fsdp"
    fsdp_sharding: str = "full_shard"
    ddp_find_unused_parameters: bool | None = None

    def __post_init__(self) -> None:
        self.strategy = self.strategy.lower()
        self.fsdp_sharding = self.fsdp_sharding.lower()
        if self.strategy == "hsdp":
            self.strategy = "fsdp"
            self.fsdp_sharding = "hybrid"
        if self.strategy not in {"fsdp", "ddp"}:
            raise ValueError(
                "distributed.strategy must be 'fsdp', 'hsdp', or 'ddp'; "
                f"got {self.strategy!r}."
            )


@dataclass
class EvalConfig:
    """Periodic SD3.5 image generation and reward evaluation."""

    enabled: bool = True
    eval_interval: int = 5000
    eval_prompt_source: str = "prompt_file"
    eval_prompt_path: str = "prompts.json"
    dataset: str = "pickscore"
    dataset_path: str = "dataset"
    reward_fn: dict = field(default_factory=lambda: {"pickscore": 1.0})
    reward_dataset_map: dict = field(default_factory=dict)
    reward_ckpt_path: str = ""
    eval_num_steps: int = 4
    eval_guidance_scale: float = 1.0
    eval_batch_size: int = 4
    num_media_images: int = 8
    eval_seed: int = 12345


@dataclass
class LoggingConfig:
    project: str = "fewsteprl"
    run_name: str = ""
    entity: str = ""
    tags: list = field(default_factory=list)
    enabled: bool = True


@dataclass
class MeanFlowNFTConfig:
    """Root configuration with the Stage 1/2 (``anyflow_*``) and
    ``meanflow_nft`` sections; :class:`FewStepRLConfig` replaces the latter
    with ``fewstep_rl``."""

    model: ModelConfig = field(default_factory=ModelConfig)
    flowmap_model: FlowMapModelConfig = field(default_factory=FlowMapModelConfig)
    anyflow_pretrain: AnyFlowPretrainConfig = field(default_factory=AnyFlowPretrainConfig)
    anyflow_onpolicy: AnyFlowOnPolicyConfig = field(default_factory=AnyFlowOnPolicyConfig)
    anyflow_dmd: AnyFlowDMDConfig = field(default_factory=AnyFlowDMDConfig)
    meanflow_nft: MeanFlowNFTRLConfig = field(default_factory=MeanFlowNFTRLConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    prompt_path: str = "prompts.json"
    output_dir: str = "./outputs"

    @classmethod
    def from_yaml(
        cls,
        path: str,
        overrides: dict[str, Any] | None = None,
    ) -> "MeanFlowNFTConfig":
        with open(path, "r") as handle:
            raw = yaml.safe_load(handle) or {}
        if overrides:
            raw = _apply_flat_overrides(raw, overrides)
        return _dataclass_from_dict(cls, raw)

    def to_dict(self) -> dict:
        return asdict(self)

    def save_yaml(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as handle:
            yaml.safe_dump(self.to_dict(), handle, sort_keys=False)


@dataclass
class FewStepRLConfig:
    """Root configuration loaded by ``main.py``; RL settings live in ``fewstep_rl``."""

    model: ModelConfig = field(default_factory=ModelConfig)
    flowmap_model: FlowMapModelConfig = field(default_factory=FlowMapModelConfig)
    anyflow_pretrain: AnyFlowPretrainConfig = field(default_factory=AnyFlowPretrainConfig)
    anyflow_onpolicy: AnyFlowOnPolicyConfig = field(default_factory=AnyFlowOnPolicyConfig)
    anyflow_dmd: AnyFlowDMDConfig = field(default_factory=AnyFlowDMDConfig)
    fewstep_rl: FewstepRLTrainConfig = field(default_factory=FewstepRLTrainConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    prompt_path: str = "prompts.json"
    output_dir: str = "./outputs"

    @classmethod
    def from_yaml(
        cls,
        path: str,
        overrides: dict[str, Any] | None = None,
    ) -> "FewStepRLConfig":
        with open(path, "r") as handle:
            raw = yaml.safe_load(handle) or {}
        if overrides:
            raw = _apply_flat_overrides(raw, overrides)
        return _dataclass_from_dict(cls, raw)

    def to_dict(self) -> dict:
        return asdict(self)

    def save_yaml(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as handle:
            yaml.safe_dump(self.to_dict(), handle, sort_keys=False)