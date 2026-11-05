"""Shared engine of the RL trainers for the few-step SD3.5 flow-map generator.

:class:`GeneralRLTrainer` holds everything the registered trainers have in
common; :class:`AnchorInputs` is the checked container handed to the anchor
hooks. The anchor-form policy loss, with ``t`` normalized to ``[0, 1]`` and
``u`` the flow map's average velocity over ``[r, t]``::

    x_0_pred_u = x_t - t * u_theta(x_t, t, r)                    # no S term
    x_anchor_1 = x_t - t * u_old(x_t, t, r)                      # S-free
    x_anchor_2 = x_t - t * v_inst                                # per-trainer probe center
    dev        = x_0_pred_u - x_anchor_1                         # probe direction
    x_0_target = x_anchor_1 + omega * (x_0_rwd - x_anchor_2)
    loss       = weight * MSE(x_0_pred_u, sg(x_0_target))

``omega`` and ``weight`` come from :meth:`GeneralRLTrainer.decoupled_weights`
(built from ``dev``, ``x_anchor_2``, the rewarded sample ``x_0_rwd`` and
``r_hat``, the clipped advantage mapped to ``[0, 1]``);
:meth:`GeneralRLTrainer.decoupled_rl_loss` evaluates the loss as two
separately logged terms with the same gradient.
``S = (t - r) * dF/dt`` lifts an average velocity to the instantaneous one
(``v = u + S``), where ``dF/dt`` is one central difference on ``old_model``
shared by every consumer in a step. S never enters ``x_0_pred_u`` or ``dev``,
only the detached ``x_anchor_2``, so no bf16 subtraction of two S-carrying
velocities has to cancel it.

Direction rule (load-bearing)
-----------------------------
``dev`` uses ``x_anchor_1``, never ``x_anchor_2``. With S added to both the
generator's and the old model's velocity, the probe offset carries S on both
sides and reduces exactly to ``x_0_pred_u - x_anchor_1``; using
``x_0_pred_u - x_anchor_2`` would add the spurious offset
``x_anchor_1 - x_anchor_2`` (``t * S`` for MeanFlowNFTTrainer). With ``S = 0``
(NFTTrainer) the two coincide; the aux-v-predictor anchor (DMDNFTTrainer)
keeps the same rule.

Subclass hooks
--------------
:meth:`~GeneralRLTrainer._instantaneous_velocity_from_u`,
:meth:`~GeneralRLTrainer._average_velocity_from_v`,
:meth:`~GeneralRLTrainer._compute_x_anchor_1`,
:meth:`~GeneralRLTrainer._compute_x_anchor_2` and the per-microbatch
:meth:`~GeneralRLTrainer._rl_loss` are abstract here. The essential choice is
the instantaneous velocity behind ``x_anchor_2``: the MeanFlow identity on
``old_model`` (MeanFlowNFTTrainer), the aux v-predictor (DMDNFTTrainer,
DMDAWMTrainer), or ``S = 0`` (the S-term ablations NFTTrainer, AWMTrainer,
FlowGRPOTrainer). The GRPO, AWM, OPSD and ReFL trainers fill the same anchor
slots with their own loss weighting, and some also override the sampling or
inner-update phase.

What the base provides
----------------------
SD3.5 + flow-map model setup (generator, EMA, ``old_model``, optional KL
reference and aux v-predictor, FSDP/DDP wrapping); the K-repeat prompt sampler
and ODE/SDE flow-map rollouts; reward scoring and per-prompt advantages; the
shared central difference; the training-time three-mode ``(t, r)`` sampler;
the decoupled loss and the KL term; the inner update loop with gradient
accumulation; generator EMA and ``old_model`` decay; the aux v-predictor
epoch; the epoch-based training loop and eval sampler; checkpoint hooks.
"""

from __future__ import annotations

import copy
import logging
import os
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.training_utils import compute_density_for_timestep_sampling
from torch.utils.data import DataLoader

from fewsteprl.config import FewStepRLConfig
from fewsteprl.data.prompt_dataset import PromptDataset
from fewsteprl.models.sd35 import encode_prompts_sd35, load_sd35_models, predict_velocity_sd35
from fewsteprl.models.sd35_flowmap import (
    predict_velocity_sd35_flowmap,
    setup_flowmap_for_sd3,
)
from fewsteprl.parallel.utils import (
    fsdp_wrap_model,
    get_rank,
    get_transformer_wrap_policy,
    get_world_size,
    is_main_process,
)
from fewsteprl.rewards.multi_scorer import MultiScorer
from fewsteprl.rewards.reward_ckpt_path import set_ckpt_path
from fewsteprl.schedulers.flowmap_scheduler import FlowMapScheduler
from fewsteprl.trainers.base_trainer import BaseTrainer
from fewsteprl.trainers.rl_utils import DistributedKRepeatSampler, PerPromptStatTracker
from fewsteprl.utils.lora import setup_lora

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnchorInputs:
    """Everything the anchor hooks may read, packed and checked once.

    The x0-space operands (:attr:`x_t`, :attr:`t_normed`, :attr:`u_old`,
    :attr:`u_theta`) arrive already converted: work dtype, detached, and
    ``t_normed`` broadcast to the latent rank. A hook can therefore write its
    algebra in plain names without being able to reach the network's bf16
    tensors or an unbroadcast ``[B]`` timestep. Both mistakes would be silent,
    which is why :meth:`__post_init__` checks for them:

    * a bf16 operand quantizes the S-free deviation to ~4e-3 relative, swamping
      the exact zero on the ``r == t`` slice that ``s_rel_median`` reads;
    * a ``[B]`` timestep right-aligns against the width axis instead of the
      batch axis, so it raises only when the two sizes differ and silently
      scales per column when they match.

    :attr:`u_theta` is the generator's own velocity but enters detached, so the
    whole container is stop-grad; ``_rl_loss`` builds the differentiable
    ``x_0_pred_u`` from the grad-carrying copy separately. The anchors must stay
    constant: a gradient path into ``x_anchor_1`` / ``x_anchor_2`` would break
    the S-free ``dev`` and its exact zero on the ``r == t`` slice.

    :attr:`prompt_embeds` / :attr:`pooled_embeds` are not checked: they feed the
    network and stay in the network's dtype.
    """

    x_t: torch.Tensor              #: noisy latent at t
    t_normed: torch.Tensor                #: t/T, broadcast to x_t's rank
    u_old: torch.Tensor            #: old_model average velocity u(x_t, r, t)
    u_theta: torch.Tensor           #: generator average velocity u(x_t, r, t), detached
    dF_dt: Optional[torch.Tensor]  #: shared central difference; None when it was skipped
    t_cd: torch.Tensor             #: raw timesteps, in the central difference's own units
    r_cd: Optional[torch.Tensor]   #: raw flow-map target timesteps, same units as t_cd
    prompt_embeds: torch.Tensor
    pooled_embeds: torch.Tensor

    def __post_init__(self) -> None:
        # Must match the field names declared above; a wrong name raises
        # AttributeError on every construction.
        for name in ("x_t", "t_normed", "u_old", "u_theta"):
            operand = getattr(self, name)
            if operand.dtype not in (torch.float32, torch.float64):
                raise TypeError(
                    f"AnchorInputs.{name} is {operand.dtype}; the anchor algebra "
                    "runs in x0 space and needs the fp32/fp64 copy, not the "
                    "network's dtype."
                )
            if operand.requires_grad:
                raise RuntimeError(
                    f"AnchorInputs.{name} carries grad; the anchors are stop-grad "
                    "reference points and only x_0_pred_u may be differentiable."
                )
        if self.t_normed.ndim != self.x_t.ndim:
            raise ValueError(
                f"AnchorInputs.t_normed has ndim {self.t_normed.ndim}, expected {self.x_t.ndim} "
                "to match x_t. Pass the broadcast timestep, not the [B] vector."
            )


class GeneralRLTrainer(BaseTrainer):
    """Shared RL engine (see the module docstring).

    Subclasses implement the velocity / anchor hooks and :meth:`_rl_loss`.
    """

    #: Per-trainer identifier, set by subclasses. Written into the checkpoint
    #: and checked on resume, because the saved config.yaml does not record the
    #: ``--trainer`` key.
    TRAINER_KEY: str = "rl_general"

    #: Working dtype of the x0-space algebra (anchors, ``dev``, loss). Must be
    #: fp32 or fp64: :class:`AnchorInputs` rejects anything else.
    _rl_work_dtype: torch.dtype = torch.float32

    #: Dtype the two forwards in :meth:`_central_diff_dudt` are cast to before
    #: they are subtracted (casting after the subtraction cannot recover the
    #: lost precision). The forwards are bf16 evaluations at inputs only
    #: ``eps_cd / T`` apart, so in bf16 the quotient has a large relative error
    #: and many elements round to exactly zero. This bounds how well any
    #: S-using trainer resolves S, so keep it the same across trainers being
    #: compared. ``None`` keeps the network dtype.
    _cd_out_dtype: Optional[torch.dtype] = None

    #: Seed offset that puts :meth:`_draw_training_s_given_t` on a different RNG
    #: stream from :meth:`_draw_training_tr`. Without it an ``s`` drawn at
    #: ``(inner_epoch, chunk, j)`` would get the same seed as the ``(t, r)`` draw
    #: at the same coordinates, correlating the two. Only that method passes it
    #: (``_make_training_tr_rng(stream_offset=...)``, default 0), so the
    #: ``(t, r)`` stream is unaffected; declared on the base because a base
    #: method reads it. The value is arbitrary but large and shares no small
    #: factors with the seed multipliers (1_000_003 / 10_007 / 101 / 17 / 3 /
    #: 7919), so the two streams do not realias.
    _S_RNG_STREAM_OFFSET: int = 2_654_435_761

    #: v-space KL weighting, read by :meth:`compute_kl_term` and
    #: :meth:`_v_kl_reweight` (hence declared on the base). When
    #: ``_kl_use_stage1_weighting`` is set, the per-sample KL gets AnyFlow
    #: Stage 1's two weighting layers: a t-weight from the ``_kl_weight_type``
    #: curve, then :meth:`_kl_mode_rebalance`. The rationale is documented on
    #: :class:`~fewsteprl.trainers.meanflownft.MeanFlowNFTTrainer`.
    _kl_use_stage1_weighting: bool = True
    _kl_weight_type: str = "beta08"

    #: Emitted every step on every rank as ``rl_diagnostic/*``. Declared as a
    #: class constant so the key set is identical across ranks and trainers:
    #: :meth:`BaseTrainer._reduce_metrics` all-reduces per key with no key-set
    #: reconciliation, so a rank- or step-dependent key set hangs until the NCCL
    #: watchdog fires instead of raising.
    _EXTRA_DIAGNOSTIC_KEYS: tuple[str, ...] = (
        "s_x0_abs_mean",        # |S_x0| = |x_anchor_1 - x_anchor_2|
        "s_rel_median",         # mean|S_x0| / mean|dev|
    )

    #: Extra ``upd`` keys that :meth:`_run_rl_epoch` copies into ``rl_loss/*``
    #: on top of its fixed list; its ``if k in upd`` filter silently drops any
    #: key not registered here, and skips registered keys a trainer does not
    #: emit. A class constant for the same reason as
    #: :attr:`_EXTRA_DIAGNOSTIC_KEYS`. The two split-loss values come from
    #: :meth:`decoupled_rl_loss` and are not comparable with a fused single-MSE
    #: loss value (they differ by an indefinite-sign cross term); the three KL
    #: entries come from :meth:`compute_kl_term`, which every trainer calls.
    _EXTRA_LOSS_KEYS: tuple[str, ...] = (
        "self_regression_loss",       # W * mean_sp[(P - x_anchor_1)^2]
        "reward_augmentation_loss",   # W * mean_sp[norm_grad^2]
        "loss_kl_unweighted",         # KL before the t-weight / mode rebalance
        "kl_t_weight_mean",           # mean KL t-weight (0.0 when inactive)
        "kl_rebalance_mean",          # mean mode-rebalance multiplier (1.0 when inactive)
    )

    #: Where the aux epoch's ``set_epoch`` stream starts. The policy path uses
    #: ``_rl_epoch * num_batches_per_epoch + i``; parking the aux stream above
    #: any reachable value of that keeps the two prompt streams from ever
    #: seeding the sampler identically.
    _AUX_SAMPLER_EPOCH_OFFSET: int = 1_000_000_000

    def __init__(self, config: FewStepRLConfig):
        super().__init__(config)
        self.model_type = str(self.config.model.model_type).lower()
        if self.model_type != "sd35":
            raise ValueError(
                "FewStepRL release only supports model_type='sd35'; "
                f"got {self.model_type!r}."
            )
        if not self.config.flowmap_model.enabled:
            raise ValueError(
                f"{type(self).__name__} requires flowmap_model.enabled=True."
            )

        # Models (populated in setup_models)
        self.generator: Optional[nn.Module] = None
        self.generator_ema: Optional[nn.Module] = None
        self.old_model: Optional[nn.Module] = None
        self.ref_u_model: Optional[nn.Module] = None  # only built if kl_weight > 0 and not use_ref_v_model
        self.text_encoders: list[nn.Module] = []
        self.tokenizers: list = []
        self.scheduler: Optional[FlowMatchEulerDiscreteScheduler] = None
        self.vae = None
        self.latent_channels: Optional[int] = None
        self.latent_size: Optional[int] = None
        self.ref_v_model: Optional[nn.Module] = None
        self.aux_v_model: Optional[nn.Module] = None
        
        # Flow-map scheduler is initialized before LoRA injection.
        self.flowmap_scheduler: Optional[FlowMapScheduler] = None

        # Shared central-difference cache. ``share_cd_with_old`` is mandatory
        # here (see :meth:`_pre_compute_v_hook`): the CD runs once per
        # :meth:`_rl_loss` call on ``self.old_model`` and lands in this slot,
        # which ``_rl_loss`` hands to the anchor hooks as ``AnchorInputs.dF_dt``.
        # Cleared by :meth:`_post_compute_v_hook`.
        self._shared_dF_dt: Optional[torch.Tensor] = None

        # Reward scorer (lazy-initialized on first sampling phase).
        self._reward_scorer: Optional[MultiScorer] = None
        self._stat_tracker: Optional[PerPromptStatTracker] = None
        self._uncond_embeds: Optional[tuple[torch.Tensor, torch.Tensor]] = None

        # Epoch-based outer loop state.
        self._rl_epoch: int = 0
        self._rl_dataloader: Optional[DataLoader] = None
        self._rl_sampler: Optional[DistributedKRepeatSampler] = None
        self._rl_sample_iter = None

        # Aux-epoch prompt stream. Separate iterator over the same dataloader:
        # sharing ``_rl_sample_iter`` would advance the policy path's prompt
        # stream by whatever the aux phase consumed. The round counter feeds
        # ``set_epoch`` so the two phases land on different K-repeat partitions.
        self._aux_sample_iter = None
        self._aux_sample_round: int = 0


    def _rl_cfg(self):
        return self.config.fewstep_rl
    
    def _kl_usage(self) -> tuple[bool, bool]:
        """``(use_u_kl, use_v_kl)`` -- the two KL modes are mutually exclusive."""
        cfg = self._rl_cfg()
        kl_weight = float(cfg.kl_weight)

        if self.ref_u_model is not None and self.ref_v_model is not None:
            raise RuntimeError(
                "ref_u_model and ref_v_model are both set; the two KL reference "
                "modes are mutually exclusive and setup_models picks exactly one "
                "via use_ref_v_model."
            )

        use_u_kl = kl_weight > 0.0 and self.ref_u_model is not None
        use_v_kl = kl_weight > 0.0 and self.ref_v_model is not None
        return use_u_kl, use_v_kl

    # ==================================================================
    # SD3.5 model setup
    # ==================================================================

    def setup_models(self) -> None:
        model_cfg = self.config.model
        dist_cfg = self.config.distributed

        logger.info("=" * 60)
        logger.info("Setting up RL models (trainer=%s)", self.TRAINER_KEY)
        logger.info("=" * 60)

        components = load_sd35_models(model_cfg)
        base_transformer = components["transformer"]

        self.text_encoders = components["text_encoders"]
        self.tokenizers = components["tokenizers"]
        self.scheduler = components["scheduler"]
        self.vae = components["vae"]

        if model_cfg.gradient_checkpointing:
            base_transformer.enable_gradient_checkpointing()

        self.generator = copy.deepcopy(base_transformer)

        # Keep the pristine transformer for ref_v_model / aux_v_model. This
        # hand-off cannot be deferred: the next lines apply generator_init_path,
        # the flow-map two-time wrapper and LoRA to ``self.generator``, after
        # which no plain v-predictor can be recovered from it.
        # ``self.generator`` is already a copy, so the original is handed over
        # rather than deepcopy'd again; when both consumers exist, aux_v takes
        # the copy. ``base_transformer`` already has gradient checkpointing on,
        # which aux_v wants (it backprops) and ref_v_model turns off below.
        rl_cfg = self._rl_cfg()
        need_ref_v = rl_cfg.kl_weight > 0.0 and rl_cfg.use_ref_v_model
        need_aux_v = bool(rl_cfg.aux_v_enabled)
        pretrained_v = base_transformer if need_ref_v else None
        if need_aux_v:
            aux_v = copy.deepcopy(base_transformer) if need_ref_v else base_transformer
        else:
            aux_v = None
        del base_transformer

        # Optional role-specific base-weight initialization.
        self._load_model_init_from_path(
            self.generator, model_cfg.generator_init_path, "generator",
        )

        self._setup_scheduler()

        if model_cfg.generator_lora.enabled:
            setup_lora(self.generator, model_cfg.generator_lora)
            logger.info(
                f"  Generator: LoRA injected (rank={model_cfg.generator_lora.rank})"
            )
        else:
            self.generator.train()
            logger.info("  Generator: full-weight trainable")

        self._post_setup_lora()

        # Freeze text encoders + VAE; move to GPU.
        for enc in self.text_encoders:
            enc.requires_grad_(False)
            enc.eval()
        self.vae.requires_grad_(False)
        self.vae.eval()
        device = torch.device("cuda")
        self.vae.to(device)
        for enc in self.text_encoders:
            enc.to(device)

        # SD3.5 latent geometry.
        vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.latent_channels = self.generator.config.in_channels
        latent_divisor = vae_scale_factor
        if model_cfg.image_resolution % latent_divisor != 0:
            raise ValueError(
                f"image_resolution={model_cfg.image_resolution} must be divisible by "
                f"the SD3.5 VAE scale factor {latent_divisor}."
            )
        self.latent_size = model_cfg.image_resolution // latent_divisor

        # EMA generator (optional). Must be deepcopy'd before the FSDP wrap.
        if self.config.train.use_ema:
            self.generator_ema = copy.deepcopy(self.generator)
            self.generator_ema.requires_grad_(False)
            self.generator_ema.eval()

        # old_model: frozen deepcopy of generator at training start, decays
        # slowly toward `generator` via EMA. Required for RL loss.
        self.old_model = copy.deepcopy(self.generator)
        self.old_model.requires_grad_(False)
        self.old_model.eval()

        # KL reference, frozen for the entire run. The two modes are mutually
        # exclusive, selected by ``use_ref_v_model``:
        #
        #   False -> ref_u_model (u-space): a snapshot of the generator as it
        #            stands now, i.e. with generator_init_path, the flow-map
        #            two-time wrapper and LoRA. KL is measured against the
        #            run's own starting point.
        #   True  -> ref_v_model (v-space): the pristine pretrained SD3.5
        #            transformer, i.e. the instantaneous-velocity predictor
        #            v(x_t, t). KL is measured against the pretrained prior.
        #            It cannot be built from ``self.generator`` at this point,
        #            hence the hand-off above.
        #
        # The unselected attribute stays None.
        if rl_cfg.kl_weight > 0.0:
            if rl_cfg.use_ref_v_model:
                self.ref_v_model = pretrained_v
                if model_cfg.gradient_checkpointing:
                    # Frozen and always forwarded under no_grad: checkpointing
                    # buys nothing here and only adds recompute.
                    self.ref_v_model.disable_gradient_checkpointing()
                ref = self.ref_v_model
                logger.info("  ref_v_model: pretrained SD3.5 v-predictor (frozen)")
            else:
                self.ref_u_model = copy.deepcopy(self.generator)
                ref = self.ref_u_model
                logger.info("  ref_u_model: generator snapshot (frozen)")
            ref.requires_grad_(False)
            ref.eval()

        # aux_v_model: a trainable pristine v-predictor, trained by
        # :meth:`_run_aux_epoch` with a plain flow-matching loss. Unlike
        # ref_v_model it keeps gradient checkpointing on and stays
        # requires_grad. It gets no flow-map wrapper and no LoRA: it answers
        # v(x_t, t), which has no ``r`` input at all.
        #
        # Assigned before _wrap_models so the FSDP loop wraps it and the
        # self.models registration below records it; checkpoint save/resume
        # then needs no extra code.
        if aux_v is not None:
            self.aux_v_model = aux_v
            # Optional init, mirroring generator_init_path. Loaded before
            # _wrap_models: afterwards the module is sharded and a plain
            # load_state_dict no longer addresses it.
            #
            # With an empty path aux_v keeps the pristine pretrained weights,
            # identical to ref_v_model's (both come from ``base_transformer``,
            # deepcopy'd apart so the trainable copy cannot write through to
            # the frozen one). ref_v_model is always the pretrained v(x_t, t)
            # and ignores model.teacher_init_path; if it ever honours that
            # field, this call has to follow the same path to keep the two
            # aligned.
            self._load_model_init_from_path(
                self.aux_v_model, model_cfg.aux_v_init_path, "aux_v",
                subdirs=("aux_v_transformer", "transformer"),
            )
            self.aux_v_model.requires_grad_(True)
            self.aux_v_model.train()
            logger.info(
                "  aux_v_model: SD3.5 v-predictor (trainable, init=%s)",
                model_cfg.aux_v_init_path
                or "pretrained (identical to teacher ref_v_model)",
            )

        self._wrap_models(dist_cfg)

        # Register models for checkpointing. The base trainer saves all
        # models in self.models with their respective LoRA-or-full path.
        self.models = {"generator": self.generator}
        if self.generator_ema is not None:
            self.models["generator_ema"] = self.generator_ema
        if self.old_model is not None:
            self.models["old_model"] = self.old_model
        if self.ref_u_model is not None:
            self.models["ref_u_model"] = self.ref_u_model
        if self.aux_v_model is not None:
            self.models["aux_v_model"] = self.aux_v_model

        # Per-prompt stat tracker (lazy-initialized).
        if self._rl_cfg().per_prompt_stat_tracking:
            self._stat_tracker = PerPromptStatTracker(
                global_std=self._rl_cfg().per_prompt_global_std,
            )

        logger.info("RL model setup complete")
        logger.info("=" * 60)

    def _setup_scheduler(self) -> None:
        """Install the flow-map (two-time) wrapper and build the FlowMap scheduler.

        Must run before LoRA is injected: ``delta_embedder`` has to exist
        before ``setup_lora`` freezes everything, so that resumed LoRA
        checkpoints' ``delta_embedder`` weights load correctly.
        """
        fm = self.config.flowmap_model
        setup_flowmap_for_sd3(
            self.generator, gate_value=fm.gate_value, deltatime_type=fm.deltatime_type,
        )

        # Build the FlowMap scheduler with SD3.5's static shift.
        sched_cfg = self.scheduler.config
        self.flowmap_scheduler = FlowMapScheduler(
            num_train_timesteps=int(sched_cfg.num_train_timesteps),
            shift=float(getattr(sched_cfg, "shift", 1.0)),
            weight_type="uniform",
        )

    def _post_setup_lora(self) -> None:
        """Re-enable delta_embedder gradients after setup_lora freezes everything."""
        if not self.config.model.generator_lora.enabled:
            return
        for name, param in self.generator.named_parameters():
            if "delta_embedder" in name:
                param.requires_grad = True

    def _wrap_models(self, dist_cfg) -> None:
        """Wrap generator + EMA + old_model (+ ref_* / aux_v_model) with FSDP/DDP.

        ``ref_v_model`` must be listed here even though it is frozen: it is
        forwarded on GPU every step the KL term is active, so leaving it out
        strands it unsharded on CPU. ``aux_v_model`` is the one non-generator
        module here that is trained, which is why the DDP branch wraps it
        instead of only moving it to the device.
        """
        strategy = dist_cfg.strategy
        fsdp_precision = self.config.model.dtype

        if strategy == "fsdp":
            try:
                from diffusers.models.transformers.transformer_sd3 import (
                    JointTransformerBlock,
                )
                wrap_policy = get_transformer_wrap_policy(JointTransformerBlock)
            except ImportError:
                wrap_policy = None
            for name in (
                "generator", "generator_ema", "old_model", "ref_u_model",
                "ref_v_model", "aux_v_model",
            ):
                mod = getattr(self, name)
                if mod is not None:
                    setattr(
                        self,
                        name,
                        fsdp_wrap_model(
                            mod,
                            sharding_strategy=dist_cfg.fsdp_sharding,
                            fsdp_precision=fsdp_precision,
                            auto_wrap_policy=wrap_policy,
                        ),
                    )
        elif strategy == "ddp":
            from fewsteprl.parallel.utils import ddp_wrap_model
            device = torch.device("cuda")
            has_lora = self.config.model.generator_lora.enabled
            ddp_find_unused_cfg = self.config.distributed.ddp_find_unused_parameters
            if ddp_find_unused_cfg is None:
                # Default heuristic: enable for LoRA.
                find_unused = has_lora
            else:
                find_unused = bool(ddp_find_unused_cfg)
            self.generator = ddp_wrap_model(
                self.generator.to(device), find_unused_parameters=find_unused,
            )
            if get_rank() == 0:
                logger.info(
                    "DDP config: find_unused_parameters=%s (lora_enabled=%s, user_override=%s)",
                    find_unused, has_lora, ddp_find_unused_cfg,
                )
            # aux_v_model is trained, so it needs a real DDP wrap: moving it to
            # the device alone would leave every rank optimizing its own private
            # copy with no gradient sync, and they would silently diverge. It
            # carries no LoRA, so nothing is left unused in its graph.
            if self.aux_v_model is not None:
                self.aux_v_model = ddp_wrap_model(
                    self.aux_v_model.to(device),
                    find_unused_parameters=(
                        False if ddp_find_unused_cfg is None
                        else bool(ddp_find_unused_cfg)
                    ),
                )
            for name in ("generator_ema", "old_model", "ref_u_model", "ref_v_model"):
                mod = getattr(self, name)
                if mod is not None:
                    setattr(self, name, mod.to(device))
        else:
            raise ValueError(f"Unknown distributed strategy: {strategy}")

    def setup_optimizers(self) -> None:
        gen_cfg = self.config.solver.generator
        gen_params = [p for p in self.generator.parameters() if p.requires_grad]
        self.optimizers["generator"] = torch.optim.AdamW(
            gen_params,
            lr=gen_cfg.lr,
            betas=(gen_cfg.beta1, gen_cfg.beta2),
            eps=gen_cfg.eps,
            weight_decay=gen_cfg.weight_decay,
        )
        self.schedulers["generator"] = self.create_warmup_constant_scheduler(
            self.optimizers["generator"], gen_cfg.warmup_steps,
        )
        logger.info(
            f"Optimizer created: generator (lr={gen_cfg.lr}, "
            f"trainable_params={sum(p.numel() for p in gen_params) / 1e6:.1f} M)"
        )

        if self.aux_v_model is not None:
            aux_cfg = self.config.solver.aux_v
            aux_params = [p for p in self.aux_v_model.parameters() if p.requires_grad]
            self.optimizers["aux_v"] = torch.optim.AdamW(
                aux_params,
                lr=aux_cfg.lr,
                betas=(aux_cfg.beta1, aux_cfg.beta2),
                eps=aux_cfg.eps,
                weight_decay=aux_cfg.weight_decay,
            )
            self.schedulers["aux_v"] = self.create_warmup_constant_scheduler(
                self.optimizers["aux_v"], aux_cfg.warmup_steps,
            )
            logger.info(
                f"Optimizer created: aux_v (lr={aux_cfg.lr}, "
                f"trainable_params={sum(p.numel() for p in aux_params) / 1e6:.1f} M)"
            )

    def _load_model_init_from_path(
        self,
        model: nn.Module,
        init_path: str,
        role_name: str,
        subdirs: tuple[str, ...] = ("transformer",),
    ) -> None:
        """Load role-specific init weights into ``model``, if a path is given.

        ``subdirs`` are the directory names probed, in order, when ``init_path``
        itself holds no ``config.json`` -- which is what lets a caller pass a
        previous run's ``output_dir`` rather than the transformer directory
        inside it. Roles that export under a different name (``aux_v`` writes
        ``aux_v_transformer``) pass their own name first.
        """
        if not init_path:
            return
        init_path = str(init_path).strip()
        if not init_path:
            return
        if os.path.isfile(init_path):
            state_dict = torch.load(init_path, map_location="cpu", weights_only=False)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            logger.info(
                f"  {role_name}: loaded init checkpoint {init_path} "
                f"(keys={len(state_dict)}, missing={len(missing)}, unexpected={len(unexpected)})"
            )
            return
        from fewsteprl.utils.fast_init import fast_init
        model_dir = init_path
        if not os.path.exists(os.path.join(model_dir, "config.json")):
            for sub in subdirs:
                alt = os.path.join(model_dir, sub)
                if os.path.exists(os.path.join(alt, "config.json")):
                    model_dir = alt
                    break
            else:
                raise FileNotFoundError(
                    f"{role_name}_init_path has no config.json at {init_path}, "
                    f"nor in any of {list(subdirs)}"
                )
        with fast_init(torch.device("cpu")):
            loaded = model.__class__.from_pretrained(
                model_dir, torch_dtype=next(model.parameters()).dtype,
            )
        sd = loaded.state_dict()
        del loaded
        missing, unexpected = model.load_state_dict(sd, strict=False)
        logger.info(
            f"  {role_name}: loaded init transformer dir {model_dir} "
            f"(keys={len(sd)}, missing={len(missing)}, unexpected={len(unexpected)})"
        )

    # ==================================================================
    # Dataloader
    # ==================================================================

    def _create_dataloader(self):
        """Not used -- this trainer overrides :meth:`_train_loop`.

        Returns ``(None, None)`` to satisfy :class:`BaseTrainer`'s abstract
        interface. The actual K-repeat dataloader is constructed in
        :meth:`_setup_rl_dataloader`, called from ``_train_loop``.
        """
        return None, None

    def _setup_rl_dataloader(self) -> None:
        """Build the K-repeat prompt dataloader (aligned with DiffusionNFT).

        Each rank gets ``train.batch_size`` prompt indices per sampling
        iteration; globally each call draws ``M = train.batch_size *
        world_size / K`` unique prompts and K-repeats them, then shards
        across ranks. K samples per prompt are scattered across ranks and
        re-grouped via cross-rank ``all_gather`` in :meth:`_compute_advantages`.
        """
        cfg = self._rl_cfg()
        K = int(cfg.num_image_per_prompt)
        per_rank_bs = int(self.config.train.batch_size)
        ws = get_world_size()
        if (per_rank_bs * ws) % K != 0:
            raise ValueError(
                f"train.batch_size * world_size ({per_rank_bs * ws}) must be divisible "
                f"by fewstep_rl.num_image_per_prompt={K} for K-repeat sampling."
            )

        dataset = PromptDataset(self.config.prompt_path, repeat=1)
        self._rl_sampler = DistributedKRepeatSampler(
            dataset_size=len(dataset),
            batch_size=per_rank_bs,
            k=K,
            num_replicas=ws,
            rank=get_rank(),
            seed=self.config.train.seed,
        )
        self._rl_dataloader = DataLoader(
            dataset,
            batch_sampler=self._rl_sampler,
            num_workers=0,
            collate_fn=lambda items: list(items),
        )
        self._rl_sampler.set_epoch(
            self._rl_epoch * int(cfg.num_batches_per_epoch)
        )
        self._rl_sample_iter = iter(self._rl_dataloader)
        if is_main_process():
            logger.info(
                f"RL dataloader: dataset_size={len(dataset)}, K={K}, "
                f"per_rank_bs={per_rank_bs}, num_batches_per_epoch="
                f"{cfg.num_batches_per_epoch}"
            )

    # ==================================================================
    # SD3.5 / flow-map helpers
    # ==================================================================

    def _predict_average_velocity(
        self,
        model: nn.Module,
        noisy_latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        timesteps: torch.Tensor,
        pooled_embeds: torch.Tensor,
        r_timesteps: torch.Tensor,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        return predict_velocity_sd35_flowmap(
            model=model,
            noisy_latents=noisy_latents,
            text_embeddings=prompt_embeds,
            timesteps=timesteps,
            pooled_prompt_embeds=pooled_embeds,
            r_timesteps=r_timesteps,
            guidance_scale=guidance_scale,
        )
    
    def _predict_instantaneous_velocity(
        self,
        model: nn.Module,
        noisy_latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        timesteps: torch.Tensor,
        pooled_embeds: torch.Tensor,
        r_timesteps: torch.Tensor,
        guidance_scale: float = 1.0,
        model_u_pred: bool = False,
        dF_dt: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Instantaneous velocity ``v(x_t, t)`` from either kind of predictor.

        ``model_u_pred`` says what ``model`` emits, not what the caller wants:

        * ``False`` -- a one-time v-predictor (the pristine pretrained SD3.5
          transformer). It has no ``r`` input, so ``r_timesteps`` is deliberately
          not forwarded; passing it is a TypeError, not a no-op.
        * ``True`` -- a two-time flow map. It is queried at the ``r`` the time
          sampler drew, not at a hardcoded ``r == t``: pinning ``r = t`` here
          would silently override the sampler's schedule. What comes back is
          therefore an average velocity over ``[r, t]``, lifted to the
          instantaneous velocity with ``dF_dt``. The correction is exactly zero
          on whichever samples the sampler happened to draw with ``r == t``, so
          that case needs no special handling -- and ``dF_dt=None`` is accepted
          only when the sampler drew ``r == t`` everywhere.
        """
        if model_u_pred:
            u = self._compute_u(
                model=model,
                x_t=noisy_latents,
                t_raw=timesteps,
                prompt_embeds=prompt_embeds,
                pooled_embeds=pooled_embeds,
                r_raw=r_timesteps,
                with_grad=False,
            )
            # Lifted in the x0 work dtype, not the network's: the subclasses'
            # _s_term rejects a bf16 operand, because folding S in bf16
            # destroys the cancellation the anchors depend on.
            return self._instantaneous_velocity_from_u(
                u_old=u.to(self._rl_work_dtype),
                t_cd=timesteps,
                r_cd=r_timesteps,
                dF_dt=dF_dt,
            )

        uncond_embeds = uncond_pooled = None
        if guidance_scale > 1.0:
            # predict_velocity_sd35 raises without the unconditional
            # embeddings once CFG is on.
            uncond_embeds, uncond_pooled = self._get_uncond_embeds(
                noisy_latents.shape[0]
            )
            uncond_embeds = uncond_embeds.to(prompt_embeds.dtype)
            uncond_pooled = uncond_pooled.to(pooled_embeds.dtype)
        return predict_velocity_sd35(
            model=model,
            noisy_latents=noisy_latents,
            text_embeddings=prompt_embeds,
            timesteps=timesteps,
            pooled_prompt_embeds=pooled_embeds,
            guidance_scale=guidance_scale,
            uncond_text_embeddings=uncond_embeds,
            uncond_pooled_prompt_embeds=uncond_pooled,
        )

    def _encode_prompts(self, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        device = torch.device("cuda")
        return encode_prompts_sd35(
            prompts,
            self.text_encoders,
            self.tokenizers,
            device,
            max_sequence_length=int(self._rl_cfg().text_max_sequence_length),
        )

    @torch.no_grad()
    def _get_uncond_embeds(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self._uncond_embeds is None:
            device = torch.device("cuda")
            self._uncond_embeds = encode_prompts_sd35(
                [""],
                self.text_encoders,
                self.tokenizers,
                device,
                max_sequence_length=int(self._rl_cfg().text_max_sequence_length),
            )
        emb, pool = self._uncond_embeds
        return emb.expand(batch_size, -1, -1), pool.expand(batch_size, -1)

    # ==================================================================
    # Sampling rollouts: N-step flow-map Euler (ODE) and overshoot-renoise (SDE).
    # ==================================================================

    def _get_last_rollout_timestep_pairs(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(t_arr, r_arr)`` from the last flow-map rollout.

        ``flowmap_scheduler.timesteps`` has length ``N + 1`` after the
        last rollout (``[t_0, t_1, ..., t_N=0]``); the actual flow-map
        jumps the rollout performed are ``(t_i, t_{i+1})`` for ``i in [0, N)``.
        Return those ``N`` pairs as ``t_arr = timesteps[:-1]`` and
        ``r_arr = timesteps[1:]`` -- both length ``N``.
        """
        ts = self.flowmap_scheduler.timesteps.detach().to(dtype=torch.float32)
        return ts[:-1], ts[1:]

    @torch.no_grad()
    def _rollout_samples_ode(
        self,
        prompt_embeds: torch.Tensor,
        pooled_embeds: torch.Tensor,
        *,
        num_steps: int,
        guidance_scale: float,
        model: nn.Module,
        return_intermediate: bool = False,
    ) -> tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """N-step flow-map Euler rollout (one transformer fwd per step).

        Also the eval-time sampler (see :meth:`_generate_latents`). The
        flow-map rollout is CFG-free (guidance_scale=1.0); the
        ``guidance_scale`` arg is accepted for signature compatibility but
        ignored.

        Returns ``(x_0, x_r_all, x_t_all, end_ts, mid_ts, start_ts)``. When
        ``return_intermediate`` the trajectory tensors are
        ``[B, num_steps, C, H, W]`` and the three timestep tensors are
        ``[B, num_steps]``; otherwise all five are None. ``mid_ts`` is the
        flow map's jump target, which this ODE path always takes to be the
        step's end point -- it exists so that the SDE rollout, where the two
        differ, can share this return contract.
        """
        del guidance_scale  # Flow-map inference is CFG-free.
        device = prompt_embeds.device
        dtype = next(model.parameters()).dtype
        batch_size = prompt_embeds.shape[0]

        self.flowmap_scheduler.set_timesteps(num_steps, device=device)
        timesteps = self.flowmap_scheduler.timesteps  # [N+1]

        x_r_list: list[torch.Tensor] = []
        x_t_list: list[torch.Tensor] = []
        end_timestep_list: list[torch.Tensor] = []
        mid_timestep_list: list[torch.Tensor] = []
        start_timestep_list: list[torch.Tensor] = []

        x_t = torch.randn(
            batch_size, self.latent_channels, self.latent_size, self.latent_size,
            device=device, dtype=dtype,
        )
        for i in range(num_steps):
            if return_intermediate:
                x_t_list.append(x_t.clone())
            t = timesteps[i].expand(batch_size).to(device=device, dtype=torch.float32)
            r = timesteps[i + 1].expand(batch_size).to(device=device, dtype=torch.float32)
            end_timestep_list.append(r)
            # s == r on this path: the ODE step lands exactly on its end point.
            mid_timestep_list.append(r)
            start_timestep_list.append(t)
            with self._autocast():
                u = self._predict_average_velocity(
                    model, x_t, prompt_embeds, t, pooled_embeds, r,
                    guidance_scale=1.0,
                )
            x_r = self.flowmap_scheduler.step(u, x_t, t, r)
            if return_intermediate:
                x_r_list.append(x_r.clone())
            x_t = x_r
        if not return_intermediate:
            return x_t, None, None, None, None, None
        # dim=1 on every stack: the sample axis has to stay leading so that the
        # cat over sampling rounds in _rl_sampling_phase concatenates samples.
        return (
            x_t,
            torch.stack(x_r_list, dim=1),
            torch.stack(x_t_list, dim=1),
            torch.stack(end_timestep_list, dim=1),
            torch.stack(mid_timestep_list, dim=1),
            torch.stack(start_timestep_list, dim=1),
        )


    @torch.no_grad()
    def _rollout_samples_sde(
        self,
        prompt_embeds: torch.Tensor,
        pooled_embeds: torch.Tensor,
        *,
        num_steps: int,
        guidance_scale: float,
        model: nn.Module,
        return_intermediate: bool = False,
    ) -> tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """N-step stochastic (overshoot + renoise) flow-map rollout.

        Same grid as :meth:`_rollout_samples_ode`, but each step jumps past its
        target: the flow map goes from ``t_i`` to a randomly drawn
        ``s <= t_{i+1}``, then the sample is renoised back up to ``t_{i+1}``.
        With ``s == t_{i+1}`` on every step this degenerates to the ODE
        rollout. The flow-map rollout is CFG-free (guidance_scale=1.0); the
        ``guidance_scale`` arg is accepted for signature compatibility but
        ignored.

        Returns ``(x_0, x_r_all, x_t_all, end_ts, mid_ts, start_ts)`` -- the
        same contract as :meth:`_rollout_samples_ode`. ``mid_ts`` is the drawn
        ``s``, which is both the point the model was queried at and the point
        the renoise starts from, so a consumer replaying a step needs all three
        of ``(start, mid, end)``; it is not recoverable from the grid.
        """
        del guidance_scale  # Flow-map inference is CFG-free.
        device = prompt_embeds.device
        dtype = next(model.parameters()).dtype
        batch_size = prompt_embeds.shape[0]

        self.flowmap_scheduler.set_timesteps(num_steps, device=device)
        timesteps = self.flowmap_scheduler.timesteps.to(device)  # [N+1], raw units
        T = float(self.flowmap_scheduler.config.num_train_timesteps)

        x_r_list: list[torch.Tensor] = []
        x_t_list: list[torch.Tensor] = []
        end_timestep_list: list[torch.Tensor] = []
        mid_timestep_list: list[torch.Tensor] = []
        start_timestep_list: list[torch.Tensor] = []
        x_t = torch.randn(
            batch_size, self.latent_channels, self.latent_size, self.latent_size,
            device=device, dtype=dtype,
        )
        for i in range(num_steps):
            if return_intermediate:
                x_t_list.append(x_t.clone())
            t = timesteps[i].expand(batch_size).to(device=device, dtype=torch.float32)
            r = timesteps[i + 1].expand(batch_size).to(device=device, dtype=torch.float32)
            end_timestep_list.append(r)
            start_timestep_list.append(t)
            # Overshoot target: one s per sample, drawn uniformly from
            # ``timesteps[i + 1 : ]`` (i.e. s <= r <= t; the last step can only
            # draw s == r == 0, which makes the renoise below the identity).
            s_idx = torch.randint(i + 1, num_steps + 1, (batch_size,), device=device)
            s = timesteps[s_idx].to(dtype=torch.float32)  # [B]
            mid_timestep_list.append(s)
            with self._autocast():
                u = self._predict_average_velocity(
                    model, x_t, prompt_embeds, t, pooled_embeds, s,
                    guidance_scale=1.0,
                )
            x_s = self.flowmap_scheduler.step(u, x_t, t, s)  # [B, C, H, W]
            # Renoise x_s back up from s to r on the linear path
            # x_a = (1 - a) * x_0 + a * eps, in normalized time a = raw / T.
            # Matching the x_0 coefficient gives alpha, matching the noise
            # variance gives tau. Both are [B, 1, 1, 1] so they broadcast
            # against [B, C, H, W].
            r01 = (r / T).view(batch_size, 1, 1, 1)
            s01 = (s / T).view(batch_size, 1, 1, 1)
            alpha_r = (1.0 - r01) / (1.0 - s01)
            tau_r = torch.sqrt(torch.clamp(r01**2 - (alpha_r * s01) ** 2, min=0.0))
            eps = torch.randn(x_s.shape, device=device, dtype=torch.float32)
            x_r = (alpha_r * x_s.float() + tau_r * eps).to(dtype)
            if return_intermediate:
                x_r_list.append(x_r.clone())
            x_t = x_r
        if not return_intermediate:
            return x_t, None, None, None, None, None
        # dim=1 on every stack: the sample axis has to stay leading so that the
        # cat over sampling rounds in _rl_sampling_phase concatenates samples.
        return (
            x_t,
            torch.stack(x_r_list, dim=1),
            torch.stack(x_t_list, dim=1),
            torch.stack(end_timestep_list, dim=1),
            torch.stack(mid_timestep_list, dim=1),
            torch.stack(start_timestep_list, dim=1),
        )

    @torch.no_grad()
    def _generate_latents(
        self,
        prompt_embeds: torch.Tensor,
        pooled_embeds: torch.Tensor,
        uncond_embeds: torch.Tensor,  # noqa: ARG002 - signature compatibility w/ BaseTrainer._evaluate
        uncond_pooled: torch.Tensor,  # noqa: ARG002 - same as above
        num_steps: int = 4,
        gradient_truncation: bool = False,  # noqa: ARG002 - eval path is no_grad
        random_stop: bool = False,  # noqa: ARG002 - eval uses fixed num_steps
        model: Optional[nn.Module] = None,
    ) -> tuple[torch.Tensor, int]:
        """Eval-side adapter over :meth:`_rollout_samples_ode`.

        The signature matches :meth:`BaseTrainer._evaluate`.
        """
        if model is None:
            model = self.generator_ema if self.generator_ema is not None else self.generator
        x0 = self._rollout_samples_ode(
            prompt_embeds=prompt_embeds,
            pooled_embeds=pooled_embeds,
            num_steps=int(num_steps),
            guidance_scale=float(self.config.eval.eval_guidance_scale),
            model=model,
        )[0]
        return x0, int(num_steps)

    # ==================================================================
    # dF/dt estimator: central difference (no-grad)
    # ==================================================================

    def _central_diff_dudt(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        v_dir: torch.Tensor,
        t_cd: torch.Tensor,
        r_cd: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Compute ``(F(z+, t+eps_cd, r) - F(z-, t-eps_cd, r)) / (2 * eps_cd)``.

        Same step formula and epsilon scaling as the AnyFlow Stage-1 central
        difference: perturbations to ``x_t`` are in [0,1] units (eps_cd / T),
        perturbations to ``t`` are in raw timestep units (eps_cd), and the
        denominator is ``2 * eps_cd``.

        Always invoked under :func:`torch.no_grad`.
        """
        T = float(self.scheduler.config.num_train_timesteps)
        eps_cd = float(self._rl_cfg().central_diff_epsilon)

        t_plus = t_cd + eps_cd
        t_minus = t_cd - eps_cd
        z_plus = x_t + v_dir * (eps_cd / T)
        z_minus = x_t - v_dir * (eps_cd / T)

        with self._autocast():
            f_plus = self._predict_average_velocity(
                model, z_plus, prompt_embeds, t_plus, pooled_embeds, r_cd,
                guidance_scale=1.0,
            )
            f_minus = self._predict_average_velocity(
                model, z_minus, prompt_embeds, t_minus, pooled_embeds, r_cd,
                guidance_scale=1.0,
            )
        # Promote before the subtraction -- see the ``_cd_out_dtype`` docstring.
        out_dtype = self._cd_out_dtype
        if out_dtype is not None:
            f_plus, f_minus = f_plus.to(out_dtype), f_minus.to(out_dtype)
        return (f_plus - f_minus) / (2.0 * eps_cd)

    @torch.no_grad()
    def _compute_dF_dt_for_model(
        self,
        *,
        cd_model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        r_cd: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_embeds: torch.Tensor,
        x_0: torch.Tensor,
    ) -> torch.Tensor:
        """Run one central-difference dF/dt on ``cd_model``.

        ``v_dir`` is selected by ``fewstep_rl.cd_velocity_source``:

          - ``"noise_minus_x0"``: recover ``eps`` from (x_t, x_0, t) and use
            ``v_dir = eps - x_0`` (MeanFlow baseline; model-independent).
          - ``"u_self"``: ``v_dir = cd_model(x_t, t, t, c).detach()``
            (iMF arXiv:2512.02012 Algorithm 2 bare-cond tangent; depends on
            ``cd_model``). No reverse-CFG fusion: the loss supervises ``f_c``
            directly, so the tangent must be ``f_c(z, t, t) = v_cfg``.

        Always runs inside :func:`torch.no_grad`.
        """
        cfg = self._rl_cfg()
        T = float(self.scheduler.config.num_train_timesteps)

        # The branch strings must match the values FewstepRLTrainConfig
        # validates ("noise_minus_x0", the default, and "u_self").
        if cfg.cd_velocity_source == "noise_minus_x0":
            # Recover eps from x_t / x_0 / t (deterministic;
            # eps = (x_t - (1 - t01) * x_0) / t01).
            t01 = (t.to(x_0.dtype) / T).view(x_0.shape[0], *([1] * (x_0.ndim - 1)))
            eps = (x_t - (1.0 - t01) * x_0) / t01.clamp(min=1.0 / T)
            v_dir = eps - x_0
        elif cfg.cd_velocity_source == "u_self":
            with self._autocast():
                u_dir = self._predict_average_velocity(
                    cd_model, x_t, prompt_embeds, t, pooled_embeds, t,  # r = t
                    guidance_scale=1.0,
                )
            v_dir = u_dir.to(x_t.dtype)
        else:
            raise ValueError(
                f"Unknown fewstep_rl.cd_velocity_source={cfg.cd_velocity_source!r}; "
                "expected 'noise_minus_x0' or 'u_self'."
            )

        return self._central_diff_dudt(
            model=cd_model,
            x_t=x_t.detach(),
            v_dir=v_dir.detach(),
            t_cd=t,
            r_cd=r_cd,
            prompt_embeds=prompt_embeds,
            pooled_embeds=pooled_embeds,
        )

    # ------------------------------------------------------------------
    # dF/dt precompute for one ``_rl_loss`` call.
    # ------------------------------------------------------------------
    def _pre_compute_v_hook(
        self,
        *,
        x_t: torch.Tensor,
        t_for_model: torch.Tensor,
        r_for_model: Optional[torch.Tensor],
        prompt_embeds: torch.Tensor,
        pooled_embeds: torch.Tensor,
        x_0: torch.Tensor,
    ) -> None:
        """Populate ``self._shared_dF_dt`` for this step.

        The CD is skipped only on config-level conditions
        (``diffusion_ratio >= 1``, ``disable_cd``) or when there is no
        ``old_model`` / ``r``, all of which are identical on every rank.

        Collective safety
        -----------------
        The central difference runs two forwards through an FSDP-wrapped model,
        i.e. it issues collectives, so every rank must make the same decision
        about whether to run it. Skipping on a data-dependent predicate such as
        ``torch.equal(t_for_model, r_for_model)`` deadlocks: the three-mode
        sampler partitions by global index, so with the shipped configs
        (``diffusion_ratio=0.5``, ``consistency_ratio=0.25``, 16 x 9 = 144
        samples per rank, 8 ranks) ranks 0-3 see only ``r == t`` (skip) while
        ranks 4-7 see ``r == 0`` or generic pairs (run). Ranks 0-3 then reach
        the ``dist.all_reduce`` in ``_compute_grad_norm`` while ranks 4-7 are
        still inside the CD's collectives, and the job hangs until the NCCL
        watchdog fires.

        On the ``r == t`` slice the S term still vanishes arithmetically in
        :meth:`_instantaneous_velocity_from_u`, since ``(t - r) == 0``.
        """
        cfg = self._rl_cfg()
        if not cfg.share_cd_with_old:
            raise ValueError(
                f"{type(self).__name__} requires share_cd_with_old=True: the x0 "
                "form assumes ONE shared dF/dt so that x_anchor_1 is exactly "
                "S-free; a per-network CD makes S_theta != S_old."
            )
        # Config-level, hence rank-uniform: every sample has r == t.
        if float(cfg.diffusion_ratio) >= 1.0 - 1e-8:
            return
        # OPSD-only opt-out, also config-level and therefore rank-uniform.
        # Unlike the branch above this leaves r random, so a plain ``_s_term``
        # would hit its ``r != t`` guard on the None dF_dt; MeanFlowOPSDTrainer
        # and MeanFlowReFLTrainer check the flag in their ``_s_term`` and treat
        # S as zero instead. See FewstepRLTrainConfig.disable_cd.
        if bool(getattr(cfg, "disable_cd", False)):
            return
        if self.old_model is None or r_for_model is None:
            return
        self._shared_dF_dt = self._compute_dF_dt_for_model(
            cd_model=self.old_model,
            x_t=x_t, t=t_for_model, r_cd=r_for_model,
            prompt_embeds=prompt_embeds, pooled_embeds=pooled_embeds, x_0=x_0,
        )

    def _post_compute_v_hook(self) -> None:
        self._shared_dF_dt = None

    # ==================================================================
    # Velocity queries
    # ==================================================================

    def _compute_u(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t_raw: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_embeds: torch.Tensor,
        r_raw: torch.Tensor,
        *,
        with_grad: bool,
    ) -> torch.Tensor:
        """Flow-map average velocity ``u(x_t, r, t)``."""
        ctx = nullcontext() if with_grad else torch.no_grad()
        with ctx:
            with self._autocast():
                return self._predict_average_velocity(
                    model, x_t, prompt_embeds, t_raw, pooled_embeds, r_raw,
                    guidance_scale=1.0,
                )

    def _compute_v(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t_raw: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_embeds: torch.Tensor,
        r_raw: torch.Tensor,
        guidance_scale: float=1.0,
        *,
        with_grad: bool,
        model_u_pred: bool=False,
        dF_dt: Optional[torch.Tensor]=None,
    ) -> torch.Tensor:
        """Instantaneous velocity ``v(x_t, t)``.

        ``dF_dt`` is only read when ``model_u_pred`` is True, where it lifts the
        flow map's average velocity over the sampler's ``[r, t]`` to the
        instantaneous one.
        """
        ctx = nullcontext() if with_grad else torch.no_grad()
        with ctx:
            with self._autocast():
                return self._predict_instantaneous_velocity(
                    model, x_t, prompt_embeds, t_raw, pooled_embeds, r_raw,
                    guidance_scale=guidance_scale, 
                    model_u_pred=model_u_pred,
                    dF_dt=dF_dt,
                )

    # ------------------------------------------------------------------
    # Velocity conversion and anchor hooks, implemented by subclasses.
    # ------------------------------------------------------------------
    def _instantaneous_velocity_from_u(
        self,
        u_old: torch.Tensor,
        t_cd: torch.Tensor,
        r_cd: torch.Tensor,
        dF_dt: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Lift an average velocity ``u(x_t, r, t)`` to the instantaneous ``v(x_t, t)``.

        Takes loose tensors rather than an :class:`AnchorInputs` because it is
        also called outside the anchor block: by
        :meth:`_predict_instantaneous_velocity`, which has no anchors, and on
        the grad-carrying ``u_theta`` by trainers whose x0 prediction is built
        from the instantaneous velocity.

        Contract:
          * returns ``u_old``'s dtype, which must be fp32/fp64: S must not be
            folded in the network's bf16;
          * shaped like ``u_old``; propagates grad iff ``u_old`` does, so passing
            the generator's own u yields a differentiable v;
          * ``t_cd`` / ``r_cd`` are raw timesteps, matching the central
            difference's own denominator, so ``(t - r) * dF_dt`` is dimensionless;
          * ``dF_dt`` is the shared central difference computed once on
            ``old_model`` (``share_cd_with_old``), or ``None`` when the CD was
            skipped (every sample has ``r == t``, or the trainer defines S as 0).
        """
        raise NotImplementedError
    
    def _average_velocity_from_v(
        self,
        v_inst: torch.Tensor,
        t_cd: torch.Tensor,
        r_cd: torch.Tensor,
        dF_dt: Optional[torch.Tensor]=None,
    ) -> torch.Tensor:
        """Inverse of :meth:`_instantaneous_velocity_from_u` (``u = v - S``), same contract."""
        raise NotImplementedError("Not implemented")


    @torch.no_grad()
    def _compute_x_anchor_2(self, anchors: AnchorInputs) -> torch.Tensor:
        """Detached probe center ``x_anchor_2`` (see the module docstring)."""
        raise NotImplementedError

    @torch.no_grad()
    def _compute_x_anchor_1(self, anchors: AnchorInputs) -> torch.Tensor:
        """Detached self-regression anchor ``x_anchor_1``; None if the trainer has no such term."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # KL helper and computation
    # ------------------------------------------------------------------

    def compute_kl_term(
        self,
        x_t: torch.Tensor,
        t_for_model: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_embeds: torch.Tensor,
        r_for_model: Optional[torch.Tensor],
        shared_dF_dt: Optional[torch.Tensor],
        u_theta_f: torch.Tensor,
        spatial: tuple[int, ...],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """KL against the frozen reference: ``(kl_scalar, info)``.

        Returns the diagnostics alongside the scalar rather than logging them
        here, because the keys have to reach ``info`` in :meth:`_rl_loss` for
        every trainer -- see the pre-seeded defaults below.
        """
        # Pre-seeded with the inactive values so the key set is identical on
        # every rank, trainer and KL mode. ``_reduce_metrics`` all-reduces per
        # key with no key-set reconciliation, so a key that appears only in
        # v-mode (or only on ranks that happen to hold a given slice) hangs the
        # job until the NCCL watchdog fires instead of raising. u-mode and the
        # KL-off case therefore report 0.0 / 1.0 rather than nothing.
        info: dict[str, float] = {
            "loss_kl_unweighted": 0.0,
            "kl_t_weight_mean": 0.0,
            "kl_rebalance_mean": 1.0,
        }

        # KL against the frozen reference. Which space the divergence lives in
        # follows from what the reference predicts, and the generator's own
        # output must be moved into that space rather than the other way round:
        #
        #   ref_u_model -- a snapshot of this generator, so it emits the same
        #       average velocity u(x_t, r, t). Both sides are already u, and
        #       (t-r)*dF/dt would appear on both sides of a v-space MSE and
        #       cancel in exact arithmetic. So plain MSE on u is the intended
        #       objective, and it avoids the bf16 residue that folding S into
        #       both sides would leave behind.
        #
        #   ref_v_model -- the pristine pretrained SD3.5 transformer, which emits
        #       the instantaneous velocity v(x_t, t) and has no r input at all.
        #       u and v are not comparable, so this mode is a MeanFlow loss: the
        #       reference's v is pushed down to an average-velocity target,
        #       u_tgt = v_ref - (t - r) * dF/dt, and u_theta regresses onto it.
        #       Note (t - r), not (r - t): the identity is
        #       u(x_t, r, t) = v(x_t, t) - (t - r) * du/dt.
        #       Whether (t - r) is zero is the time sampler's business, not this
        #       block's -- the correction simply vanishes on the samples where
        #       the sampler drew r == t.
        #       Equivalent in exact arithmetic to lifting u_theta up to v space
        #       ((u + S) - v_ref == u - (v_ref - S)), but putting S on the
        #       detached target keeps it out of the differentiable path entirely.

        # Per-sample, not yet reduced: the Stage 1 weighting below is
        # per-sample, so the .mean() has to come after it.
        cfg = self._rl_cfg()
        wdt = self._rl_work_dtype

        kl_ps: Optional[torch.Tensor] = None
        use_u_kl, use_v_kl = self._kl_usage()
        if use_u_kl:
            u_ref = self._compute_u(
                self.ref_u_model, x_t, t_for_model, prompt_embeds, pooled_embeds,
                r_for_model, with_grad=False,
            ).detach()
            kl_ps = self.u_kl_loss(u_theta_f, u_ref.to(wdt), spatial)
        elif use_v_kl:
            # dF_dt is unread while model_u_pred is False, but passing it keeps
            # this call correct if the reference is ever swapped for a flow map.
            v_ref = self._compute_v(
                self.ref_v_model, x_t, t_for_model, prompt_embeds, pooled_embeds,
                r_for_model, guidance_scale=float(cfg.guidance_scale),
                with_grad=False, model_u_pred=False, dF_dt=shared_dF_dt,
            ).detach()
            # r_for_model is None means the run is not using the two-time flow
            # map, so u is already instantaneous; feeding r == t takes the
            # S == 0 branch instead of tripping _s_term's None compare.
            r_cd = r_for_model if r_for_model is not None else t_for_model
            # MeanFlow target, fully detached: v_ref is under no_grad and dF/dt is
            # an old_model constant, so the whole correction sits on the target
            # side and the gradient reaching u_theta is just 2*(u_theta - target).
            u_target = self._average_velocity_from_v(
                v_ref.to(wdt), t_for_model, r_cd, shared_dF_dt,
            )
            kl_ps = self.u_kl_loss(u_theta_f, u_target, spatial)
            # Captured before weighting. Weighted and unweighted KL are on
            # different scales; do not compare one with the other.
            kl_unweighted = float(kl_ps.detach().mean())
            info["loss_kl_unweighted"] = kl_unweighted
            # Stage 1's two weighting layers, v-mode only (the u-mode KL above
            # stays a plain MSE). ``_kl_use_stage1_weighting = False`` keeps
            # the unweighted per-sample MSE.
            if self._kl_use_stage1_weighting:
                # Layer 1: t-weight from the ``_kl_weight_type`` curve.
                w_t = self._v_kl_reweight(t_for_model).to(kl_ps.dtype)
                kl_ps = kl_ps * w_t
                # Layer 2: mode rebalance, applied to the already t-weighted
                # residual, as in Stage 1. Order matters: swapping them would
                # rescale to a reference mean that the t-weight then perturbs
                # back off target.
                r_cd_bal = r_for_model if r_for_model is not None else t_for_model
                rebal = self._kl_mode_rebalance(kl_ps, t_for_model, r_cd_bal)
                kl_ps = kl_ps * rebal
                info["kl_t_weight_mean"] = float(w_t.mean())
                info["kl_rebalance_mean"] = float(rebal.mean())

        # ``kl_ps is None`` whenever the KL is off (kl_weight == 0, or no
        # reference model built). Returning a zero scalar rather than reducing
        # None: the caller adds this unconditionally and ``loss_kl`` must be
        # emitted on every rank and every trainer or ``_reduce_metrics`` --
        # which all-reduces per key with no key-set reconciliation -- hangs
        # until the NCCL watchdog fires. Built from u_theta_f so dtype/device
        # follow the work tensor.
        if kl_ps is None:
            zero = torch.zeros((), device=u_theta_f.device, dtype=u_theta_f.dtype)
            return zero, info
        return kl_ps.mean(), info

    
    @torch.no_grad()
    def _kl_mode_rebalance(
        self, kl_per_sample: torch.Tensor, t_cd: torch.Tensor, r_cd: torch.Tensor,
    ) -> torch.Tensor:
        """AnyFlow Stage 1's scale rebalance: non-diffusion samples -> global diff mean.

        Returns a per-sample multiplier ``[B]`` (1.0 for diffusion samples), so
        the caller applies it and the result stays differentiable.

        Why the collective is needed
        ----------------------------
        The reference mean is global (Stage 1 all-gathers too) because the
        three-mode partition is by global index, so a rank's local batch is
        typically mode-homogeneous. For the shipped configs
        (``batch_size=9``, ``num_batches_per_epoch=16`` -> ``n_local=144``,
        ``world_size=8``, ``diffusion_ratio=0.5``, ``consistency_ratio=0.25``)::

            rank 0-3 : g_idx    0..575  -> all r == t   (diffusion)
            rank 4-5 : g_idx  576..863  -> all r == 0   (consistency)
            rank 6-7 : g_idx  864..1151 -> all generic

        Ranks 4-7 hold no diffusion samples and cannot form the reference mean
        locally; ranks 0-3 hold no non-diffusion samples and have nothing to
        rescale. A rank-local implementation would silently be a no-op on
        every rank.

        Collective safety
        -----------------
        The ``all_reduce`` below is unconditional: every rank enters it
        regardless of which modes it holds, and the mask/count arithmetic that
        follows is pure local tensor math. As in :meth:`_pre_compute_v_hook`, a
        data-dependent predicate gating the collective (e.g. an
        ``if is_diff.any()`` shortcut) would deadlock the ranks that skip it.
        """
        is_diff = (t_cd == r_cd)
        # Sum + count reduced together so one collective carries both; the
        # global diffusion mean then needs no second round trip.
        stat = torch.stack([
            (kl_per_sample.detach() * is_diff).sum().double(),
            is_diff.sum().double(),
        ])
        if dist.is_initialized() and get_world_size() > 1:
            dist.all_reduce(stat, op=dist.ReduceOp.SUM)
        n_diff = stat[1]
        if float(n_diff) <= 0.0:
            # diffusion_ratio == 0: no reference slice exists anywhere, so
            # Stage 1's rebalance is undefined. Fall through unscaled rather
            # than inventing a reference. Config-level, hence rank-uniform.
            return torch.ones_like(kl_per_sample)
        diff_mean = (stat[0] / n_diff).to(kl_per_sample.dtype)
        # Stage 1's formula:
        #   scale = diff_mean / (local_nondiff + 1e-5)
        # so each non-diffusion sample lands exactly ON diff_mean. The 1e-5
        # matches Stage 1 and also guards the r==t-only case where the
        # denominator would be the sample's own ~0 residual.
        scale = diff_mean / (kl_per_sample.detach() + 1e-5)
        return torch.where(is_diff, torch.ones_like(scale), scale)


    def _v_kl_reweight(self, t_raw: torch.Tensor) -> torch.Tensor:
        """AnyFlow Stage 1's per-sample t-weight, ``[B]``, in fp32.

        Built from its own :class:`FlowMapScheduler` because the shared
        ``self.flowmap_scheduler`` is constructed ``weight_type="uniform"``:
        calling ``get_train_weight`` on it would return a flat 0.999 and
        silently make this a no-op. Cached on first use; the curve is a pure
        function of ``(T, shift, weight_type)`` so one instance serves the
        whole run.
        """
        cached = getattr(self, "_kl_weight_sched", None)
        if cached is None:
            from fewsteprl.schedulers.flowmap_scheduler import FlowMapScheduler
            sched_cfg = self.scheduler.config
            cached = FlowMapScheduler(
                num_train_timesteps=int(sched_cfg.num_train_timesteps),
                shift=float(getattr(sched_cfg, "shift", 1.0)),
                weight_type=self._kl_weight_type,
            )
            self._kl_weight_sched = cached
        return cached.get_train_weight(t_raw).to(torch.float32).flatten()
    



    # ==================================================================
    # Loss kernels (single (x_0, t, eps, advantage) microbatch)
    # ==================================================================

    @staticmethod
    def _draw_eps(x_0: torch.Tensor) -> torch.Tensor:
        """Draw the forward-noising epsilon from the global RNG.

        A separate method so the noise can be pinned or replaced. Comparing two
        trainers on the same batch also requires re-seeding the global RNG
        immediately before the call; otherwise they noise different samples.
        """
        return torch.randn_like(x_0)


    @staticmethod
    def _per_sample_mse(
        pred: torch.Tensor, target: torch.Tensor, spatial: tuple[int, ...],
    ) -> torch.Tensor:
        """``mean_spatial[(pred - target) ** 2]`` -> ``[B]``.

        Not ``F.mse_loss``: its default ``reduction='mean'`` folds the batch
        away too, and the caller needs one value per sample (it scales by a
        per-sample fp64 weight and by ``adv_clip_max``). The reduction runs in
        the residual's own dtype (fp32); only the coefficient is fp64.
        """
        return ((pred - target) ** 2).mean(dim=spatial)

    def self_regression(
        self,
        x_pred: torch.Tensor,
        x_reg: torch.Tensor,
        weight: torch.Tensor,
        spatial: tuple[int, ...],
    ) -> torch.Tensor:
        """``W * mean_sp[(x_pred - x_reg)^2]`` -> ``[B]``.

        Gradient ``(2W/N) * (x_pred - x_reg)``: the self-regression half of the
        gradient bracket in :meth:`decoupled_rl_loss`.
        """
        return weight * self._per_sample_mse(x_pred, x_reg, spatial)

    def reward_augmentation(
        self,
        x_pred: torch.Tensor,
        x_pos: torch.Tensor,
        x_neg: torch.Tensor,
        omega: torch.Tensor,
        weight: torch.Tensor,
        spatial: tuple[int, ...],
    ) -> torch.Tensor:
        """Injects the gradient ``-(2W/N) * omega * (x_pos - x_neg)`` -> ``[B]``.

        ``x_target`` is a stop-grad probe center built from ``x_pred.detach()``,
        so the residual ``x_pred - x_target`` is ``-norm_grad`` in value while
        its Jacobian w.r.t. ``x_pred`` is the identity. The gradient identity is
        therefore exact, not a first-order approximation, and the loss value is
        ``W * mean_sp[norm_grad ** 2]``.

        Sign (load-bearing)
        -------------------
        ``omega > 0`` means a positive advantage, which must move ``x_pred``
        toward the rewarded sample. That requires the residual to be
        ``-norm_grad``, i.e. ``norm_grad`` must be added to the detached center
        and the direction must be ``x_pos - x_neg`` (rewarded minus probe
        center). Writing ``x_pred - (x_pred.detach() - norm_grad)`` instead
        yields ``+norm_grad`` and flips the gradient, training the policy away
        from the rewarded sample -- while leaving the loss value bit-identical
        at ``norm_grad ** 2``, so no logged metric can detect it.

        Numerical cost
        --------------
        Materializing ``x_pred.detach() + norm_grad`` rounds the smaller
        increment against the larger prediction, so the recovered residual is
        ``-norm_grad + eps`` with ``|eps| <~ u * |x_pred|`` (``u`` the unit
        roundoff of ``x_pred``'s dtype, ~1e-7 relative in fp32). The
        cancellation-free alternative is ``(x_pred - x_pred.detach()) -
        norm_grad``, whose residual is exactly ``-norm_grad``; switch to it if
        bitwise or fp64-level agreement is ever required.
        """
        with torch.no_grad():
            norm_grad = (omega * (x_pos.double() - x_neg.double())).to(x_pred.dtype)
        x_target = (x_pred.detach() + norm_grad).to(x_pred.dtype)
        return weight * self._per_sample_mse(x_pred, x_target, spatial)


    def u_kl_loss(self, u_pred: torch.Tensor, u_ref: torch.Tensor, spatial: tuple[int, ...]) -> torch.Tensor:
        """Per-sample KL surrogate: ``mean_spatial[(u_pred - u_ref) ** 2]``.

        :param u_pred: The predicted velocity, shape [B, C, H, W].
        :param u_ref: The reference velocity, same shape.
        :param spatial: The non-batch dims to average over, e.g. (1, 2, 3).
        :return: The per-sample KL term, shape [B].
        """
        return self._per_sample_mse(u_pred, u_ref, spatial)
    
    @torch.no_grad()
    def decoupled_weights(
        self, 
        x_0_pred_u: torch.Tensor, 
        x_anchor_1: torch.Tensor, 
        x_anchor_2: torch.Tensor, 
        x_0_rwd: torch.Tensor, 
        r_hat: torch.Tensor, 
        beta: float, 
        spatial: tuple[int, ...]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Anchor-form weights ``(reg_weight, rwd_weight, omega)``, in fp64.

        ``reg_weight == rwd_weight == W = beta * (a + b)`` (``[B]``) and
        ``omega = (a - b) / W`` (``[B, 1, 1, 1]``), with ``a = r_hat / w_pos``,
        ``b = (1 - r_hat) / w_neg`` and ``w_pos`` / ``w_neg`` the mean absolute
        distances of the probes ``x_anchor_2 +/- beta * dev`` from ``x_0_rwd``.
        """
        r_broadcast = r_hat.view(-1, *([1] * (x_0_pred_u.ndim - 1))).double()
        # Probe direction from anchor_1 (S-free), center from anchor_2.
        # Keep dev in fp32: promoting it to fp64 here shifts it by ~9e-08
        # even on the S == 0 slice.
        dev = x_0_pred_u.detach() - x_anchor_1
        probe_pos = x_anchor_2 + beta * dev
        probe_neg = x_anchor_2 - beta * dev
        w_pos = (
            (probe_pos.double() - x_0_rwd.double())
            .abs().mean(dim=spatial, keepdim=True).clip(min=1e-5)
        )
        w_neg = (
            (probe_neg.double() - x_0_rwd.double())
            .abs().mean(dim=spatial, keepdim=True).clip(min=1e-5)
        )
        a = r_broadcast / w_pos
        b = (1.0 - r_broadcast) / w_neg
        ab = a + b
        omega = (a - b) / (beta * ab)
        # Same weight on both terms (see decoupled_rl_loss). Flattened to [B]
        # so it scales the per-sample losses rather than broadcasting over the
        # spatial residual; omega keeps its [B, 1, 1, 1] shape because it
        # multiplies a spatial direction.
        reg_weight = rwd_weight = (beta * ab).reshape(-1)
        return reg_weight, rwd_weight, omega

    def decoupled_rl_loss(
        self,
        x_0_pred_u: torch.Tensor,   # fp32, the only tensor carrying grad
        x_anchor_1: torch.Tensor,   # fp32 detached, S-free: direction origin + target base
        x_anchor_2: torch.Tensor,   # fp32 detached: probe center (per-trainer)
        x_0_rwd: torch.Tensor,    # fp32 detached, the rewarded sample
        r_hat: torch.Tensor,
        beta: float,
        spatial: tuple[int, ...],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Term-split form of the fused x0 policy loss (see the module docstring).

        The fused loss is a single weighted MSE,
        ``W * mean_sp[(P - sg(x_anchor_1 + omega * (x_0_rwd - x_anchor_2)))^2]``
        with ``P = x_0_pred_u``, whose gradient bracket is::

            dL/dP = (2/N) * [ beta*(a+b) * (P - x_anchor_1)          # self-reg
                              - (a-b)    * (x_0_rwd - x_anchor_2) ]  # reward

        This method emits those two terms as separate losses so each is
        observable on its own.

        Why both weights are ``W = beta*(a+b)``
        ---------------------------------------
        ``W * omega == a - b`` exactly (beta and (a+b) cancel), so the reward
        term's ``omega`` already carries the ``1/W`` that turns ``W`` back into
        ``a - b``. Matching the bracket therefore requires
        ``(w_r - W)*(P - x_anchor_1) == (w_a - W)*omega*(x_0_rwd - x_anchor_2)``
        for all inputs; since ``x_anchor_1``, ``x_anchor_2`` and ``x_0_rwd`` vary
        independently the two directions can be made non-parallel, which forces
        both coefficients to zero. So ``reg_weight == rwd_weight == W`` is
        necessary, not merely one convenient choice.

        Both weights are applied inside the sub-losses, so the caller must not
        multiply the result by a weight again.

        What is and is not preserved
        ----------------------------
        * Gradients match the fused loss exactly (up to the rounding noted in
          :meth:`reward_augmentation`).
        * Loss values do not. They exceed the fused value by the cross term
          ``2*W*mean_sp[(P - x_anchor_1) * omega*(x_0_rwd - x_anchor_2)]``, whose
          sign is indefinite, so compare runs on gradients, never on loss.
        * The Hessian differs by 2x: each sub-loss contributes ``2W/N``, versus
          ``2W/N`` total for the fused form. Harmless here (AdamW's second moment
          tracks squared gradients, which match, and the grad-norm clip only
          reads gradients) but any curvature-aware method would see it.

        ``x_anchor_1`` / ``x_anchor_2`` may be None, which zeroes the matching
        term; trainers that pass None override :meth:`decoupled_weights`
        accordingly.

        Metric keys
        -----------
        ``self_regression_loss`` / ``reward_augmentation_loss`` reach the logs
        via :attr:`_EXTRA_LOSS_KEYS`, which is a class constant so the key set
        is identical on every rank (see there).
        """
        with torch.no_grad():
            reg_weight, rwd_weight, omega = self.decoupled_weights(
                x_0_pred_u, x_anchor_1, x_anchor_2, x_0_rwd, r_hat, beta, spatial,
            )
        if x_anchor_1 is not None:
            self_reg_loss = self.self_regression(
                x_0_pred_u, x_anchor_1, reg_weight, spatial,
            )
        else:
            self_reg_loss = torch.zeros_like(reg_weight)
        if x_anchor_2 is not None:
            rwd_aug_loss = self.reward_augmentation(
                x_0_pred_u, x_0_rwd, x_anchor_2, omega, rwd_weight, spatial,
            )
        else:
            rwd_aug_loss = torch.zeros_like(rwd_weight)
        return self_reg_loss + rwd_aug_loss, {
            # ``affine_*``: mean W and mean omega; the other two are the
            # per-term losses.
            "affine_coeff_mean": float(reg_weight.mean().item()),
            "affine_adv_coeff_mean": float(omega.mean().item()),
            "self_regression_loss": float(self_reg_loss.detach().mean()),
            "reward_augmentation_loss": float(rwd_aug_loss.detach().mean()),
        }

    def fm_loss(
        self,
        v_pred,
        v_cond,
        spatial: tuple[int, ...],
    ) -> torch.Tensor:
        """Per-sample flow-matching loss ``mean_spatial[(v_pred - v_cond) ** 2]``.

        :param v_pred: The predicted velocity, shape [B, C, H, W].
        :param v_cond: The conditional velocity, same shape.
        :param spatial: The non-batch dims to average over, e.g. (1, 2, 3).
        :return: The per-sample loss, shape [B].
        """
        return self._per_sample_mse(v_pred, v_cond, spatial)
    # ==================================================================
    # Reward + advantage pipeline
    # ==================================================================

    def _ensure_reward_scorer(self) -> None:
        if self._reward_scorer is not None:
            return
        cfg = self._rl_cfg()
        if not cfg.reward_fn:
            raise ValueError(
                f"{type(self).__name__} requires fewstep_rl.reward_fn "
                "with at least one weighted reward."
            )
        if cfg.reward_ckpt_path:
            set_ckpt_path(cfg.reward_ckpt_path)
        # Init on CPU and move to GPU per call (matches eval scorer pattern).
        self._reward_scorer = MultiScorer(
            device=torch.device("cpu"),
            score_dict=cfg.reward_fn,
            allow_unavailable=False,
        )

    @staticmethod
    def _normalize_score_array(vals: Any) -> torch.Tensor:
        """Convert MultiScorer per-reward output to a CPU fp32 tensor [B]."""
        if isinstance(vals, torch.Tensor):
            return vals.detach().cpu().float()
        if isinstance(vals, np.ndarray):
            return torch.from_numpy(vals).float()
        if isinstance(vals, list):
            return torch.tensor(
                [
                    float(v.detach().cpu().item()) if isinstance(v, torch.Tensor)
                    else float(v)
                    for v in vals
                ],
                dtype=torch.float32,
            )
        return torch.tensor(np.asarray(vals), dtype=torch.float32)

    @torch.no_grad()
    def _score_rewards(
        self, images: torch.Tensor, prompts: list[str],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Score images with the multi-reward backend.

        Returns:
            mean_t:        weighted sum reward, shape [B], device=cuda.
            per_reward:    dict {name: [B] cpu fp32 tensor} for each active
                           reward backend (used for per-reward wandb panels).
        """
        self._ensure_reward_scorer()
        device = torch.device("cuda")
        self._reward_scorer.to(device)
        try:
            score_details, _ = self._reward_scorer(
                images, prompts, metadata=None, only_strict=False,
            )
            mean_vals = score_details["mean"]
            per_reward: dict[str, torch.Tensor] = {}
            for name in getattr(
                self._reward_scorer, "active_reward_names",
                list(self._rl_cfg().reward_fn.keys()),
            ):
                if name in score_details:
                    per_reward[name] = self._normalize_score_array(score_details[name])
        finally:
            self._reward_scorer.to(torch.device("cpu"))
            torch.cuda.empty_cache()

        mean_t = self._normalize_score_array(mean_vals).to(device=device)
        return mean_t, per_reward

    @torch.no_grad()
    def _gather_local_tensor(self, t: torch.Tensor) -> torch.Tensor:
        """All-gather a 1-D tensor across ranks; returns concatenated tensor.

        Used for global reward stats so every rank reports the same mean / std.
        """
        ws = get_world_size()
        if not dist.is_initialized() or ws <= 1:
            return t.clone().to(device=torch.device("cuda"))
        t = t.contiguous().to(device=torch.device("cuda"))
        bucket = [torch.empty_like(t) for _ in range(ws)]
        dist.all_gather(bucket, t)
        return torch.cat(bucket, dim=0)

    @torch.no_grad()
    def _gather_prompts_and_rewards(
        self, prompts_local: list[str], rewards_local: torch.Tensor,
    ) -> tuple[list[str], torch.Tensor]:
        """All-gather prompts and rewards across ranks."""
        ws = get_world_size()
        if not dist.is_initialized() or ws <= 1:
            return list(prompts_local), rewards_local.clone()

        prompts_bucket: list[list[str] | None] = [None] * ws
        dist.all_gather_object(prompts_bucket, list(prompts_local))
        all_prompts: list[str] = []
        for chunk in prompts_bucket:
            if chunk:
                all_prompts.extend(chunk)

        rewards_local = rewards_local.contiguous()
        all_rewards = [torch.empty_like(rewards_local) for _ in range(ws)]
        dist.all_gather(all_rewards, rewards_local)
        all_rewards = torch.cat(all_rewards, dim=0)
        return all_prompts, all_rewards

    @torch.no_grad()
    def _compute_advantages(
        self, prompts_local: list[str], rewards_local: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-sample advantages with cross-rank stat tracking."""
        rank = get_rank()
        n_local = rewards_local.shape[0]

        all_prompts, all_rewards = self._gather_prompts_and_rewards(
            prompts_local, rewards_local,
        )

        if self._stat_tracker is None:
            # Without per-prompt tracking, fall back to global standardization.
            r = all_rewards.cpu().numpy().astype(np.float64)
            adv = (r - r.mean()) / (r.std() + 1e-4)
            adv_local = torch.from_numpy(adv).to(rewards_local.device, dtype=torch.float32)
            return adv_local[rank * n_local:(rank + 1) * n_local]

        all_adv = self._stat_tracker.update(all_prompts, all_rewards.cpu().numpy())
        adv_local_np = all_adv[rank * n_local:(rank + 1) * n_local]
        return torch.from_numpy(adv_local_np).to(
            device=rewards_local.device, dtype=torch.float32,
        )

    @torch.no_grad()
    def _compute_advantages_weighted(
        self,
        prompts_local: list[str],
        rewards_local: torch.Tensor,
        per_reward_local: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """``weight_advantages=True``: per-reward advantages, then weight + normalize.

        Instead of normalizing the weighted-sum reward (see
        :meth:`_compute_advantages`), compute each reward's advantage
        independently -- via its own fresh per-prompt stat tracker, or global
        standardization when per-prompt tracking is off -- weighted-sum the
        advantages, then normalize the result to zero-mean / unit-std. Mirrors the
        DiffusionNFT's ``weight_advantages`` path. Returns this rank's slice.

        Falls back to :meth:`_compute_advantages` if no per-reward scores are
        available (e.g. every reward backend was skipped).
        """
        cfg = self._rl_cfg()
        rank = get_rank()
        n_local = rewards_local.shape[0]
        weights: dict[str, float] = dict(cfg.reward_fn or {})

        # Gather prompts once (+ the weighted-sum reward, used only to keep the
        # main stat tracker's summary stats meaningful in this mode).
        all_prompts, all_weighted = self._gather_prompts_and_rewards(
            prompts_local, rewards_local,
        )

        total_adv: np.ndarray | None = None
        for name, weight in weights.items():
            if name not in per_reward_local:
                continue
            r_local = per_reward_local[name].to(device=rewards_local.device).float()
            all_r = self._gather_local_tensor(r_local)
            r_np = all_r.cpu().numpy().astype(np.float64)
            if cfg.per_prompt_stat_tracking:
                tracker = PerPromptStatTracker(global_std=cfg.per_prompt_global_std)
                adv = np.asarray(tracker.update(all_prompts, r_np), dtype=np.float64)
            else:
                adv = (r_np - r_np.mean()) / (r_np.std() + 1e-4)
            adv = adv * float(weight)
            total_adv = adv if total_adv is None else total_adv + adv

        if total_adv is None:
            # No per-reward scores available -> plain weighted-sum advantages.
            return self._compute_advantages(prompts_local, rewards_local)

        # Normalize the weighted-sum advantage (zero-mean / unit-std).
        total_adv = (total_adv - total_adv.mean()) / (total_adv.std() + 1e-4)

        # Populate the main tracker on the weighted-sum reward so the
        # DiffusionNFT-style summary stats in ``_collect_reward_stats`` stay valid
        # (the caller clears it afterwards).
        if self._stat_tracker is not None:
            self._stat_tracker.update(all_prompts, all_weighted.cpu().numpy())

        adv_local = np.ascontiguousarray(total_adv[rank * n_local:(rank + 1) * n_local])
        return torch.from_numpy(adv_local).to(
            device=rewards_local.device, dtype=torch.float32,
        )

    # ==================================================================
    # Old model EMA decay (toward generator)
    # ==================================================================

    def _current_decay(self) -> float:
        """DiffusionNFT-aligned decay schedule.

        Mirrors ``DiffusionNFT/scripts/train_nft_sd3.py::return_decay`` line-
        by-line. The driving counter is ``self.global_step`` (generator
        optimizer steps), not outer epochs; the two coincide only when every
        epoch makes exactly one optimizer step.
        """
        cfg = self._rl_cfg()
        step = int(self.global_step)
        if cfg.decay_type == 0:
            flat, uprate, uphold = 0, 0.0, 0.0
        elif cfg.decay_type == 1:
            flat, uprate, uphold = 0, 0.001, 0.5
        elif cfg.decay_type == 2:
            flat, uprate, uphold = 75, 0.0075, 0.999
        elif cfg.decay_type == 3:
            return 1.0
        else:
            raise ValueError(f"Unknown fewstep_rl.decay_type={cfg.decay_type}")
        if step < flat:
            return 0.0
        return min(float((step - flat) * uprate), float(uphold))

    def _current_ema_decay(self) -> float:
        """DiffusionNFT EMA warmup schedule for ``generator_ema``.

            decay(s) = min((1 + s) / (10 + s), target_decay)

        where ``s = self.global_step`` (per-opt-step counter). Early steps
        use a smaller decay so EMA tracks the model quickly; the cap kicks
        in once ``(1+s)/(10+s) >= target_decay``. ``target_decay`` is read from
        ``train.ema_decay`` (DiffusionNFT default 0.9, but configurable).
        """
        s = int(self.global_step)
        target = float(self.config.train.ema_decay)
        return min((1.0 + float(s)) / (10.0 + float(s)), target)

    def _decay_old_model(self) -> float:
        cfg = self._rl_cfg()
        if cfg.rl_decay_interval <= 0:
            return 0.0
        if (self._rl_epoch + 1) % cfg.rl_decay_interval != 0:
            return 0.0
        decay = self._current_decay()
        # ema_update(src, tgt, decay): tgt = decay * tgt + (1 - decay) * src.
        # We want old <- decay * old + (1 - decay) * generator, so src=gen, tgt=old.
        self.ema_update(self.generator, self.old_model, decay)
        return decay

    # ==================================================================
    # Sampling phase: collect num_batches_per_epoch sampling rounds
    # ==================================================================

    @torch.no_grad()
    def _one_sampling_batch(
        self, prompts: list[str], return_rewards: bool = True, return_intermediate: bool = False,
    ) -> dict[str, Any]:
        """Roll out one K-repeat batch on this rank; returns x_0, embeds, rewards, ...

        ``return_rewards=False`` keeps the rollout but scores nothing, so
        ``rewards`` and ``per_reward`` come back None. The VAE decode is skipped
        with it unless ``log_sample_images`` still wants a preview: decode plus
        scoring is where the cost of this path is, so gating the scorer alone
        would leave most of it behind.

        The sampler is ``old_model`` (or ``generator_ema`` under
        ``rl_sample_use_ema``) for every caller, policy and aux alike -- one
        behaviour policy per training step, so the rewarded ``x_0`` and both
        anchors describe the same distribution. Not parameterized: an override
        here would have to be rank-uniform anyway, since the rollout issues FSDP
        collectives.

        Must be called with the same ``return_rewards`` on every rank -- the
        scorers and the advantage reduction downstream are collective.
        """
        cfg = self._rl_cfg()
        dtype = next(self.generator.parameters()).dtype

        with self._autocast():
            prompt_embeds, pooled_embeds = self._encode_prompts(prompts)
        prompt_embeds = prompt_embeds.to(dtype=dtype)
        pooled_embeds = pooled_embeds.to(
            dtype=dtype if pooled_embeds.is_floating_point() else pooled_embeds.dtype,
        )

        num_steps = int(cfg.sampling_num_steps)
        sampler_model = (
            self.generator_ema if (cfg.rl_sample_use_ema and self.generator_ema is not None)
            else self.old_model
        )
        sampler_model.eval()


        rollout_type = self._rl_cfg().rollout_type
        if rollout_type == "ode":
            x_0, x_r, x_t, ets, mts, sts = self._rollout_samples_ode(
                prompt_embeds=prompt_embeds,
                pooled_embeds=pooled_embeds,
                num_steps=num_steps,
                guidance_scale=cfg.sampling_guidance_scale,
                model=sampler_model,
                return_intermediate=return_intermediate
            )
        elif rollout_type == "sde":
            x_0, x_r, x_t, ets, mts, sts = self._rollout_samples_sde(
                prompt_embeds=prompt_embeds,
                pooled_embeds=pooled_embeds,
                num_steps=num_steps,
                guidance_scale=cfg.sampling_guidance_scale,
                model=sampler_model,
                return_intermediate=return_intermediate
            )
        else:
            raise ValueError(
                f"rollout_type must be 'ode' or 'sde', got {rollout_type!r}."
            )

        # Record the rollout's (t, r) jump grid per sample, as DiffusionNFT
        # stores ``timesteps`` / ``next_timesteps``. The base inner loop draws
        # fresh (t, r) and reads this only for its shape and
        # :meth:`_prepare_inner_epoch_tr_state`.


        t_arr, r_arr = self._get_last_rollout_timestep_pairs()
        t_per_sample = t_arr.detach().cpu().repeat(x_0.shape[0], 1)
        r_per_sample = r_arr.detach().cpu().repeat(x_0.shape[0], 1)


        want_preview = int(cfg.log_sample_images) > 0
        images = (
            self._decode_latents_to_tensor(x_0)
            if (return_rewards or want_preview) else None
        )
        if return_rewards:
            rewards, per_reward = self._score_rewards(images, prompts)
            rewards = rewards.detach()
        else:
            rewards, per_reward = None, None

        return {
            "x_0": x_0.detach().cpu(),
            "prompt_embeds": prompt_embeds.detach().cpu(),
            "pooled_embeds": pooled_embeds.detach().cpu(),
            "timesteps": t_per_sample,
            "next_timesteps": r_per_sample,
            "prompts": list(prompts),
            "rewards": rewards,
            "per_reward": per_reward,
            "images_preview": (
                images.detach().cpu() if (images is not None and want_preview) else None
            ),
            "x_r": x_r.detach().cpu() if return_intermediate else None,
            "x_t": x_t.detach().cpu() if return_intermediate else None,
            "end_timesteps": ets.detach().cpu() if return_intermediate else None,
            "mid_timesteps": mts.detach().cpu() if return_intermediate else None,
            "start_timesteps": sts.detach().cpu() if return_intermediate else None,
        }

    def _rl_sampling_phase(
        self, return_rewards: bool = True, return_intermediate: bool = False,
    ) -> tuple[dict[str, Any], Optional[tuple[torch.Tensor, list[str]]]]:
        """Run ``num_batches_per_epoch`` sampling rounds + global advantage compute.

        Returns a concatenated dict of CPU tensors per rank
        (``x_0``, ``prompt_embeds``, ``pooled_embeds``, ``advantages``) plus
        the flat prompt list (for image logging).

        ``return_rewards=False`` collects rollouts only: no scorer runs and
        ``rewards`` / ``per_reward`` / ``advantages`` / ``reward_stats`` are all
        None. The stat tracker is left untouched rather than cleared, since
        nothing was added to it.

        The flag must be rank-uniform. ``_compute_advantages`` all-gathers, so
        ranks that disagree here hang for the full NCCL timeout instead of
        raising.
        """
        cfg = self._rl_cfg()
        N = max(1, int(cfg.num_batches_per_epoch))
        self.generator.eval()

        batches: list[dict[str, Any]] = []
        first_preview = None
        for i in range(N):
            # Advance the K-repeat partition deterministically per global
            # sampling batch (matches DiffusionNFT's
            # `epoch * num_batches_per_epoch + i` seeding).
            sampler_epoch = self._rl_epoch * N + i
            self._rl_sampler.set_epoch(sampler_epoch)
            try:
                prompts = next(self._rl_sample_iter)
            except StopIteration:
                self._rl_sample_iter = iter(self._rl_dataloader)
                prompts = next(self._rl_sample_iter)
            assert isinstance(prompts, list) and len(prompts) > 0

            pack = self._one_sampling_batch(
                prompts, return_rewards=return_rewards,  
                return_intermediate=return_intermediate
            )
            batches.append(pack)
            if first_preview is None and pack["images_preview"] is not None:
                first_preview = (pack["images_preview"], pack["prompts"])

        # Concatenate per-rank slices across the sampling phase.
        x_0_all = torch.cat([b["x_0"] for b in batches], dim=0)
        emb_all = torch.cat([b["prompt_embeds"] for b in batches], dim=0)
        pool_all = torch.cat([b["pooled_embeds"] for b in batches], dim=0)
        ts_all = torch.cat([b["timesteps"] for b in batches], dim=0)
        nts_all = torch.cat([b["next_timesteps"] for b in batches], dim=0)

        # Only populated under ``return_intermediate``; every one of these is
        # None otherwise, so the cat has to be guarded rather than unconditional.
        # Sample axis is leading on all five, so dim=0 concatenates samples.
        intermediates: dict[str, Optional[torch.Tensor]] = {}
        for key in ("x_r", "x_t", "end_timesteps", "mid_timesteps", "start_timesteps"):
            intermediates[key] = (
                torch.cat([b[key] for b in batches], dim=0)
                if batches[0].get(key) is not None else None
            )

        prompts_all: list[str] = [p for b in batches for p in b["prompts"]]

        rewards_all: Optional[torch.Tensor] = None
        per_reward_all: Optional[dict[str, torch.Tensor]] = None
        advantages: Optional[torch.Tensor] = None
        reward_stats: Optional[dict[str, float]] = None
        if return_rewards:
            rewards_all = torch.cat([b["rewards"] for b in batches], dim=0)

            # Per-reward concatenation across sampling rounds.
            per_reward_all = {}
            for k in list((batches[0]["per_reward"] or {}).keys()):
                per_reward_all[k] = torch.cat(
                    [b["per_reward"][k] for b in batches], dim=0,
                )

            # Cross-rank gather + per-prompt advantage. After computing
            # advantages, capture stat-tracker summary stats (DiffusionNFT-style
            # logging), then clear so the next epoch's stats are derived only
            # from that epoch's K samples (pure on-policy normalization).
            if getattr(cfg, "weight_advantages", False):
                advantages = self._compute_advantages_weighted(
                    prompts_all, rewards_all, per_reward_all,
                )
            else:
                advantages = self._compute_advantages(prompts_all, rewards_all)
            reward_stats = self._collect_reward_stats(prompts_all, rewards_all)
            if self._stat_tracker is not None:
                self._stat_tracker.clear()

        sample_pack = {
            "x_0": x_0_all,
            "prompt_embeds": emb_all,
            "pooled_embeds": pool_all,
            "timesteps": ts_all,
            "next_timesteps": nts_all,
            "prompts": prompts_all,
            "rewards": rewards_all,
            "per_reward": per_reward_all,
            "advantages": advantages,
            "reward_stats": reward_stats,
            **intermediates,
        }
        return sample_pack, first_preview

    @torch.no_grad()
    def _collect_reward_stats(
        self, prompts_all: list[str], rewards_all: torch.Tensor,
    ) -> dict[str, float]:
        """Capture per-prompt summary stats before ``stat_tracker.clear()``.

        Aligned with DiffusionNFT's wandb log block:
            group_size, trained_prompt_num
            zero_std_ratio, reward_std_mean
            mean_reward_{100, 75, 50, 25, 10}

        Returns a dict of CPU floats; safe to log from any rank (the
        tracker's per-rank stats already include all globally-gathered
        prompts/rewards via ``_compute_advantages``'s ``all_gather``).
        ``zero_std_ratio`` / ``reward_std_mean`` are computed from the
        arguments, i.e. from this rank's samples only.
        """
        stats: dict[str, float] = {}
        if self._stat_tracker is not None:
            group_size, trained_prompt_num = self._stat_tracker.get_stats()
            stats["group_size"] = float(group_size)
            stats["trained_prompt_num"] = float(trained_prompt_num)
            for pct in (100, 75, 50, 25, 10):
                stats[f"mean_reward_{pct}"] = float(
                    self._stat_tracker.get_mean_of_top_rewards(pct)
                )
        prompts_arr = np.array(prompts_all)
        rewards_np = rewards_all.detach().cpu().numpy().astype(np.float64)
        unique_prompts, inverse_indices, counts = np.unique(
            prompts_arr, return_inverse=True, return_counts=True,
        )
        if len(unique_prompts) > 0 and rewards_np.size > 0:
            grouped = rewards_np[np.argsort(inverse_indices)]
            split_indices = np.cumsum(counts)[:-1]
            reward_groups = np.split(grouped, split_indices)
            prompt_stds = np.array([np.std(g) for g in reward_groups])
            zero_std_count = int(np.count_nonzero(prompt_stds == 0))
            stats["zero_std_ratio"] = float(zero_std_count / len(prompt_stds))
            stats["reward_std_mean"] = float(prompt_stds.mean())
        return stats

    # ==================================================================
    # Training-time (t, r) sampling -- decoupled from the rollout grid
    #
    # r follows AnyFlow Stage 1's three-mode (diffusion / consistency /
    # generic) partition; t is clamped to [rl_min_timestep, rl_max_timestep]
    # as in the AnyFlow on-policy generator loss. Each inner backward draws a
    # fresh (t, r) per chunk from a seeded RNG keyed by (global_step,
    # _rl_epoch, inner_epoch_idx, chunk_start, j_idx, rank). The partition is
    # a deterministic function of the global index (every rank computes its
    # own slice with the same rule), so no broadcast is needed.
    # ==================================================================

    def _num_training_timesteps(self, n_ts_total: int) -> int:
        """Fresh (t, r) draws per inner chunk, decoupled from the rollout grid."""
        n = int(self._rl_cfg().num_training_timesteps_per_sample)
        if n <= 0:
            raise ValueError(
                "fewstep_rl.num_training_timesteps_per_sample must be positive."
            )
        # One-time alignment log so a misconfigured ratio is obvious on rank 0.
        if not getattr(self, "_three_mode_log_emitted", False):
            cfg = self._rl_cfg()
            total_r = float(cfg.diffusion_ratio) + float(cfg.consistency_ratio)
            if not 0.0 <= total_r <= 1.0 + 1e-6:
                raise ValueError(
                    f"fewstep_rl.diffusion_ratio ({cfg.diffusion_ratio}) + "
                    f"consistency_ratio ({cfg.consistency_ratio}) must lie in "
                    f"[0, 1] (got {total_r:.4f})."
                )
            if get_rank() == 0:
                logger.info(
                    "[GeneralRL] (t, r) sampling: three-mode partition "
                    "(diffusion_ratio=%.3f, consistency_ratio=%.3f, "
                    "generic_ratio=%.3f), t clamp=[%d, %d], "
                    "iters/inner-batch=%d (decoupled from rollout N=%d), "
                    "trainer=%s.",
                    cfg.diffusion_ratio, cfg.consistency_ratio,
                    1.0 - total_r,
                    cfg.rl_min_timestep, cfg.rl_max_timestep,
                    n, n_ts_total, self.TRAINER_KEY,
                )
            self._three_mode_log_emitted = True
        return max(1, n)

    def _prepare_inner_epoch_tr_state(self, **_kwargs: Any) -> dict[str, Any]:
        """Fresh (t, r) is drawn per iteration -- no pre-shuffle needed."""
        return {}

    def _draw_training_tr(
        self,
        *,
        state: dict[str, Any],   # noqa: ARG002 - unused; we draw fresh
        chunk_start: int,
        chunk_end: int,
        j_idx: int,
        inner_epoch_idx: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-iter (t, r) draw: three-mode partition + t clamp.

        See :meth:`_sample_training_tr_three_mode` for the recipe; returns a
        ``(t_raw, r_raw)`` pair on ``device`` in float32 raw-timestep units
        (callers do their own dtype casting).
        """
        chunk_size = max(0, chunk_end - chunk_start)
        if chunk_size == 0:
            empty = torch.empty(0, device=device, dtype=torch.float32)
            return empty, empty.clone()
        return self._sample_training_tr_three_mode(
            batch_size=chunk_size,
            chunk_start=chunk_start,
            j_idx=j_idx,
            inner_epoch_idx=inner_epoch_idx,
            device=device,
            dtype=dtype,
        )

    def _make_training_tr_rng(
        self,
        *,
        inner_epoch_idx: int,
        chunk_start: int,
        j_idx: int,
        device: torch.device,
        stream_offset: int = 0,
    ) -> torch.Generator:
        """Per-iteration seeded RNG (FSDP-symmetric partition; per-rank noise).

        Each call inside one inner train loop step must use a distinct seed so
        that consecutive ``num_training_timesteps_per_sample`` (t, r) draws are
        independent (``global_step`` only ticks on opt.step, so it is constant
        across all draws in one optimizer cycle). The seed mixes:
          - ``train.seed``                : run-level determinism
          - ``self._rl_epoch``           : outer RL epoch
          - ``self.global_step``          : generator optimizer step
          - ``inner_epoch_idx``           : inner epoch within sampling slice
          - ``chunk_start``               : chunk position
          - ``j_idx``                     : iteration index within chunk
          - ``get_rank()``                : per-rank noise (different t1/t2)
          - ``stream_offset``             : separate stream per caller
            (see :attr:`_S_RNG_STREAM_OFFSET`)
        """
        gen = torch.Generator(device=device)
        seed = (
            int(self.config.train.seed)
            + int(self._rl_epoch) * 1_000_003
            + int(self.global_step) * 10_007
            + int(inner_epoch_idx) * 101
            + int(chunk_start) * 17
            + int(j_idx) * 3
            + get_rank() * 7919
            + int(stream_offset)
        )
        gen.manual_seed(int(seed % (2**63 - 1)))
        return gen

    def _sample_training_tr_three_mode(
        self,
        *,
        batch_size: int,
        chunk_start: int,
        j_idx: int,
        inner_epoch_idx: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw one (t, r) batch: three-mode partition, then shift and clamp.

        Recipe (r as in AnyFlow Stage 1's three-mode sampler, t clamped as in
        the AnyFlow on-policy generator loss):

        1. ``t1, t2 ~ U(0, 1)`` per local sample (per-rank seeded; rank in seed
           makes ranks see different noise but the partition rule is global).
        2. ``t01 = max(t1, t2)``, ``r01 = min(t1, t2)``.
        3. Global-index partition (``global_idx = rank * batch_size + b``):
            - first ``round(diffusion_ratio * global_bsz)``  : ``r01 = t01`` (r = t)
            - next  ``round(consistency_ratio * global_bsz)``: ``r01 = 0`` (r = 0)
            - rest                                           : keep ``r01``.
        4. Apply SD3.5's static shift to both endpoints and scale to raw
           timestep units.
        5. Clamp ``t_raw`` to ``[rl_min_timestep, rl_max_timestep]``. Then
           re-clamp ``r_raw`` to ``[0, t_raw]`` so the flow-map jump direction
           stays valid (``r <= t``). Finally enforce the partition equalities
           exactly (``r = t`` on the diffusion slice, ``r = 0`` on the
           consistency slice) -- the clamp does not perturb those samples.

        FSDP-symmetric: the partition is a deterministic function of global
        index; the per-rank RNG seed mixes rank so t1/t2 differ across ranks
        but the partition rule (which sample goes to which mode) is identical.
        No broadcast or all_reduce needed.

        The ``t`` law (``fewstep_rl.renoise_distribution``)
        ---------------------------------------------------
        ``"uniform"`` (the default) is steps 1-2 above. ``"lognormal"`` instead
        draws ``t01 = sigmoid(renoise_logit_mean + renoise_logit_std * N(0,1))``,
        which moves where the training ``t`` concentrates while leaving the
        rest of the recipe unchanged.

        The conditional law of ``r`` given ``t`` is the same under either
        choice. On the lognormal path there is no second uniform to take a min
        against, so ``r01 ~ U(0, t01)`` is drawn explicitly -- the same
        conditional law ``min(t1, t2) | max(t1, t2)`` has (a standard
        order-statistics identity) -- so the three-mode partition and every
        downstream r invariant keep their meaning.
        """
        nc = self._rl_cfg()
        T = float(self.scheduler.config.num_train_timesteps)

        ws = max(1, get_world_size())
        rank = get_rank()
        global_start_idx = rank * batch_size
        global_bsz = ws * batch_size

        # Per-iteration seeded RNG (FSDP-symmetric partition; per-rank noise).
        gen = self._make_training_tr_rng(
            inner_epoch_idx=inner_epoch_idx,
            chunk_start=chunk_start,
            j_idx=j_idx,
            device=device,
        )
        renoise = str(getattr(nc, "renoise_distribution", "uniform"))
        if renoise == "uniform":
            # Default law: max / min of two uniforms, drawn in ``dtype``.
            t1 = torch.rand(batch_size, dtype=dtype, device=device, generator=gen)
            t2 = torch.rand(batch_size, dtype=dtype, device=device, generator=gen)
            t01 = torch.maximum(t1, t2)
            r01 = torch.minimum(t1, t2)
        else:
            # The normal draw and the sigmoid run in fp32 regardless of
            # ``dtype`` (in bf16, with 8 mantissa bits, the nonlinear map would
            # carry a ~1e-3 relative error); t01 is then cast to ``dtype`` like
            # the uniform path's draws, so r01 and the shift below run in
            # ``dtype``.
            g_mean = float(getattr(nc, "renoise_logit_mean", -2.053))
            g_std = float(getattr(nc, "renoise_logit_std", 0.6))
            t01 = torch.sigmoid(
                g_mean + g_std * torch.randn(
                    batch_size, dtype=torch.float32, device=device, generator=gen,
                )
            ).to(dtype=dtype)
            # r01 | t01 ~ U(0, t01): the same conditional law the uniform path's
            # min(t1,t2) has given max(t1,t2). Drawn from the same generator so
            # the stream stays chunk-addressed.
            r01 = t01 * torch.rand(
                batch_size, dtype=dtype, device=device, generator=gen,
            )

        # Three-mode partition by global index, as boolean masks.
        n_diffusion = round(float(nc.diffusion_ratio) * global_bsz)
        n_consistency = round(float(nc.consistency_ratio) * global_bsz)
        g_idx = torch.arange(
            global_start_idx, global_start_idx + batch_size,
            device=device, dtype=torch.long,
        )
        is_diffusion = g_idx < n_diffusion
        is_consistency = (g_idx >= n_diffusion) & (g_idx < n_diffusion + n_consistency)
        r01 = torch.where(is_diffusion, t01, r01)
        r01 = torch.where(is_consistency, torch.zeros_like(r01), r01)

        # Shift and scale both endpoints with SD3.5's static schedule.
        t_shifted = self.flowmap_scheduler.apply_shift(t01) * T
        r_shifted = self.flowmap_scheduler.apply_shift(r01) * T

        # Clamp t to [rl_min_timestep, rl_max_timestep]; then re-clamp r to
        # [0, t] so the flow-map jump direction stays valid (r <= t). Finally
        # re-pin the partition equalities (r = t for diffusion, r = 0 for
        # consistency) against any clamp drift.
        t_min = float(nc.rl_min_timestep)
        t_max = float(nc.rl_max_timestep)
        t_raw = t_shifted.clamp(min=t_min, max=t_max)
        r_raw = r_shifted.clamp(min=0.0).minimum(t_raw)
        r_raw = torch.where(is_diffusion, t_raw, r_raw)
        r_raw = torch.where(is_consistency, torch.zeros_like(r_raw), r_raw)

        return t_raw.to(dtype=torch.float32), r_raw.to(dtype=torch.float32)



    def _three_mode_index_masks(
        self, *, batch_size: int, device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The three-mode partition, as two ``[batch_size]`` boolean masks.

        The same expressions as the partition block inside
        :meth:`_sample_training_tr_three_mode` (same ``round``, same integer
        types). The two must stay in sync; change them together.

        Assignment is a deterministic function of the global index
        ``rank * batch_size + b`` measured against the global batch
        ``world_size * batch_size``: the first ``round(diffusion_ratio * G)``
        indices are the diffusion slice, the next ``round(consistency_ratio *
        G)`` the consistency slice, and whatever is left is generic. Every rank
        evaluates the same rule on its own slice, so the partition agrees
        across ranks with no broadcast or all_reduce -- the property that makes
        the sampler FSDP-symmetric.

        Which sample is in which mode has to match the ``(t, r)`` sampler's
        verdict for the same chunk: a sample landing in the diffusion slice for
        ``r`` but the generic slice for ``s`` would produce a transition no
        mode intended.
        """
        nc = self._rl_cfg()
        ws = max(1, get_world_size())
        rank = get_rank()
        global_start_idx = rank * batch_size
        global_bsz = ws * batch_size

        n_diffusion = round(float(nc.diffusion_ratio) * global_bsz)
        n_consistency = round(float(nc.consistency_ratio) * global_bsz)
        g_idx = torch.arange(
            global_start_idx, global_start_idx + batch_size,
            device=device, dtype=torch.long,
        )
        is_diffusion = g_idx < n_diffusion
        is_consistency = (g_idx >= n_diffusion) & (g_idx < n_diffusion + n_consistency)
        return is_diffusion, is_consistency

    def _invert_flowmap_shift(self, sigmas: torch.Tensor) -> torch.Tensor:
        """Inverse of ``flowmap_scheduler.apply_shift`` on ``[0, 1]``.

        ``apply_shift`` is the Mobius map ``y = k s / (1 + (k - 1) s)``, which
        is strictly increasing on ``[0, 1]`` and fixes both endpoints, so it is
        a bijection there and inverts in closed form::

            s = y / (k - (k - 1) y).

        The denominator runs from ``k`` at ``y = 0`` to ``1`` at ``y = 1``, so
        it is bounded below by ``min(1, k) > 0`` across the interval -- no
        singularity to guard, provided the input really is a shifted sigma.
        Reads the shift off the scheduler's own config (never a local copy of
        the constant) and mirrors its ``shift == 1.0`` short circuit, so the
        pair round-trips exactly on the no-shift configs too.
        """
        k = float(self.flowmap_scheduler.config.shift)
        if k == 1.0:
            return sigmas
        return sigmas / (k - (k - 1.0) * sigmas)

    def _draw_training_s_given_t(
        self,
        t_raw: torch.Tensor,
        *,
        state: dict[str, Any],   # noqa: ARG002 - unused; we draw fresh
        chunk_start: int,
        chunk_end: int,
        j_idx: int,
        inner_epoch_idx: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Draw ``s`` conditioned on a given ``t``, same law as ``r`` given ``t``.

        The companion to :meth:`_draw_training_tr` for callers that already
        hold a ``t`` they cannot re-draw -- e.g. MeanFlowGRPOTrainer replays
        the rollout's own transitions, so its ``t`` comes off the sampling grid
        rather than out of the sampler here. Same signature as that method,
        plus the ``t_raw`` it consumes and minus the ``t_raw`` it would have
        returned, so it drops into the same call site.

        Why the shift is undone first
        -----------------------------
        :meth:`_sample_training_tr_three_mode` draws ``t1, t2 ~ U(0, 1)`` and
        takes ``t01 = max``, ``r01 = min`` before applying the shift, then maps
        both endpoints through it. For two iid uniforms the conditional law of
        the min given the max is exactly

            r01 | t01  ~  U(0, t01),

        so reproducing ``r``'s relation to ``t`` means drawing ``u ~ U(0, 1)``
        and setting ``s01 = u * t01`` in that same pre-shift coordinate. Hence
        ``t_raw`` is pushed back through :meth:`_invert_flowmap_shift` rather
        than being treated as uniform: at SD3.5's ``shift = 3.0`` the map moves
        the midpoint from 0.5 to 0.75, so scaling the post-shift value directly
        would concentrate ``s`` toward ``t`` and silently draw from a different
        distribution than the ``r`` this is meant to mirror.

        The three-mode partition and both endpoint pins carry over unchanged:
        ``s = t`` on the diffusion slice, ``s = 0`` on the consistency slice,
        the drawn value elsewhere. ``t`` is not re-clamped to
        ``[rl_min_timestep, rl_max_timestep]`` -- it is an input here, not a
        draw, and clamping it would return an ``s`` bounded by a ``t`` the
        caller never had. ``s`` is still clamped into ``[0, t]``, which the
        shift's monotonicity already guarantees up to fp error.

        Caveat for overshoot-renoise callers
        ------------------------------------
        This bounds ``s`` by ``t`` and nothing else, because that is what
        mirroring ``r | t`` means. It does not enforce ``s <= r``, which the
        SDE rollout's overshoot does satisfy (it draws ``s`` from
        ``timesteps[i + 1:]``) and which the renoise needs: the transition std
        is ``tau_r**2 = r01**2 - (alpha_r * s01)**2`` with ``alpha_r =
        (1 - r01) / (1 - s01)``, and that is non-negative exactly when
        ``s <= r``. Feeding an ``s`` from here into a step that renoises to
        some ``r < s`` hits the ``clamp(min=0)`` guard in ``tau_r`` and yields
        a degenerate zero-variance kernel, not an error. Callers wanting the
        overshoot semantics must bound the result by their own ``r``.

        Returns ``[chunk_size]`` in float32 raw-timestep units, matching
        :meth:`_draw_training_tr`; callers do their own dtype casting.
        """
        chunk_size = max(0, chunk_end - chunk_start)
        if chunk_size == 0:
            return torch.empty(0, device=device, dtype=torch.float32)
        if t_raw.shape != (chunk_size,):
            raise ValueError(
                f"t_raw has shape {tuple(t_raw.shape)}, expected ({chunk_size},) "
                f"from chunk [{chunk_start}, {chunk_end}). The global-index "
                "partition is keyed by position in the chunk, so a mismatch "
                "would assign modes to the wrong samples."
            )

        T = float(self.scheduler.config.num_train_timesteps)
        is_diffusion, is_consistency = self._three_mode_index_masks(
            batch_size=chunk_size, device=device,
        )
        gen = self._make_training_tr_rng(
            inner_epoch_idx=inner_epoch_idx,
            chunk_start=chunk_start,
            j_idx=j_idx,
            device=device,
            stream_offset=self._S_RNG_STREAM_OFFSET,
        )
        # Drawn in the caller's ``dtype`` to consume the RNG exactly as the
        # (t, r) sampler's t1/t2 do, then promoted: the inverse-shift / shift
        # algebra below runs in fp32 (in bf16, with 8 mantissa bits, it would
        # carry a ~1e-3 relative error).
        u = torch.rand(
            chunk_size, dtype=dtype, device=device, generator=gen,
        ).to(torch.float32)

        t_f = t_raw.detach().to(device=device, dtype=torch.float32)
        # Back to the coordinate the pair was drawn in; clamped because a raw
        # timestep landing a hair outside [0, T] would leave the inverse's
        # domain and come back negative.
        t01 = self._invert_flowmap_shift((t_f / T).clamp(0.0, 1.0))
        s_shifted = self.flowmap_scheduler.apply_shift(u * t01) * T

        s_raw = s_shifted.clamp(min=0.0).minimum(t_f)
        s_raw = torch.where(is_diffusion, t_f, s_raw)
        s_raw = torch.where(is_consistency, torch.zeros_like(s_raw), s_raw)
        return s_raw.to(dtype=torch.float32)

    # ==================================================================
    # Inner train phase: num_inner_epochs * batches * num_timesteps updates
    # ==================================================================

    def _rl_loss(
        self,
        x_0: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_embeds: torch.Tensor,
        advantages: torch.Tensor,
        t_raw: torch.Tensor,
        r_raw: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """One optimizer-free loss evaluation over a single microbatch.

        Returns ``(loss, info)``; implemented by subclasses. The signature is
        spelled out rather than left as ``**kw`` because :meth:`_rl_train_phase`
        calls this by keyword, so an override whose parameter names drift is a
        TypeError at the first training step and nowhere earlier.
        """
        raise NotImplementedError

    def _rl_train_phase(
        self, sample_pack: dict[str, Any],
    ) -> dict[str, float]:
        """Run ``num_inner_epochs`` x ``num_batches`` x ``num_train_timesteps`` updates.

        Aligned with ``DiffusionNFT/scripts/train_nft_sd3.py`` (lines 869-1100)
        line-by-line:

            effective_grad_accum_steps = gradient_accumulation_steps * num_train_timesteps
            current_accumulated_steps = 0
            optimizer.zero_grad()

            for inner_epoch in range(num_inner_epochs):
                perm = randperm(B_local)
                # shuffle samples
                for k_batch in chunks(B_local, train_bs):
                    for j_idx in range(num_train_timesteps):
                        (t, r) <- _draw_training_tr(...)
                        loss = _rl_loss(sub.x_0, ..., t, r)
                        (loss / effective_grad_accum_steps).backward()
                        current_accumulated_steps += 1
                        if current_accumulated_steps % effective_grad_accum_steps == 0:
                            clip + optimizer.step + scheduler.step + zero_grad
                            ema.step  (per opt step, not per backward)
                            global_step += 1

        Notes:
        - ``global_step`` increments only on opt.step (DiffusionNFT convention).
        - The loss is divided by ``effective_grad_accum_steps`` before backward
          so the accumulated gradient equals the mean of per-backward losses
          across the accumulation window.
        - ``num_train_timesteps`` is configured independently of rollout NFE.
        - A fresh three-mode (t, r) partition is drawn per iteration.
        """
        cfg = self._rl_cfg()
        device = torch.device("cuda")

        x_0 = sample_pack["x_0"].to(device)
        prompt_embeds = sample_pack["prompt_embeds"].to(device)
        pooled_embeds = sample_pack["pooled_embeds"].to(device)
        advantages = sample_pack["advantages"].to(device)
        timesteps_all = sample_pack["timesteps"].to(device=device, dtype=torch.float32)
        next_timesteps_all = sample_pack["next_timesteps"].to(device=device, dtype=torch.float32)
        n_local, n_ts_total = timesteps_all.shape
        if n_local == 0:
            return {}

        n_train_ts = self._num_training_timesteps(n_ts_total)

        inner_bs = int(cfg.rl_inner_batch_size) if cfg.rl_inner_batch_size > 0 else n_local
        inner_bs = max(1, min(inner_bs, n_local))
        n_inner_epochs = max(1, int(cfg.num_inner_epochs))
        max_grad_norm = float(self.config.solver.generator.max_grad_norm)
        grad_accum = max(1, int(self.config.train.gradient_accumulation_steps))
        effective_grad_accum_steps = grad_accum * n_train_ts

        self.generator.train()
        accumulator: dict[str, float] = {}
        n_backward = 0       # cumulative backwards across the whole train phase
        n_opt_steps = 0
        current_accumulated_steps = 0
        self.optimizers["generator"].zero_grad(set_to_none=True)
        last_grad_norm: float = 0.0

        dtype = next(self.generator.parameters()).dtype

        for inner_epoch_idx in range(n_inner_epochs):
            # Per-sample shuffle (DiffusionNFT line 870-871).
            perm = torch.randperm(n_local, device=device)
            sx0 = x_0[perm]
            spe = prompt_embeds[perm]
            spp = pooled_embeds[perm]
            sadv = advantages[perm]
            tr_state = self._prepare_inner_epoch_tr_state(
                timesteps_all=timesteps_all,
                next_timesteps_all=next_timesteps_all,
                sample_perm=perm,
                n_ts_total=n_ts_total,
                n_train_ts=n_train_ts,
                device=device,
            )

            for start in range(0, n_local, inner_bs):
                end = start + inner_bs
                x0_b = sx0[start:end]
                pe_b = spe[start:end]
                pp_b = spp[start:end]
                adv_b = sadv[start:end]
                for j_idx in range(n_train_ts):
                    t_raw, r_raw = self._draw_training_tr(
                        state=tr_state,
                        chunk_start=start,
                        chunk_end=end,
                        j_idx=j_idx,
                        inner_epoch_idx=inner_epoch_idx,
                        device=device,
                        dtype=dtype,
                    )
                    loss, info = self._rl_loss(
                        x_0=x0_b,
                        prompt_embeds=pe_b,
                        pooled_embeds=pp_b,
                        advantages=adv_b,
                        t_raw=t_raw,
                        r_raw=r_raw,
                    )
                    # Scale loss for grad accumulation; gradients accumulate
                    # across `effective_grad_accum_steps` backward calls.
                    (loss / float(effective_grad_accum_steps)).backward()
                    current_accumulated_steps += 1
                    n_backward += 1

                    for k, v in info.items():
                        accumulator[k] = accumulator.get(k, 0.0) + float(v)

                    if current_accumulated_steps % effective_grad_accum_steps == 0:
                        last_grad_norm = self._compute_grad_norm(self.generator)
                        if max_grad_norm > 0:
                            self._clip_grad_norm(self.generator, max_grad_norm)
                        self.optimizers["generator"].step()
                        self.schedulers["generator"].step()
                        self.optimizers["generator"].zero_grad(set_to_none=True)

                        # EMA per opt step (DiffusionNFT line 1088-1092).
                        # Uses the warmup-aware decay (matches
                        # EMAModuleWrapper.get_current_decay).
                        if (
                            self.generator_ema is not None
                            and self.global_step % self.config.train.ema_every == 0
                        ):
                            self.ema_update(
                                self.generator, self.generator_ema,
                                self._current_ema_decay(),
                            )

                        # ``global_step`` increments per optimizer step (DiffusionNFT
                        # convention; drives ``return_decay(global_step, ...)``).
                        self.global_step += 1
                        n_opt_steps += 1
                        accumulator["grad_norm"] = (
                            accumulator.get("grad_norm", 0.0) + float(last_grad_norm)
                        )

        if n_backward == 0:
            return {}
        averaged = {k: v / n_backward for k, v in accumulator.items()}
        if n_opt_steps > 0:
            averaged["grad_norm"] = accumulator.get("grad_norm", 0.0) / n_opt_steps
        averaged["lr"] = self.schedulers["generator"].get_last_lr()[0]
        averaged["inner_backward_steps"] = float(n_backward)
        averaged["inner_opt_steps"] = float(n_opt_steps)
        averaged["num_train_timesteps"] = float(n_train_ts)
        averaged["effective_grad_accum_steps"] = float(effective_grad_accum_steps)
        return averaged

    # ==================================================================
    # One outer RL epoch
    # ==================================================================

    def _run_rl_epoch(
        self,
        sample_pack: Optional[dict[str, Any]] = None,
        preview: Optional[tuple[Any, list[str]]] = None,
    ) -> dict[str, dict[str, float]]:
        """One RL epoch: sampling + inner update + old_model decay.

        ``sample_pack`` lets :meth:`_train_loop` roll out once per epoch and hand
        the same pack to :meth:`_run_aux_epoch` first, so the aux network is
        fitted to exactly the samples this update then scores. ``None`` rolls it
        out here, which is what a caller with no aux loop wants.

        Returns a metrics dict whose top-level keys are wandb section names:

        - ``rl_reward``     -- reward summary + DiffusionNFT stat-tracker stats
        - ``rl_advantage``  -- advantage / r_hat distribution
        - ``rl_loss``       -- per-iteration policy / KL losses
        - ``rl_train``      -- optimization counters (lr, grad_norm, decay, ...)
        - ``rl_diagnostic`` -- sanity-check signals (x0 norm, old/gen deviation,
          plus :attr:`_EXTRA_DIAGNOSTIC_KEYS`)

        The base trainer's :meth:`_log_step` automatically adds a ``profile``
        section with GPU memory + timer phases.
        """
        cfg = self._rl_cfg()
        active = self._rl_epoch >= int(cfg.rl_start_epoch)

        if sample_pack is None:
            with self.timer.measure("rl_sampling"):
                sample_pack, preview = self._rl_sampling_phase(
                    return_intermediate=getattr(self._rl_cfg(), "reuse_x_t", False)
                )

        # After sampling either way: _rl_sampling_phase and the aux epoch both
        # leave the generator in eval, and _rl_train_phase needs it in train.
        self.generator.train()
        # ``is not None``, not ``hasattr``: __init__ sets aux_v_model = None, so
        # hasattr is True on every trainer and would call None.eval().
        if active and getattr(self, "aux_v_model", None) is not None:
            self.aux_v_model.eval()

        # Build reward / advantage sections from the sampling phase output.
        # Use the global (cross-rank gathered) rewards/per_reward for stats so
        # wandb reports the same value on every rank; the per-rank tensors are
        # local slices and would otherwise differ.
        gathered_rewards_all = self._gather_local_tensor(sample_pack["rewards"])
        reward_section: dict[str, float] = {
            "reward_mean": float(gathered_rewards_all.mean().item()),
            "reward_std": float(gathered_rewards_all.std().item()),
            "reward_min": float(gathered_rewards_all.min().item()),
            "reward_max": float(gathered_rewards_all.max().item()),
        }
        # Per-reward breakdown (DiffusionNFT logs each scorer separately).
        for name, vals in sample_pack.get("per_reward", {}).items():
            gathered = self._gather_local_tensor(vals.to(sample_pack["rewards"].device))
            reward_section[f"{name}_mean"] = float(gathered.mean().item())
            reward_section[f"{name}_std"] = float(gathered.std().item())
            reward_section[f"{name}_min"] = float(gathered.min().item())
            reward_section[f"{name}_max"] = float(gathered.max().item())
        reward_section.update(sample_pack.get("reward_stats", {}))

        advantage_section: dict[str, float] = {
            "advantage_mean": float(sample_pack["advantages"].mean().item()),
            "advantage_std": float(sample_pack["advantages"].std().item()),
            "advantage_abs_mean": float(sample_pack["advantages"].abs().mean().item()),
        }

        loss_section: dict[str, float] = {}
        train_section: dict[str, float] = {}
        diagnostic_section: dict[str, float] = {
            "n_samples_local": float(sample_pack["x_0"].shape[0]),
        }

        if active:
            with self.timer.measure("rl_update"):
                upd = self._rl_train_phase(sample_pack)

            # Dispatch upd entries into the section dicts (``affine_*`` are the
            # mean weights from :meth:`decoupled_rl_loss`).
            for k in ("loss_policy", "loss_unscaled", "loss_kl",
                      "loss_total", "affine_coeff_mean",
                      "affine_adv_coeff_mean") + self._EXTRA_LOSS_KEYS:
                if k in upd:
                    loss_section[k] = float(upd[k])
            for k in ("lr", "grad_norm", "inner_backward_steps",
                      "inner_opt_steps", "num_train_timesteps",
                      "effective_grad_accum_steps", "t_mean"):
                if k in upd:
                    train_section[k] = float(upd[k])
            if "r_hat_mean" in upd:
                advantage_section["r_hat_mean"] = float(upd["r_hat_mean"])
            for k in ("x0_norm_mean", "old_deviate") + self._EXTRA_DIAGNOSTIC_KEYS:
                if k in upd:
                    diagnostic_section[k] = float(upd[k])

            with self.timer.measure("rl_old_decay"):
                decay_val = self._decay_old_model()
            train_section["old_decay"] = float(decay_val)

        metrics: dict[str, dict[str, float]] = {
            "rl_reward": reward_section,
            "rl_advantage": advantage_section,
        }
        if loss_section:
            metrics["rl_loss"] = loss_section
        if train_section:
            metrics["rl_train"] = train_section
        if diagnostic_section:
            metrics["rl_diagnostic"] = diagnostic_section

        # Optional preview image logging.
        if (
            cfg.log_sample_images > 0
            and is_main_process()
            and self.wandb_logger is not None
            and preview is not None
        ):
            imgs, prompts_preview = preview
            n = min(int(cfg.log_sample_images), imgs.shape[0])
            try:
                from PIL import Image as _PIL
                pil_imgs = [
                    _PIL.fromarray(
                        (img.permute(1, 2, 0) * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
                    )
                    for img in imgs[:n]
                ]
                self.wandb_logger.log_images(
                    images=pil_imgs,
                    captions=prompts_preview[:n],
                    step=self.global_step,
                    key="media/rl_samples",
                )
            except Exception as e:
                logger.warning(f"RL sample image logging failed: {e}")

        return metrics

    # ==================================================================
    # Aux v-predictor epoch and main training loop (epoch-based, aligned with
    # DiffusionNFT)
    # ==================================================================

    @torch.no_grad()
    def _aux_sampling_phase(
        self, shared_pack: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Assemble ``max_aux_sample`` unscored rollouts on this rank.

        Reuses ``shared_pack`` -- the policy epoch's already-rolled-out samples
        -- as far as it goes, and only rolls out the shortfall. Rollout is by far
        the most expensive thing in either loop, and both loops sample from the
        same model, so one pack serves both. With ``max_aux_sample <=
        num_batches_per_epoch * train.batch_size`` the aux epoch needs no extra
        rollouts.

        Sharing is sound because the aux loss consumes only ``x_0`` and the two
        embeddings; it never touches rewards, advantages or the rollout's
        ``(t, r)`` grid. Nor does it double-dip in any meaningful sense: the
        policy loss queries ``aux_v_model`` at a fresh ``(t, eps)`` drawn in
        ``_rl_train_phase``, whereas the fm loss trained it at an independent
        ``(t, eps)`` of its own, so the anchor is never evaluated at a point the
        aux model just fitted. Only the clean latent is common.

        A shortfall is topped up rather than tolerated because the pack size is
        what ``renoise_per_epoch`` is validated against: shrinking it silently
        would turn the config's exact-coverage guarantee into over-reuse.

        Any rollout done here uses the same sampler as the policy path, i.e.
        ``old_model`` (or ``generator_ema`` under ``rl_sample_use_ema``): one
        behaviour policy per training step, so ``x_anchor_1`` (from ``u_old``),
        ``x_anchor_2`` (from ``aux_v_model``) and the rewarded ``x_0`` all
        describe the same distribution -- which is what makes the anchor
        difference mean anything. The cost is staleness: ``aux_v_model`` tracks
        whatever ``old_model`` currently is, and that lags the generator by its
        own EMA (``rl_decay_interval``, ``decay_type``).

        Every size decision below is derived from config, hence rank-uniform.
        Required: the rollout issues FSDP collectives, so a rank that tops up a
        different number of times deadlocks the rest.
        """
        cfg = self._rl_cfg()
        target = int(cfg.max_aux_sample)

        x0_parts: list[torch.Tensor] = []
        emb_parts: list[torch.Tensor] = []
        pool_parts: list[torch.Tensor] = []
        collected = 0

        if shared_pack is not None:
            avail = int(shared_pack["x_0"].shape[0])
            take = min(target, avail)
            if take > 0:
                if take < avail:
                    # A random subset, not a prefix: the policy pack is
                    # concatenated in sampling-round order, so a prefix would
                    # hand the aux model rounds 0..k and never show it the rest.
                    # Per-rank indices may differ freely -- these are local
                    # slices and the fm loss is per-sample -- only ``take`` has
                    # to agree, and it is config-derived.
                    idx = torch.randperm(
                        avail, device=shared_pack["x_0"].device,
                    )[:take]
                    x0_parts.append(shared_pack["x_0"][idx])
                    emb_parts.append(shared_pack["prompt_embeds"][idx])
                    pool_parts.append(shared_pack["pooled_embeds"][idx])
                else:
                    x0_parts.append(shared_pack["x_0"])
                    emb_parts.append(shared_pack["prompt_embeds"])
                    pool_parts.append(shared_pack["pooled_embeds"])
                collected = take

        if collected < target:
            if self._aux_sample_iter is None:
                self._aux_sample_iter = iter(self._rl_dataloader)
            if get_rank() == 0 and not getattr(self, "_aux_topup_log_emitted", False):
                logger.info(
                    "[GeneralRL] aux pack needs %d extra rollout sample(s)/GPU "
                    "beyond the shared policy pack (%d of %d reused). Set "
                    "max_aux_sample <= num_batches_per_epoch * train.batch_size "
                    "to make the aux epoch rollout-free.",
                    target - collected, collected, target,
                )
                self._aux_topup_log_emitted = True
            while collected < target:
                # The sampler seeds off ``seed + epoch``, so a top-up round that
                # happened to equal an RL round's ``_rl_epoch * N + i`` would
                # draw the byte-identical prompt batch -- reusing the policy
                # pack's prompts at fresh noise, which is not what a top-up is
                # for. Offsetting past any reachable RL round keeps them
                # disjoint.
                self._rl_sampler.set_epoch(
                    self._AUX_SAMPLER_EPOCH_OFFSET + self._aux_sample_round
                )
                self._aux_sample_round += 1
                try:
                    prompts = next(self._aux_sample_iter)
                except StopIteration:
                    self._aux_sample_iter = iter(self._rl_dataloader)
                    prompts = next(self._aux_sample_iter)

                pack = self._one_sampling_batch(prompts, return_rewards=False)
                x0_parts.append(pack["x_0"])
                emb_parts.append(pack["prompt_embeds"])
                pool_parts.append(pack["pooled_embeds"])
                collected += pack["x_0"].shape[0]

        # Truncated to exactly ``max_aux_sample``: the rollout batch size is a
        # per-rank config constant, so every rank overshoots by the same amount
        # and the truncation stays rank-uniform.
        #
        # The single-part case is aliased rather than cat'd. It is the common one
        # (whole shared pack reused, no top-up) and ``torch.cat`` of a one-element
        # list still copies -- ~150MB of prompt embeddings per epoch for nothing.
        # Aliasing is safe because the aux path only ever reads: _aux_train_phase
        # indexes with ``[idx]`` and _aux_fm_loss's scale_noise returns new
        # tensors, so the policy pack is never written through.
        def _finalize(parts: list[torch.Tensor]) -> torch.Tensor:
            out = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
            return out if out.shape[0] == target else out[:target]

        return {
            "x_0": _finalize(x0_parts),
            "prompt_embeds": _finalize(emb_parts),
            "pooled_embeds": _finalize(pool_parts),
        }

    def _aux_fm_loss(
        self,
        x_0: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_embeds: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Plain SD3.5 rectified-flow matching loss for :attr:`aux_v_model`.

        Logit-normal t, SD3.5's static shift, linear forward noising, and
        ``target = eps - x_0`` (the rectified-flow velocity, hence the sign),
        reduced in fp32, with no timestep weighting.

        The forward goes through :func:`predict_velocity_sd35`, not the flow-map
        wrapper: ``aux_v_model`` answers ``v(x_t, t)`` and has no ``r`` input, so
        there is no second timestep to pass. CFG is off -- the model is being
        trained toward the conditional velocity, and mixing in an unconditional
        branch would regress it toward a guided field it is never queried for.
        """
        cfg = self._rl_cfg()
        device = x_0.device
        model_dtype = next(self.aux_v_model.parameters()).dtype
        T = float(self.scheduler.config.num_train_timesteps)
        batch_size = x_0.shape[0]

        t01 = compute_density_for_timestep_sampling(
            weighting_scheme="logit_normal",
            batch_size=batch_size, logit_mean=0.0, logit_std=1.0,
        ).to(device=device, dtype=model_dtype)
        timesteps = (self.flowmap_scheduler.apply_shift(t01) * T).to(device)
        # Reuses the policy path's t window rather than adding aux-specific
        # knobs: the aux model exists to answer queries on that same window.
        timesteps = timesteps.clamp(cfg.rl_min_timestep, cfg.rl_max_timestep)

        noise = self._draw_eps(x_0)
        noisy_latents = self.flowmap_scheduler.scale_noise(x_0, timesteps, noise)

        with self._autocast():
            velocity_pred = predict_velocity_sd35(
                model=self.aux_v_model,
                noisy_latents=noisy_latents,
                text_embeddings=prompt_embeds,
                timesteps=timesteps,
                pooled_prompt_embeds=pooled_embeds,
            )
        target = noise - x_0
        per_sample = (velocity_pred.float() - target.float()) ** 2
        loss = per_sample.reshape(batch_size, -1).mean(dim=-1).mean()

        return loss, {
            "loss_fm": float(loss.detach()),
            "aux_t_mean": float(timesteps.mean().item()),
        }

    def _aux_train_phase(self, sample_pack: dict[str, Any]) -> dict[str, float]:
        """``renoise_per_epoch`` flow-matching updates over one aux sample pack.

        "Renoise" is the point: the pack is rolled out once and then swept
        repeatedly, each sweep drawing fresh ``eps`` and fresh ``t`` for the same
        latents. Rollout is the expensive half, so this amortizes it.

        Sampling discipline: the pack is consumed as a cycling permutation, not
        drawn i.i.d. That is what makes coverage a guarantee rather than a
        probability -- every sample is trained on exactly once per completed
        sweep, and reuse begins only after the sweep closes. Consequences:

        * ``renoise_per_epoch >= ceil(max_aux_sample / aux_v_batch_size)`` means
          no rollout is wasted. Enforced in ``FewstepRLTrainConfig``, since a
          shortfall throws away the most expensive thing the aux epoch produces.
        * anything above that reuses samples at fresh ``(t, eps)``; the reuse
          factor is ``renoise_per_epoch * micro_bs / max_aux_sample``.

        Nothing here touches ``self.global_step``, the generator EMA, or the
        ``old_model`` decay. All three are keyed to generator optimizer steps --
        ``global_step`` is what ``_current_decay`` and ``_current_ema_decay``
        read and what ``_log_step`` uses as its x axis -- so ticking it from the
        aux loop would silently rescale the policy path's schedules.
        """
        cfg = self._rl_cfg()
        device = torch.device("cuda")

        x_0 = sample_pack["x_0"].to(device)
        prompt_embeds = sample_pack["prompt_embeds"].to(device)
        pooled_embeds = sample_pack["pooled_embeds"].to(device)
        n_local = x_0.shape[0]
        if n_local == 0:
            return {}

        # Not clamped to ``n_local``: a microbatch larger than the pack is a
        # legitimate ask, because ``_aux_fm_loss`` draws t and eps per element.
        # The same latent appearing twice in one batch therefore carries two
        # independent noise/timestep draws, not a duplicated gradient.
        micro_bs = max(1, int(cfg.aux_v_batch_size) or n_local)
        accum = max(1, int(cfg.aux_v_gradient_accum))
        n_renoise = max(1, int(cfg.renoise_per_epoch))
        max_grad_norm = float(self.config.solver.aux_v.max_grad_norm)

        # One-time alignment log: the reuse factor and the step count are what
        # the four knobs actually mean, and neither is readable off the config.
        if not getattr(self, "_aux_align_log_emitted", False):
            if get_rank() == 0:
                sweep_len = -(-n_local // micro_bs)
                logger.info(
                    "[GeneralRL] aux fm loop: pack=%d/GPU, micro_bs=%d, "
                    "renoise=%d, accum=%d -> %d sweep(s) of %d evals, "
                    "%.2f uses per sample, %d opt step(s)/epoch, "
                    "global batch/step=%d.",
                    n_local, micro_bs, n_renoise, accum,
                    n_renoise // sweep_len, sweep_len,
                    n_renoise * micro_bs / n_local, n_renoise // accum,
                    micro_bs * max(1, get_world_size()),
                )
            self._aux_align_log_emitted = True

        opt = self.optimizers["aux_v"]
        sched = self.schedulers["aux_v"]
        opt.zero_grad(set_to_none=True)

        accumulator: dict[str, float] = {}
        n_backward = 0
        n_opt_steps = 0
        order = torch.randperm(n_local, device=device)
        cursor = 0

        for _ in range(n_renoise):
            # Draw exactly ``micro_bs`` indices, wrapping into a fresh
            # permutation when the current one runs out mid-batch. Slicing to
            # the end of the permutation instead would emit a short final
            # microbatch, whose per-sample gradient weight is inflated by
            # (micro_bs / actual) once the accumulation divisor is applied.
            # Wrapping keeps every batch the configured size while preserving
            # the coverage guarantee: a permutation is always consumed in full
            # before the next one starts, so one completed sweep touches every
            # sample exactly once. Reuse only begins after that.
            picks: list[torch.Tensor] = []
            remaining = micro_bs
            while remaining > 0:
                if cursor >= n_local:
                    order = torch.randperm(n_local, device=device)
                    cursor = 0
                take = min(remaining, n_local - cursor)
                picks.append(order[cursor:cursor + take])
                cursor += take
                remaining -= take
            idx = picks[0] if len(picks) == 1 else torch.cat(picks)

            loss, info = self._aux_fm_loss(
                x_0=x_0[idx],
                prompt_embeds=prompt_embeds[idx],
                pooled_embeds=pooled_embeds[idx],
            )
            (loss / float(accum)).backward()
            n_backward += 1
            for k, v in info.items():
                accumulator[k] = accumulator.get(k, 0.0) + float(v)

            if n_backward % accum == 0:
                grad_norm = self._compute_grad_norm(self.aux_v_model)
                if max_grad_norm > 0:
                    self._clip_grad_norm(self.aux_v_model, max_grad_norm)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                n_opt_steps += 1
                accumulator["grad_norm"] = (
                    accumulator.get("grad_norm", 0.0) + float(grad_norm)
                )

        # A trailing partial window is discarded rather than stepped: it would
        # apply an update scaled by (leftover / accum) instead of 1, and it
        # would carry into the next epoch's first window if left in place.
        if n_backward % accum != 0:
            opt.zero_grad(set_to_none=True)

        averaged = {k: v / n_backward for k, v in accumulator.items()}
        if n_opt_steps > 0:
            averaged["grad_norm"] = accumulator.get("grad_norm", 0.0) / n_opt_steps
        averaged["lr"] = sched.get_last_lr()[0]
        averaged["aux_backward_steps"] = float(n_backward)
        averaged["aux_opt_steps"] = float(n_opt_steps)
        averaged["aux_micro_batch_size"] = float(micro_bs)
        return averaged

    def _run_aux_epoch(
        self, shared_pack: Optional[dict[str, Any]] = None,
    ) -> dict[str, dict[str, float]]:
        """Aux v-predictor epoch, run before the policy update.

        ``shared_pack`` is the policy epoch's rollout, handed over so the aux
        loop does not pay for a second one; see :meth:`_aux_sampling_phase`. It
        must be the pack the policy update will then train on, and this method
        must run before that update: trainers whose anchor reads
        ``aux_v_model`` (e.g. DMDNFTTrainer) then query a model already fitted
        to the samples ``_rl_loss`` is about to score. ``None`` rolls out from
        scratch.

        A no-op returning ``{}`` unless ``setup_models`` actually built an
        ``aux_v_model``. Two reasons it is written this way:

        * ``_train_loop`` calls this unconditionally on every epoch, so raising
          here would abort every run of every trainer -- including those that
          have no aux network at all.
        * the guard is ``is not None``, not ``hasattr``: ``__init__`` sets
          ``self.aux_v_model = None``, so ``hasattr`` is always True and would
          send every trainer into ``None.train()``.

        Both the predicate and the start-epoch gate are config-level, hence
        rank-uniform -- required, because the phases below run FSDP forwards and
        a rank that skips them deadlocks the rest.
        """
        if self.aux_v_model is None:
            return {}
        cfg = self._rl_cfg()
        if self._rl_epoch < int(cfg.rl_start_epoch):
            return {}

        # Any top-up rollout reads self.generator (via old_model / EMA) and the
        # fm loss backprops through aux_v_model only, so the two mode flags are
        # swapped for the duration and restored on the way out. The policy epoch
        # calls generator.train() itself, but not depending on call order is
        # cheaper than depending on it.
        self.generator.eval()
        self.aux_v_model.train()
        try:
            with self.timer.measure("aux_sampling"):
                sample_pack = self._aux_sampling_phase(shared_pack)
            with self.timer.measure("aux_update"):
                upd = self._aux_train_phase(sample_pack)
        finally:
            self.aux_v_model.eval()
            self.generator.train()

        if not upd:
            return {}
        train_section = {
            k: float(upd[k])
            for k in ("loss_fm", "lr", "grad_norm",
                      "aux_backward_steps", "aux_opt_steps")
            if k in upd
        }
        diagnostic_section = {
            k: float(upd[k])
            for k in ("aux_t_mean", "aux_micro_batch_size")
            if k in upd
        }
        diagnostic_section["aux_n_samples_local"] = float(
            sample_pack["x_0"].shape[0]
        )
        return {"aux_train": train_section, "aux_diagnostic": diagnostic_section}

    def train_step(self, batch: Any) -> dict[str, dict[str, float]]:
        """Not used -- this trainer overrides :meth:`_train_loop`."""
        raise NotImplementedError(
            f"{type(self).__name__} uses a _train_loop() override"
        )

    def _train_loop(self) -> None:
        """Epoch-based loop following DiffusionNFT's outer-loop structure."""
        cfg = self._rl_cfg()
        train_cfg = self.config.train

        # Build K-repeat dataloader / sampler / iterator.
        self._setup_rl_dataloader()

        # Restore RNG state if resuming.
        self._restore_rng_state()

        if is_main_process():
            logger.info(
                f"Starting RL training ({self.TRAINER_KEY}): "
                f"num_epochs={cfg.num_epochs}, "
                f"num_batches_per_epoch={cfg.num_batches_per_epoch}, "
                f"num_inner_epochs={cfg.num_inner_epochs}, "
                f"K={cfg.num_image_per_prompt}, "
                f"per_rank_bs={train_cfg.batch_size}"
            )

        while self._rl_epoch < int(cfg.num_epochs):
            # Eval at epoch boundary.
            eval_cfg = self.config.eval
            if (
                eval_cfg.enabled
                and eval_cfg.eval_interval > 0
                and self._rl_epoch % eval_cfg.eval_interval == 0
            ):
                with self.timer.measure("eval"):
                    self._evaluate()

            # Save at epoch boundary (skip step 0).
            if (
                train_cfg.save_interval > 0
                and self._rl_epoch % train_cfg.save_interval == 0
                and self._rl_epoch > 0
            ):
                self._save_checkpoint()

            with self.timer.measure("epoch"):
                # One rollout per epoch, shared by the aux loop and the policy
                # update: both sample the same behaviour policy.
                # (_rl_sampling_phase puts the generator in eval itself.)
                with self.timer.measure("rl_sampling"):
                    sample_pack, preview = self._rl_sampling_phase(
                        return_intermediate=getattr(self._rl_cfg(), "reuse_x_t", False)
                    )

                # Aux epoch first: an anchor that reads aux_v_model in _rl_loss
                # needs it fitted to this pack before the policy update scores
                # it. ``{}`` when there is no aux_v_model. Its sections are
                # merged into the policy metrics and logged by the same
                # _log_step call; whether they exist is config-level, so the
                # key set is identical on every rank (_reduce_metrics
                # all-reduces per key with no reconciliation).
                aux_metrics = self._run_aux_epoch(sample_pack)
                metrics = self._run_rl_epoch(sample_pack, preview)
                for section, vals in (aux_metrics or {}).items():
                    metrics.setdefault(section, {}).update(vals)

            # Log epoch summary.
            if self._rl_epoch % train_cfg.log_interval == 0:
                self._log_step(metrics)

            self._rl_epoch += 1

        logger.info(
            f"RL training complete: epoch={self._rl_epoch}, "
            f"global_step={self.global_step}"
        )
        self._save_checkpoint()
        self._save_final_transformer()

    # ==================================================================
    # Gradient-norm helpers
    # ==================================================================

    @staticmethod
    def _compute_grad_norm(model: nn.Module) -> float:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        if isinstance(model, FSDP):
            total = torch.zeros((), device=torch.cuda.current_device())
            for p in model.parameters():
                if p.grad is not None:
                    total = total + p.grad.detach().float().norm() ** 2
            total = total.clone()
            if dist.is_initialized():
                dist.all_reduce(total, op=dist.ReduceOp.SUM)
            return float(total.sqrt().item())
        total = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total += float(p.grad.detach().float().norm().item()) ** 2
        return total ** 0.5

    @staticmethod
    def _clip_grad_norm(model: nn.Module, max_norm: float) -> None:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        if isinstance(model, FSDP):
            model.clip_grad_norm_(max_norm)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

    # ==================================================================
    # Checkpoint hooks: outer-loop counters, per-rank stat tracker,
    # and trainer provenance (the saved config.yaml does not record --trainer).
    # ==================================================================

    def _get_extra_checkpoint_state(self) -> dict[str, Any]:
        extra: dict[str, Any] = super()._get_extra_checkpoint_state()
        extra["rl_epoch"] = self._rl_epoch
        # Without this the aux prompt stream restarts at round 0 on resume and
        # replays the batches the run already trained on.
        extra["aux_sample_round"] = self._aux_sample_round
        if self._stat_tracker is not None:
            extra["rl_stat_tracker"] = {
                "rank": get_rank(),
                "state": self._stat_tracker.state_dict(),
            }
        extra["trainer_key"] = self.TRAINER_KEY
        # Numerical settings, recorded for provenance; ``cd_out_dtype`` must
        # match across runs that are compared (see the class attribute).
        extra["rl_work_dtype"] = str(self._rl_work_dtype)
        extra["cd_out_dtype"] = str(self._cd_out_dtype)
        return extra

    def _restore_extra_checkpoint_state(self, extra: dict[str, Any]) -> None:
        super()._restore_extra_checkpoint_state(extra)
        # Restore rl_epoch before anything else: skipping it silently restarts
        # the epoch counter at 0 while global_step is N, which breaks the
        # num_epochs stop key.
        self._rl_epoch = int(extra.get("rl_epoch", 0))
        self._aux_sample_round = int(extra.get("aux_sample_round", 0))
        if self._stat_tracker is not None:
            st = extra.get("rl_stat_tracker")
            if st:
                self._stat_tracker.load_state_dict(st.get("state", {}))
        saved = extra.get("trainer_key")
        if saved is not None and saved != self.TRAINER_KEY:
            raise ValueError(
                f"checkpoint was written by trainer {saved!r} but this is "
                f"{self.TRAINER_KEY!r}. Resuming across trainers silently changes the "
                "experiment. Pass a matching --trainer, or start fresh."
            )

    def _save_final_transformer(self) -> None:
        """Base export (the generator) plus ``aux_v_model`` alongside it.

        The base writes ``output_dir/transformer`` from generator_ema-or-generator
        and knows nothing about ``aux_v_model``, so without this the one model
        :meth:`_run_aux_epoch` actually trains reaches disk only inside
        ``checkpoint-*/`` and never in a ``from_pretrained``-loadable form. It
        goes to a separate directory rather than replacing the generator export:
        both are wanted, and the aux model is not a drop-in for the generator
        (no flow-map wrapper, no LoRA, single-timestep ``v(x_t, t)``).

        Always full-weight diffusers format -- ``setup_models`` gives the aux
        model neither a LoRA adapter nor the flow-map delta_embedder, so there
        is no peft branch to take.
        """
        super()._save_final_transformer()
        if self.aux_v_model is None:
            return

        from fewsteprl.utils.checkpoint import (
            _gather_fsdp_state_dict,
            _save_diffusers_transformer,
        )

        # Every rank participates in the gather; only rank 0 writes.
        state_dict = _gather_fsdp_state_dict(self.aux_v_model)
        if is_main_process():
            aux_dir = os.path.join(self.config.output_dir, "aux_v_transformer")
            _save_diffusers_transformer(
                state_dict=state_dict,
                config_path=self.config.model.pretrained_path,
                save_dir=aux_dir,
            )
            logger.info(f"Final aux_v_model saved: {aux_dir}")
        if dist.is_initialized():
            from fewsteprl.parallel.utils import barrier
            barrier()

    def _get_lora_model_names(self) -> set[str]:
        names = super()._get_lora_model_names()
        if self.config.model.generator_lora.enabled:
            if self.old_model is not None:
                names.add("old_model")
            if self.ref_u_model is not None:
                names.add("ref_u_model")
        return names

    def _get_lora_configs(self) -> dict[str, Any]:
        cfgs = super()._get_lora_configs()
        if self.config.model.generator_lora.enabled:
            if self.old_model is not None:
                cfgs["old_model"] = self.config.model.generator_lora
            if self.ref_u_model is not None:
                cfgs["ref_u_model"] = self.config.model.generator_lora
        return cfgs
