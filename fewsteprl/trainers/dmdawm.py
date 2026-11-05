"""DMDAWM: advantage-weighted x0 matching with an aux v-predictor reward baseline.

:class:`DMDAWMTrainer` is to :class:`fewsteprl.trainers.meanflowawm.MeanFlowAWMTrainer`
what :class:`fewsteprl.trainers.dmdnft.DMDNFTTrainer` is to
:class:`fewsteprl.trainers.meanflownft.MeanFlowNFTTrainer`. :attr:`aux_v_model`
is a one-time v-predictor trained online by a plain flow-matching loss on the
generator's own rollouts (:meth:`GeneralRLTrainer._run_aux_epoch`), so
``v_fake`` is an amortized estimate of the marginal ``E[eps - x_0 | x_t, t]``
over the rollout distribution -- the "fake score" of the generator's output
distribution. Compared with MeanFlowAWMTrainer:

  * No central difference in the policy term. The prediction and the ratio's
    anchor are ``x_0_pred = x_t - t * u_theta`` and
    ``x_anchor_2 = x_t - t * u_old``, so S is absent from both sides of
    ``pred - anchor`` (as in :class:`~fewsteprl.trainers.meanflowawm.AWMTrainer`).
    The CD runs only for the v-mode KL target.
  * ``v_fake`` supplies the reward term's surrogate-gradient center
    ``x_probe = x_t - t * v_fake``, replacing AWM's ``x_pred.detach()``. This is
    the base decoupled form's ``x_neg`` slot (the one AWM discards), so the
    gradient is the base's ``-(2 W / N) omega (x_0 - x_probe)``, i.e.

        dL/du_theta = ratio * gate * adv * (v_fake - v_tgt) / N,
        v_tgt = eps - x_0,

    still t-independent under ``sigma = t``.

Where AWM regresses ``u_theta`` toward ``v_tgt - S`` (a self-attracting fixed
point), this applies an advantage-weighted DMD-style score difference: descent
moves ``u_theta`` along ``+adv * (v_tgt - v_fake)``. The intent is for the
amortized marginal to act as a per-sample baseline, i.e. a learned control
variate: since ``E[adv] = 0`` within a group, a baseline leaves the mean update
unchanged while reducing its per-sample variance.

Why ``v_fake`` stays out of the ratio
-------------------------------------
AWM's ratio reference has to lag the policy: ``old_model`` is an EMA at
``old_decay``, and that lag is what makes the PPO clip a trust region at all.
An aux network trained to track the generator does not lag, so the ratio keeps
``old_model`` (exactly 1.0 at zero drift) and ``v_fake`` only moves the reward
term's center.

What dropping the CD changes
----------------------------
Keep the aux center but put S on both sides (``x_0_pred = x_t - t(u_theta + S)``,
``x_anchor_2 = x_t - t(u_old + S)``). With ``x_t = (1-t)x_0 + t eps`` and
``sigma = t`` the ratio exponent is exactly::

    log_rho = <du, v_tgt - ubar - S>_sp,
      du = u_theta - u_old,  ubar = (u_theta + u_old)/2,
      <a,b>_sp = spatial mean of a*b

Every power of ``t`` cancels (below the ``_SIGMA_MIN`` clamp the exponent is
scaled down by ``t**2/sigma**2``). So that objective depends on S only through
one scalar per sample, ``-<du, S>_sp`` in ``log_rho``: the gradient direction
``ratio * gate * adv * (v_fake - v_tgt)`` contains no S, and dropping S, as this
class does, can only rescale it, by ``exp(+<du, S>_sp)``, and flip the PPO gate
where that crosses ``clip_range``. Without S the ratio measures ``u_theta`` vs
``u_old`` drift alone and tends to stay close to 1, so the clip seldom binds:
watch ``clipfrac`` / ``approx_kl``, and lower ``clip_range`` if the trust region
should engage. Skipping the CD also saves two ``old_model`` forwards per step.

TTUR: the aux network's cadence is ``renoise_per_epoch / aux_v_gradient_accum``
optimizer steps per epoch against the generator's one step per epoch (with the
provided configs), so ``renoise_per_epoch=5, aux_v_gradient_accum=1`` is
TTUR = 5.

Not DMD despite the name, same caveat as DMDNFTTrainer: ``v_fake`` only
supplies a stop-grad reference point (the reward term's probe center); there is
no ``real_v_model`` and no backprop through the rollout.

Inherits only :class:`GeneralRLTrainer`; the AWM kernel it needs
(``decoupled_weights`` / ``reward_augmentation`` / ``_s_term`` / the MeanFlow
lift) is re-implemented here rather than inherited from MeanFlowAWMTrainer.
"""

from __future__ import annotations

from typing import Optional

import torch

from fewsteprl.trainers.rl_general_base import AnchorInputs, GeneralRLTrainer


class DMDAWMTrainer(GeneralRLTrainer):
    """Advantage-weighted x0 matching with an ``old_model`` ratio and the aux
    marginal as the reward term's center; no central difference.

      * ``x_anchor_1 = None`` -- no self-regression term.
      * ``x_anchor_2 = x_t - t * u_old`` (:meth:`_compute_x_anchor_2`): the
        ratio compares ``u_theta`` with ``u_old``, so the trust region keeps
        its EMA lag and is exactly 1.0 at zero drift. The prediction is
        ``x_t - t * u_theta``, so S is absent from both sides of
        ``pred - anchor``.
      * ``v_fake`` supplies the reward term's surrogate-gradient center
        (:attr:`_AUX_AS_RWD_CENTER`), making the aux network a per-sample
        baseline / learned control variate.

    The resulting gradient::

        dL/du_theta = ratio * gate * adv * (v_fake - v_tgt) / N,
        ratio = exp((mse(x_anchor_2, x_0) - mse(x_0_pred, x_0)) / (2 sigma**2)),
        v_tgt = eps - x_0,  sigma = t

    -- t-independent, and still AWM's clipped objective; only the direction's
    origin moves from the policy's own prediction to the learned marginal.

    The reward direction ``x_0 - x_probe`` is not driven to zero by the
    policy's own fit (the aux marginal is not the per-sample prediction), so
    the advantage rather than the residual sets the step size and the PPO clip
    is what bounds it: ``clipfrac`` / ``approx_kl`` are the metrics to watch.
    ``rwd_dir_abs_mean`` logs that magnitude.
    """

    TRAINER_KEY = "dmdawm_rl"

    #: Whether S enters the prediction / the ratio's anchor. Both False: the
    #: policy term never uses S. Informational only -- no code reads these
    #: flags, and :meth:`_rl_loss` builds both without S unconditionally. S
    #: must go on both sides or neither; on one side only it is left
    #: uncancelled in ``pred - anchor``.
    _PRED_USES_S: bool = False
    _ANCHOR_USES_S: bool = False

    # Stage 1 KL weighting, as in MeanFlowAWMTrainer (see the rationale there).
    _kl_use_stage1_weighting: bool = True
    _kl_weight_type: str = "beta08"

    #: MeanFlowAWMTrainer's panel plus ``rwd_dir_abs_mean``. ``s_x0_abs_mean``
    #: has a different meaning here: there is no S, so it logs
    #: ``|x_0_pred - x_probe|``, the gap between the generator's per-sample x0
    #: prediction and the aux marginal. A class constant, hence rank-uniform
    #: (see :meth:`BaseTrainer._reduce_metrics`).
    _EXTRA_DIAGNOSTIC_KEYS: tuple[str, ...] = (
        "r_mean",
        "sigma_mean",
        "s_x0_abs_mean",
        "rwd_dir_abs_mean",
        "ratio_mean",
        "ratio_max_dev",
        "clipfrac",
        "approx_kl",
    )

    #: Empty: this class replays nothing. It re-noises ``x_0`` at a fresh
    #: ``(t, r)`` in the base's ``_rl_train_phase`` and runs with
    #: ``reuse_x_t=false`` / ``rollout_type=ode``.
    _REPLAY_KEYS: tuple[str, ...] = ()

    #: ``v_fake`` supplies the reward term's surrogate-gradient center rather
    #: than the ratio's reference. ``False`` falls back to AWM's
    #: ``x_pred.detach()`` center (the aux forward still runs, for the
    #: diagnostics). Class-level, hence rank-uniform.
    _AUX_AS_RWD_CENTER: bool = True

    #: Floor on the ratio temperature ``sigma = t`` -- see MeanFlowAWMTrainer for
    #: the derivation; it protects the logged loss, not the gradient.
    _SIGMA_MIN: float = 1e-2

    #: Per-step stash for the reward term's center (the aux marginal), written
    #: and cleared inside :meth:`_rl_loss` and read by
    #: :meth:`reward_augmentation`. Needed because
    #: :meth:`GeneralRLTrainer.decoupled_rl_loss` forwards a single
    #: ``x_anchor_2`` to both the ratio and the reward term, and this class
    #: needs two different tensors there, so the second travels on the
    #: instance. Declared at class level so reading it is never an
    #: AttributeError.
    _rwd_center: Optional[torch.Tensor] = None

    def setup_models(self) -> None:
        """Guard ``aux_v_enabled`` before the base builds anything.

        ``aux_v_model`` supplies the reward term's probe center (see
        :meth:`_aux_v_x0`), so without it there is no loss. Checked here (not
        per step) because it is fixed at config-load time; before ``super()`` so
        it fires without loading a transformer. Mirrors DMDNFTTrainer.
        """
        if not self._rl_cfg().aux_v_enabled:
            raise ValueError(
                f"{type(self).__name__} takes the reward term's probe center from "
                "aux_v_model, but fewstep_rl.aux_v_enabled=false, so setup_models "
                "would never build that model and the first policy step would fail "
                "-- after a full rollout and reward-scoring phase. Set "
                "fewstep_rl.aux_v_enabled=true, or use the meanflowawm_rl trainer, "
                "which needs no aux network."
            )
        super().setup_models()

    # ==================================================================
    # AWM kernel: MeanFlow identity, ratio weights and reward term.
    # ==================================================================
    def _s_term(
        self,
        t_cd: torch.Tensor,
        r_cd: torch.Tensor,
        dF_dt: Optional[torch.Tensor],
        like: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """``S = (t - r) * dF/dt`` in ``like``'s dtype, or None when ``dF_dt`` is None.

        Unlike :meth:`MeanFlowAWMTrainer._s_term`, a None ``dF_dt`` with
        ``r != t`` is not an error here: the CD is skipped on purpose while the
        ``(t, r)`` sampler still draws random ``r`` (setting
        ``diffusion_ratio=1.0`` instead would also collapse the ``r``
        coverage), so S is defined to be zero rather than arithmetically zero.
        Same opt-out as ``disable_cd`` in :meth:`MeanFlowReFLTrainer._s_term`.
        The dtype check on the ``dF_dt``-present path (the v-mode KL target)
        still applies.
        """
        if dF_dt is None:
            return None
        wdt = like.dtype
        if wdt not in (torch.float32, torch.float64):
            raise TypeError(
                f"_s_term got a {wdt} operand; S must be folded in fp32/fp64."
            )
        dt_cd = (t_cd - r_cd).view(like.shape[0], *([1] * (like.ndim - 1))).to(wdt)
        return dt_cd * dF_dt.to(wdt)

    def _instantaneous_velocity_from_u(
        self,
        u_old: torch.Tensor,
        t_cd: torch.Tensor,
        r_cd: torch.Tensor,
        dF_dt: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``v(x_t, t) = u(x_t, r, t) + (t - r) * dF/dt`` -- identity solved for v."""
        s = self._s_term(t_cd, r_cd, dF_dt, like=u_old)
        return u_old if s is None else u_old + s

    def _average_velocity_from_v(
        self,
        v_inst: torch.Tensor,
        t_cd: torch.Tensor,
        r_cd: torch.Tensor,
        dF_dt: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``u(x_t, r, t) = v(x_t, t) - (t - r) * dF/dt`` -- identity solved for u.

        Only the v-mode KL target uses this here.
        """
        s = self._s_term(t_cd, r_cd, dF_dt, like=v_inst)
        return v_inst if s is None else v_inst - s

    @torch.no_grad()
    def _compute_x_anchor_1(self, inputs: AnchorInputs) -> Optional[torch.Tensor]:  # noqa: ARG002
        """None -- no self-regression term (reward term only), same as AWM."""
        return None


    @torch.no_grad()
    def _aux_v_x0(self, inputs: AnchorInputs) -> torch.Tensor:
        """``x_t - t * v_fake(x_t, t)`` -- the aux network's amortized x0 marginal.

        Not the ratio's reference: it is the reward term's surrogate-gradient
        center, routed to :meth:`reward_augmentation` through the
        ``_rwd_center`` stash (the ratio keeps ``old_model`` as its lagging
        reference; see the module docstring).

        ``model_u_pred`` stays False: aux_v_model is already instantaneous by
        construction, so the u -> v lift must not run; ``r_timesteps`` is passed
        only to satisfy the signature and is dropped by that branch.
        """
        aux = self.aux_v_model
        if aux is None:
            raise RuntimeError(
                "DMDAWMTrainer needs aux_v_model; setup_models only builds it when "
                "fewstep_rl.aux_v_enabled is set."
            )
        # Back to the network's dtype for the forward. Exact, not lossy:
        # AnchorInputs.x_t is the widened copy of the bf16 x_t that _rl_loss
        # built, so narrowing recovers the identical bytes the other velocity
        # queries in this step were evaluated at.
        v_fake = self._predict_instantaneous_velocity(
            model=aux,
            noisy_latents=inputs.x_t.to(next(aux.parameters()).dtype),
            prompt_embeds=inputs.prompt_embeds,
            timesteps=inputs.t_cd,
            pooled_embeds=inputs.pooled_embeds,
            r_timesteps=inputs.r_cd,
            # CFG-free, matching _aux_fm_loss: aux_v_model is trained on the
            # conditional velocity only, so guiding here queries a field it has
            # never seen.
            guidance_scale=1.0,
        )
        # Cast, not check. Nothing is folded here -- v_fake is one raw network
        # output -- so bf16 is the correct dtype to receive and the only
        # requirement is that the x0 algebra below runs in work dtype.
        v_fake = v_fake.to(inputs.x_t.dtype)
        # Explicit raise, not assert: assert vanishes under `python -O`.
        if v_fake.requires_grad:
            raise RuntimeError(
                "v_fake carries grad; it is a stop-grad reference point and "
                "aux_v_model has its own optimizer -- a gradient path from the "
                "policy loss into it would train it on the wrong objective."
            )
        return inputs.x_t - inputs.t_normed * v_fake

    @torch.no_grad()
    def _compute_x_anchor_2(self, inputs: AnchorInputs) -> torch.Tensor:
        """``x_anchor_2 = x_t - t * u_old`` -- pi_old's x0 prediction, without S.

        :meth:`_rl_loss` passes ``dF_dt=None``, so
        :meth:`_instantaneous_velocity_from_u` returns ``u_old`` unchanged. The
        reference stays ``old_model``, so the ratio keeps its EMA lag and is
        exactly 1.0 at zero drift; the prediction omits S as well, so S is
        absent from both sides of ``pred - anchor``.
        """
        v_inst = self._instantaneous_velocity_from_u(
            inputs.u_old, inputs.t_cd, inputs.r_cd, inputs.dF_dt,
        )
        if v_inst.requires_grad:
            raise RuntimeError(
                "x_anchor_2 carries grad; it is the ratio's stop-grad reference "
                "and old_model is not trained by the policy loss."
            )
        return inputs.x_t - inputs.t_normed * v_inst

    @torch.no_grad()
    def decoupled_weights(
        self,
        x_0_pred_v: torch.Tensor,
        x_anchor_1: Optional[torch.Tensor],  # noqa: ARG002 - always None here
        x_anchor_2: torch.Tensor,
        x_0_rwd: torch.Tensor,
        adv: torch.Tensor,
        sigma: torch.Tensor,
        spatial: tuple[int, ...],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """AWM weights ``(ratio, ratio, omega)``, as in MeanFlowAWMTrainer.

        ``ratio = exp((mse(x_anchor_2, x_0) - mse(x_0_pred, x_0)) / (2 sigma^2))``,
        PPO gate on ``max{-A rho, -A clip(rho)}``, ``omega = gate * adv / (2
        sigma^2)``. With ``sigma = t`` the x0-space ``t^2`` cancels and the reward
        gradient is t-independent. See :meth:`MeanFlowAWMTrainer.decoupled_weights`
        for the derivation.
        """
        if sigma.ndim != 1 or sigma.shape[0] != x_0_pred_v.shape[0]:
            raise ValueError(
                f"sigma must be [B] with B == {x_0_pred_v.shape[0]}, got shape "
                f"{tuple(sigma.shape)}."
            )
        l_pred = self._per_sample_mse(x_0_pred_v, x_0_rwd, spatial)
        l_old = self._per_sample_mse(x_anchor_2, x_0_rwd, spatial)
        two_sigma2 = 2.0 * sigma**2
        ratio = torch.exp((l_old - l_pred) / two_sigma2)

        clip_range = float(self._rl_cfg().clip_range)
        unclipped = -adv * ratio
        clipped = -adv * ratio.clamp(1.0 - clip_range, 1.0 + clip_range)
        inside = (ratio > 1.0 - clip_range) & (ratio < 1.0 + clip_range)
        gate = (inside | (unclipped >= clipped)).to(ratio.dtype)

        bcast = (-1, *([1] * (x_0_pred_v.ndim - 1)))
        omega = (gate * adv / two_sigma2).view(*bcast)
        return ratio, ratio, omega

    def reward_augmentation(
        self,
        x_pred: torch.Tensor,
        x_pos: torch.Tensor,
        x_neg: torch.Tensor,  # noqa: ARG002 - center is x_pred.detach() or the aux stash
        omega: torch.Tensor,
        weight: torch.Tensor,
        spatial: tuple[int, ...],
    ) -> torch.Tensor:
        """Injects the gradient ``-weight * omega * (2/N) * (x_pos - center)``.

        ``x_target = x_pred.detach() + omega * (x_pos - center)`` so
        ``dL/dx_pred = weight*(2/N)*(x_pred - x_target) = -weight*omega*(2/N)*
        (x_pos - center)``. ``omega > 0`` pulls x_pred along ``+(x_pos - center)``.

        The center is what distinguishes this class:

          * :attr:`_AUX_AS_RWD_CENTER` False: ``center = x_pred.detach()``, as in
            MeanFlowAWMTrainer. Self-attracting: the update is a plain
            advantage-weighted regression of ``x_pred`` onto ``x_pos``.
          * True (the default): ``center = x_t - t * v_fake``, read from the
            stash set by :meth:`_rl_loss`, not from ``x_neg``:
            :meth:`GeneralRLTrainer.decoupled_rl_loss` forwards one
            ``x_anchor_2`` to both the ratio and this method, and here those
            must be different tensors (the old_model anchor and the aux
            marginal), so the second one travels on the instance.

        ``x_neg`` is unread either way; with the aux center, the stash carries
        what ``x_neg`` carries in the base's own form.
        """
        if self._AUX_AS_RWD_CENTER:
            center = self._rwd_center
            if center is None:
                raise RuntimeError(
                    f"{type(self).__name__} sets _AUX_AS_RWD_CENTER but the "
                    "aux probe center stash is empty -- reward_augmentation ran "
                    "outside the _rl_loss block that fills it. Silently falling "
                    "back to x_pred.detach() here would change the objective "
                    "without changing one logged number."
                )
            if center.shape != x_pred.shape:
                raise ValueError(
                    f"aux probe center has shape {tuple(center.shape)}, expected "
                    f"{tuple(x_pred.shape)}."
                )
        else:
            center = x_pred.detach()
        with torch.no_grad():
            norm_grad = (omega * (x_pos.double() - center.double())).to(x_pred.dtype)
        x_target = (x_pred.detach() + norm_grad).to(x_pred.dtype)
        return weight * self._per_sample_mse(x_pred, x_target, spatial)

    # ==================================================================
    # One RL step: MeanFlowAWMTrainer's _rl_loss without S, with the aux
    # marginal as the reward term's center.
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
        """One DMD-AWM step over one freshly noised query point.

        Forward noising as in MeanFlowAWMTrainer: ``x_t = (1 - t) x_0 + t eps``
        at the base's three-mode ``(t, r)``. Then

            x_0_pred   = x_t - t * u_theta     # no S
            x_anchor_2 = x_t - t * u_old       # the ratio's reference
            x_probe    = x_t - t * v_fake      # reward-term center (aux_v_model)

        The rest (ratio / omega / gate at ``sigma = t``, reward-only decoupled
        loss, u- or v-mode KL) is AWM's.
        """
        cfg = self._rl_cfg()
        dtype = x_0.dtype
        batch_size = x_0.shape[0]
        T = float(self.scheduler.config.num_train_timesteps)
        wdt = self._rl_work_dtype

        # 1. Forward noising, the same op sequence as MeanFlowAWMTrainer.
        bcast = (batch_size, *([1] * (x_0.ndim - 1)))
        t_raw = t_raw.to(device=x_0.device)
        t_normed = t_raw.to(dtype) / T
        t_broadcast = t_normed.view(bcast)
        eps = self._draw_eps(x_0)
        x_t = (1.0 - t_broadcast) * x_0 + t_broadcast * eps
        t_for_model = t_raw.to(dtype=torch.float32)
        r_for_model = r_raw.to(dtype=torch.float32) if r_raw is not None else None
        r_cd = r_for_model if r_for_model is not None else t_for_model

        _, use_v_kl = self._kl_usage()
        kl_weight = float(cfg.kl_weight)

        adv_clip_max = float(cfg.adv_clip_max)
        if adv_clip_max <= 0:
            raise ValueError(f"adv_clip_max must be > 0, got {adv_clip_max}")

        if self._shared_dF_dt is not None:
            raise RuntimeError(
                "stale _shared_dF_dt on entry to _rl_loss -- a previous step "
                "leaked its central difference (missing _post_compute_v_hook)."
            )

        # 2. Central difference, only for the v-mode KL target: the policy term
        # uses S on neither side, so the CD (two extra old_model forwards) runs
        # only when the v-mode KL is live. With use_ref_v_model=false, as in
        # the provided configs, it never runs.
        #
        # ``use_v_kl`` is a config-level predicate and therefore rank-uniform.
        # That matters: the CD issues FSDP collectives, so a data-dependent
        # skip (e.g. on torch.equal(t, r)) would deadlock. See
        # _pre_compute_v_hook's own note.
        if use_v_kl:
            self._pre_compute_v_hook(
                x_t=x_t, t_for_model=t_for_model, r_for_model=r_for_model,
                prompt_embeds=prompt_embeds, pooled_embeds=pooled_embeds, x_0=x_0,
            )
        try:
            u_theta = self._compute_u(
                self.generator, x_t, t_for_model, prompt_embeds, pooled_embeds,
                r_for_model, with_grad=True,
            )
            # old_model velocity: this is x_anchor_2's source here (the aux
            # marginal goes to the reward center instead), so it is on the
            # critical path, not merely a health metric.
            u_old = self._compute_u(
                self.old_model, x_t, t_for_model, prompt_embeds, pooled_embeds,
                r_for_model, with_grad=False,
            ).detach()

            u_theta_f = u_theta.to(wdt)
            u_old_f = u_old.to(wdt)
            x_t_f = x_t.to(wdt)
            x_0_f = x_0.to(wdt)
            t_normed_b_f = t_broadcast.to(wdt)
            spatial = tuple(range(1, x_0.ndim))


            x_0_pred_u = x_t_f - t_normed_b_f * u_theta_f      # carries grad

            anchor_inputs = AnchorInputs(
                x_t=x_t_f,
                t_normed=t_normed_b_f,
                u_old=u_old_f,
                u_theta=u_theta_f.detach(),
                dF_dt=None,
                t_cd=t_for_model,
                r_cd=r_cd,
                prompt_embeds=prompt_embeds,
                pooled_embeds=pooled_embeds,
            )
            x_anchor_1 = self._compute_x_anchor_1(anchor_inputs)
            x_anchor_2 = self._compute_x_anchor_2(anchor_inputs)

            # The aux marginal: the reward term's surrogate-gradient center, not
            # the ratio's reference. Built from the same AnchorInputs and handed
            # to reward_augmentation on the instance, because decoupled_rl_loss
            # has only one anchor slot. Cleared in the ``finally`` around
            # decoupled_rl_loss below so no later step can read it. With
            # _AUX_AS_RWD_CENTER set, reward_augmentation raises if the stash
            # is empty, so a missing stash cannot degrade silently.
            self._rwd_center = self._aux_v_x0(anchor_inputs)
            aux_gap = float((x_0_pred_u.detach() - self._rwd_center).abs().mean())

            # Logged as s_x0_abs_mean: there is no S here, so the key carries
            # the generator-vs-aux-marginal gap instead.
            s_x0_abs_mean = aux_gap

            shared_dF_dt = self._shared_dF_dt if use_v_kl else None
        finally:
            self._post_compute_v_hook()

        adv = advantages.float().clamp(-adv_clip_max, adv_clip_max)
        sigma = t_normed.to(wdt).reshape(batch_size).clamp(min=self._SIGMA_MIN)

        # The reward direction the surrogate gradient actually follows, measured
        # before the stash is cleared: |x_0 - aux marginal| with the aux center,
        # |x_0 - x_0_pred| (AWM's self-attracting residual) without it. The
        # former is not driven to zero by the policy's own fit, which is the
        # point of moving the center off the prediction.
        with torch.no_grad():
            rwd_center_d = (
                self._rwd_center if self._AUX_AS_RWD_CENTER
                else x_0_pred_u.detach()
            )
            rwd_dir_abs_mean = float((x_0_f - rwd_center_d).abs().mean())

        try:
            unscaled, form_info = self.decoupled_rl_loss(
                x_0_pred_u, x_anchor_1, x_anchor_2, x_0_f, adv, sigma, spatial,
            )
        finally:
            # Latent-sized; freed here rather than at the next step's entry so a
            # stash can never outlive the step that built it.
            self._rwd_center = None
        policy_loss = unscaled.mean()
        loss = policy_loss

        with torch.no_grad():
            clip_range = float(cfg.clip_range)
            l_pred_d = self._per_sample_mse(x_0_pred_u.detach(), x_0_f, spatial)
            l_old_d = self._per_sample_mse(x_anchor_2, x_0_f, spatial)
            log_rho = (l_old_d - l_pred_d) / (2.0 * sigma**2)
            rho_d = torch.exp(log_rho)
            info = {
                "loss_policy": float(policy_loss.detach()),
                "loss_unscaled": float(unscaled.mean().detach()),
                "t_mean": float(t_raw.mean().item()),
                "r_mean": float(r_cd.mean().item()),
                "sigma_mean": float(sigma.mean().item()),
                "advantage_mean": float(advantages.mean().item()),
                "advantage_std": float(advantages.std().item()),
                "x0_norm_mean": float((x_0_f ** 2).mean().item()),
                "old_deviate": float(((u_theta_f - u_old_f) ** 2).mean().item()),
                # |x_0_pred - aux marginal|: generator prediction vs aux marginal.
                "s_x0_abs_mean": s_x0_abs_mean,
                # |x_0 - center| of the reward term's surrogate direction.
                "rwd_dir_abs_mean": rwd_dir_abs_mean,
                "ratio_mean": float(rho_d.mean()),
                "ratio_max_dev": float((rho_d - 1.0).abs().max()),
                "clipfrac": float((rho_d - 1.0).abs().gt(clip_range).float().mean()),
                "approx_kl": float(0.5 * (log_rho**2).mean()),
                **form_info,
            }
        kl, kl_info = self.compute_kl_term(
            x_t, t_for_model, prompt_embeds, pooled_embeds, r_for_model,
            shared_dF_dt, u_theta_f, spatial,
        )
        loss = loss + kl_weight * kl
        info["loss_kl"] = float(kl.detach())
        info.update(kl_info)
        info["loss_total"] = float(loss.detach())
        return loss, info
