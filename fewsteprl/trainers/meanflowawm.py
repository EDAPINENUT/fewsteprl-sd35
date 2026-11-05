"""MeanFlowAWM: advantage-weighted x0 matching (AWM) on a two-time flow map.

:class:`fewsteprl.trainers.meanflowgrpo.MeanFlowGRPOTrainer` replays the
rollout's own ``(x_t, t) -> (x_s, s) -> (x_r, r)`` jump-and-renoise transitions
and regresses onto ``x_r``. :class:`MeanFlowAWMTrainer` does neither: it
re-noises the rewarded sample with a fresh ``eps`` at a fresh ``(t, r)`` and
asks the flow map to denoise it back to ``x_0``::

    x_t      = (1 - t) * x_0 + t * eps
    v_theta  = u_theta(x_t, t, r) + (t - r) * sg(dF/dt)
    x_0_pred = x_t - t * v_theta

so the rollout only has to supply ``x_0`` and its advantage -- no intermediate
states (hence ``fewstep_rl.reuse_x_t=false``) and no renoise (hence
``rollout_type=ode``). The inner loop is therefore the base's
``_rl_train_phase``, which draws a fresh three-mode ``(t, r)`` per iteration,
not GRPO's rollout-step replay.

Why the prediction is built from the instantaneous velocity
----------------------------------------------------------
``x_t - t * u_theta`` would only be an x0 prediction if ``u`` were instantaneous,
i.e. at ``r == t``. Converting through the MeanFlow identity first is what makes
this a MeanFlow objective rather than treating ``u`` as instantaneous. ``S`` is
detached, so ``d v_theta / d u_theta == 1`` and the gradient path is unchanged;
what changes is the residual::

    v_theta - v_tgt == u_theta - (v_tgt - S),      v_tgt = eps - x_0

i.e. ``u_theta`` regresses onto ``v_tgt - (t - r) * dF/dt`` -- the same MeanFlow
average-velocity target the v-mode KL builds
(:meth:`GeneralRLTrainer.compute_kl_term`), with the empirical flow-matching
velocity standing in for ``v_ref``.

What the likelihood ratio does and does not mean here
-----------------------------------------------------
The reward term keeps GRPO's clipped-ratio weighting, but the ratio is weaker
than GRPO's. In GRPO, ``x_r`` was genuinely drawn from ``pi_old(. | x_t)`` during
the rollout, so the ratio is a true off-policy importance weight. Here ``x_t`` is
rebuilt from a fresh ``eps`` and ``x_0`` was never drawn from
``pi_old(. | x_t)`` -- that query point never existed in the rollout. So
``ratio`` is a trust region on how far theta has moved from ``old_model`` at
synthetic query points. It still does PPO's job of bounding the step; it does not
correct a distribution mismatch, and ``approx_kl`` should be read that way.

:class:`AWMTrainer` is the S-term ablation: ``S = 0``, so ``u_theta`` regresses
onto ``v_tgt`` directly and the central difference is never computed.
"""

from __future__ import annotations

from typing import Optional

import torch

from fewsteprl.trainers.rl_general_base import AnchorInputs, GeneralRLTrainer


class MeanFlowAWMTrainer(GeneralRLTrainer):
    """Advantage-weighted x0 matching on a noise-then-denoise transition.

    ``x_anchor_1 = None`` (no self-regression term) and ``x_anchor_2`` carries
    ``old_model``'s x0 prediction so :meth:`decoupled_weights` can form the
    likelihood ratio -- the same slot usage as :class:`MeanFlowGRPOTrainer`, on
    x0 instead of ``x_r``. See the module docstring for the objective and its
    caveats.
    """

    TRAINER_KEY = "meanflowawm_rl"

    #: Apply AnyFlow Stage 1's (flow-map pretraining) two-layer loss weighting
    #: to the v-mode KL, which regresses ``u_theta`` onto the MeanFlow target
    #: ``u_tgt = v_ref - (t - r) * dF/dt``:
    #:
    #:   1. a per-sample t-weight, the ``beta08`` curve ``t*(1-t)^0.5``
    #:      normalized so ``sum(weights) == T``; it zeroes the degenerate
    #:      ``t == T`` endpoint, where a velocity target is meaningless;
    #:   2. a mode rebalance that rescales every non-diffusion (``r != t``)
    #:      sample to the global diffusion-slice mean. The ``r == 0`` slice has
    #:      the largest ``(t - r)``, hence the largest S, which can exceed the
    #:      velocity it corrects; unrebalanced, that slice dominates the batch
    #:      mean and the KL tracks S-estimation noise rather than policy
    #:      divergence.
    #:
    #: ``False`` leaves the v-mode KL a plain per-sample MSE. The u-mode KL is
    #: never weighted.
    _kl_use_stage1_weighting: bool = True

    #: Weight curve for layer 1 above. :meth:`GeneralRLTrainer._v_kl_reweight`
    #: builds its own FlowMapScheduler with this ``weight_type``, because the
    #: shared ``flowmap_scheduler`` uses ``weight_type="uniform"``, whose flat
    #: curve would make layer 1 a no-op.
    _kl_weight_type: str = "beta08"

    # No _EXTRA_LOSS_KEYS override: the kl_* keys compute_kl_term emits are
    # already registered in GeneralRLTrainer._EXTRA_LOSS_KEYS.

    #: AWM diagnostics, in place of the base's anchor-form keys:
    #: ``s_rel_median`` measures ``x_anchor_1 - x_anchor_2``, which is not
    #: defined here because this class passes ``x_anchor_1=None``.
    #: ``s_x0_abs_mean`` is kept and measures the same correction, ``|t * S|``,
    #: directly on the prediction. ``_run_rl_epoch`` logs only allowlisted
    #: keys and silently drops the rest, so every diagnostic :meth:`_rl_loss`
    #: emits must be listed here.
    #:
    #: ``clipfrac`` / ``ratio_max_dev`` show whether ``fewstep_rl.clip_range``
    #: binds: near 0 the trust region never binds, near 1 it binds everywhere
    #: and learning stalls. ``sigma_mean`` follows the ``(t, r)`` draw and sits
    #: at :attr:`_SIGMA_MIN` only on the (very rare) small-t slice.
    #:
    #: A class constant, hence rank-uniform: :meth:`BaseTrainer._reduce_metrics`
    #: all-reduces per key with no key-set reconciliation.
    _EXTRA_DIAGNOSTIC_KEYS: tuple[str, ...] = (
        "r_mean",           # mean flow-map target time
        "sigma_mean",       # post-floor ratio temperature
        "s_x0_abs_mean",    # |t * S|, the CD correction in x0 units
        "ratio_mean",       # E[rho]; 1.0 at zero drift
        "ratio_max_dev",    # max|rho - 1|; the exp-runaway detector
        "clipfrac",         # fraction of samples outside the trust region
        "approx_kl",        # 0.5*E[(log rho)^2], the PPO KL proxy
    )

    #: Empty: this class replays nothing. It re-noises ``x_0`` at a fresh
    #: ``(t, r)``, so it needs neither the rollout's intermediate states nor its
    #: timestep triples and runs with ``fewstep_rl.reuse_x_t=false``. (Only
    #: :class:`MeanFlowGRPOTrainer`'s ``_rl_train_phase`` reads this attribute.)
    _REPLAY_KEYS: tuple[str, ...] = ()

    #: Floor on the ratio temperature ``sigma = t``.
    #:
    #: With ``sigma == t`` the ``t**2`` in the x0-space residual
    #: (``x_0_pred - x_0 == t * (v_tgt - v_theta)``) cancels against the
    #: ``1 / (2 * sigma**2)`` in both the ratio's exponent and ``omega``, so the
    #: gradient reaching the network is t-independent. The floor protects the
    #: logged loss, not that cancellation: :meth:`reward_augmentation`'s loss
    #: value carries ``omega = adv / (2 * t**2)`` squared, so it grows like
    #: ``1 / t**2`` while the gradient stays flat, and unfloored ``loss_policy``
    #: / ``loss_total`` go heavy-tailed. The exponent needs no floor: ``l_pred``
    #: and ``l_old`` are both ``O(t**2)``, so its precision depends on the
    #: policy drift ``(l_old - l_pred) / l``, not on t.
    #:
    #: 1e-2 almost never binds: with the default uniform ``t`` law of
    #: :meth:`GeneralRLTrainer._sample_training_tr_three_mode`
    #: (``t01 = max(t1, t2)``) and SD3.5's ``shift=3.0``, ``t < 1e-2`` needs
    #: ``t01 < 3.36e-3``, probability ~1.1e-5. Revisit if the shift, the ``t``
    #: law or ``fewstep_rl.rl_min_timestep`` changes.
    _SIGMA_MIN: float = 1e-2

    def _s_term(
        self,
        t_cd: torch.Tensor,
        r_cd: torch.Tensor,
        dF_dt: Optional[torch.Tensor],
        like: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """``S = (t - r) * dF/dt`` in ``like``'s dtype, or None when S is exactly 0.

        The one place the sign and the units of S are decided, so the two
        directions of the MeanFlow identity -- :meth:`_instantaneous_velocity_from_u`
        (``u + S``) and :meth:`_average_velocity_from_v` (``v - S``) -- cannot
        drift apart.
        """
        if dF_dt is None:
            # Legal only when S is identically zero for every sample (the r == t
            # slice, where the CD is skipped on purpose). Anything else means the
            # shared central difference went missing.
            if not bool(torch.equal(t_cd, r_cd)):
                raise RuntimeError(
                    "dF_dt is None but r != t for some samples: the shared central "
                    "difference was not computed. Check _pre_compute_v_hook."
                )
            return None

        wdt = like.dtype
        if wdt not in (torch.float32, torch.float64):
            raise TypeError(
                f"_s_term got a {wdt} operand; S must be folded in fp32/fp64. "
                "Folding it in the network's bf16 is what destroys the "
                "cancellation that makes dev S-free."
            )
        # (t - r) in RAW timestep units, matching the CD denominator (2*eps_cd,
        # also raw), so the product is dimensionless.
        dt_cd = (t_cd - r_cd).view(
            like.shape[0], *([1] * (like.ndim - 1))
        ).to(wdt)
        return dt_cd * dF_dt.to(wdt)

    def _instantaneous_velocity_from_u(
        self, 
        u_old: torch.Tensor, 
        t_cd: torch.Tensor, 
        r_cd: torch.Tensor, 
        dF_dt: Optional[torch.Tensor]=None,
    ) -> torch.Tensor:
        """``v(x_t, t) = u(x_t, r, t) + (t - r) * dF/dt`` -- the identity solved for v.

        Called on both policies here, unlike in the anchor-form trainers. ``S``
        is a detached ``old_model`` constant either way, so applying it to the
        grad-carrying ``u_theta`` adds no gradient path -- it only moves the
        regression target from ``v_tgt`` to ``v_tgt - S``.
        """
        s = self._s_term(t_cd, r_cd, dF_dt, like=u_old)
        return u_old if s is None else u_old + s

    def _average_velocity_from_v(
        self,
        v_inst: torch.Tensor,
        t_cd: torch.Tensor,
        r_cd: torch.Tensor,
        dF_dt: Optional[torch.Tensor]=None,
    ) -> torch.Tensor:
        """``u(x_t, r, t) = v(x_t, t) - (t - r) * dF/dt`` -- the identity solved for u.

        Exact inverse of :meth:`_instantaneous_velocity_from_u`, and the MeanFlow
        regression TARGET: given a reference's instantaneous velocity at t, this
        is what the generator's average velocity over ``[r, t]`` should equal.
        Keeping the whole correction on the target side is what makes the KL a
        MeanFlow loss rather than a v-space MSE -- algebraically the same residual
        (``(u + S) - v_ref == u - (v_ref - S)``), but the gradient path then
        carries no S at all.
        """
        s = self._s_term(t_cd, r_cd, dF_dt, like=v_inst)
        return v_inst if s is None else v_inst - s

    @torch.no_grad()
    def _compute_x_anchor_1(self, inputs: AnchorInputs) -> Optional[torch.Tensor]:  # noqa: ARG002
        """None -- this class has no self-regression term.

        ``decoupled_rl_loss`` reads None as "zero that half", so the whole loss
        is the reward term. There is no policy-improvement anchor to regress
        toward here: the target is the rewarded sample itself, weighted by its
        advantage.
        """
        return None

    @torch.no_grad()
    def _compute_x_anchor_2(self, inputs: AnchorInputs) -> torch.Tensor:
        """``x_anchor_2 = x_t - t * v_old`` -- pi_old's x0 prediction.

        Same op sequence as :meth:`_rl_loss`'s ``x_0_pred_v``, on ``u_old``
        instead of ``u_theta`` and with the same shared ``S``, so the two means
        differ by exactly ``-t * (u_theta - u_old)`` and the ratio measures
        policy drift alone.
        """
        v_inst = self._instantaneous_velocity_from_u(
            inputs.u_old, inputs.t_cd, inputs.r_cd, inputs.dF_dt,
        )
        # Explicit raise, not assert: assert vanishes under `python -O`.
        if v_inst.dtype not in (torch.float32, torch.float64):
            raise TypeError(
                f"_instantaneous_velocity_from_u must return fp32/fp64, got {v_inst.dtype}. "
                "Adding S in the network's bf16 loses precision to cancellation."
            )
        if v_inst.requires_grad:
            raise RuntimeError(
                "_instantaneous_velocity_from_u must be detached; x_anchor_2 is a "
                "stop-grad probe center."
            )
        return inputs.x_t - inputs.t_normed * v_inst

    @torch.no_grad()
    def decoupled_weights(
        self,
        x_0_pred_v: torch.Tensor,
        x_anchor_1: Optional[torch.Tensor],  # noqa: ARG002 - see below: always None here
        x_anchor_2: torch.Tensor,   # pi_old's x0 prediction at this query point
        x_0_rwd: torch.Tensor,      # the rewarded sample the rollout produced
        adv: torch.Tensor,          # occupies the anchor form's ``r_hat`` slot
        sigma: torch.Tensor,        # occupies the anchor form's ``beta`` slot
        spatial: tuple[int, ...]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """AWM weights ``(ratio, ratio, omega)`` for the base's reward term.

        The anchor-form trainers read these three slots as affine probe
        coefficients; here they weight a policy-gradient loss

            L = gate * ratio * adv / (2 * sigma**2) * mse(x_0_pred_v, x_0_rwd)

        with ``ratio`` detached. ``ratio`` is the Gaussian-kernel ratio between
        the current policy and the behaviour policy at this query point, with
        means ``x_0_pred_v`` and ``x_anchor_2``. See the module docstring for why
        that is a trust region rather than an importance weight here.

        The temperature ``sigma``
        -------------------------
        ``sigma = t``, passed per-sample in the slot the anchor form uses for a
        scalar ``beta`` (which works because the base only forwards the slot to
        this method). The x0-space residual is exactly

            x_0_pred_v - x_0_rwd == t * (v_tgt - v_theta),   v_tgt = eps - x_0,

        so both per-sample losses are ``t**2 * mse(v, v_tgt)`` and the ``t**2``
        cancels against ``1 / (2 * sigma**2)``. The ratio reduces to
        ``exp((mse(v_old, v_tgt) - mse(v_theta, v_tgt)) / 2)`` and the reward
        gradient to ``ratio * adv * (u_theta - (v_tgt - S)) / N`` -- both
        t-independent. Any other temperature reintroduces a per-t scale on the
        effective learning rate. ``fewstep_rl.beta`` is therefore unread here;
        ``fewstep_rl.clip_range`` is the knob that matters instead.

        Sign of the log-ratio
        ---------------------
        ``log rho = log p_theta - log p_old = (l_old - l_pred) / (2 sigma^2)``,
        since ``log p = -mse / (2 sigma^2)``; i.e. the old loss leads. The
        reverse order yields ``1/rho``, which inverts the PPO weighting (a
        sample the policy made more likely is down-weighted). It is easy to
        miss: ``rho`` enters as a detached coefficient, so the update direction
        is unchanged and only its magnitude is wrong, by ``rho**2``.

        The ``1/(2 sigma^2)`` on the loss
        ---------------------------------
        ``omega`` carries ``adv / (2 * sigma**2)``, not bare ``adv``.
        :meth:`reward_augmentation` injects a gradient of ``(2/N) * omega *
        (x_pred - x_pos)``; the mse already supplies the ``t**2``, so what is
        left to supply is exactly ``1 / (2 sigma**2)``. It goes on ``omega``
        rather than on the weights because ``W * omega`` is the reward term's
        whole coefficient and ``omega`` is the slot that already carries
        per-sample scale.

        PPO clip
        --------
        ``gate`` implements ``L = max{-A*rho, -A*clip(rho, 1-eps, 1+eps)}`` as
        flow-GRPO's loss does: the clipped branch is piecewise constant in
        ``rho``, so the max's gradient is either the unclipped one in full or
        zero. That is a 0/1 gate on the unclipped gradient, which is why it
        lives here, in the weights, and not in :meth:`_rl_loss`: the loss value
        this family emits is a squared probe residual (see
        :meth:`reward_augmentation`) and cannot represent a ``max`` of two
        linear surrogates, but zeroing the weight reproduces the gradient
        exactly. ``>=`` sends ties to the unclipped branch; on a tie both
        branches of flow-GRPO's ``torch.maximum`` carry the unclipped gradient,
        so this matches it elementwise.

        The mse is a spatial mean, so this is the per-dimension log-ratio. The
        joint ratio would multiply the exponent by the latent dimension
        (16*64*64 = 65536) and overflow on any nonzero difference; the mean form
        keeps the exponent representable, is the normalization flow-GRPO uses,
        and is why ``clip_range`` defaults to 1e-3 rather than PPO's usual 0.2
        (see the config field).

        ``x_anchor_1`` is unread: there is no self-regression term, and
        :meth:`_compute_x_anchor_1` returns None so the base zeroes that half.
        ``reg_weight`` is still returned equal to ``rwd_weight`` rather than as
        zeros: it reaches the logs as ``affine_coeff_mean``, and the term it
        would weight is switched off anyway.

        ``omega`` is reshaped to ``[B, 1, 1, 1]`` because
        :meth:`reward_augmentation` multiplies it against a spatial direction;
        a ``[B]`` tensor would broadcast onto the width axis of
        ``[B, C, H, W]``.
        """
        # sigma must be [B] to line up with the per-sample losses. A [B, 1, 1, 1]
        # temperature broadcasts them to [B, 1, 1, B] and corrupts omega's shape
        # instead of failing here, so it is checked rather than documented.
        if sigma.ndim != 1 or sigma.shape[0] != x_0_pred_v.shape[0]:
            raise ValueError(
                f"sigma must be [B] with B == {x_0_pred_v.shape[0]}, got shape "
                f"{tuple(sigma.shape)}. Pass the flattened per-sample "
                "temperature, not the broadcast timestep."
            )
        l_pred = self._per_sample_mse(x_0_pred_v, x_0_rwd, spatial)
        l_old = self._per_sample_mse(x_anchor_2, x_0_rwd, spatial)
        two_sigma2 = 2.0 * sigma**2
        # (l_old - l_pred), not the reverse: see "Sign of the log-ratio" above.
        ratio = torch.exp((l_old - l_pred) / two_sigma2)

        # PPO gate, elementwise over the batch. Reproduces torch.maximum's
        # gradient routing on max{-A*rho, -A*clip(rho)} without building either
        # branch as a tensor op the autograd graph would have to carry.
        clip_range = float(self._rl_cfg().clip_range)
        unclipped = -adv * ratio
        clipped = -adv * ratio.clamp(1.0 - clip_range, 1.0 + clip_range)
        inside = (ratio > 1.0 - clip_range) & (ratio < 1.0 + clip_range)
        gate = (inside | (unclipped >= clipped)).to(ratio.dtype)

        bcast = (-1, *([1] * (x_0_pred_v.ndim - 1)))
        omega = (gate * adv / two_sigma2).view(*bcast)
        return ratio, ratio, omega

    def _rl_loss(
        self,
        x_0: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_embeds: torch.Tensor,
        advantages: torch.Tensor,
        t_raw: torch.Tensor,
        r_raw: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """One AWM step over one freshly noised query point.

        Forward noising, not replay: ``x_t`` is built from the rewarded sample
        and a fresh ``eps`` at the ``(t, r)`` the base's inner loop drew, and the
        flow map is asked to denoise it back::

            x_t      = (1 - t) * x_0 + t * eps
            v_theta  = u_theta(x_t, t, r) + (t - r) * sg(dF/dt)
            x_0_pred = x_t - t * v_theta

        with ``x_anchor_2`` the same construction on ``u_old``. The objective is

            L     = gate * ratio * adv / (2*sigma**2) * mse(x_0_pred, x_0),
            ratio = exp((mse(x_anchor_2, x_0) - mse(x_0_pred, x_0)) / (2*sigma**2))
            gate  = 0 where the PPO clip kills the gradient, else 1

        with ``sigma = t``. All three factors are assembled by
        :meth:`decoupled_weights` and handed to :meth:`decoupled_rl_loss`, whose
        reward half injects exactly this gradient (see
        :meth:`reward_augmentation`). Because ``sigma == t`` makes the ``t**2``
        cancel, the gradient reaching the network is

            dL/du_theta = ratio * adv * (u_theta - (v_tgt - S)) / N,
            v_tgt = eps - x_0,   S = (t - r) * sg(dF/dt),

        i.e. an advantage- and ratio-weighted MeanFlow regression, t-independent.

        ``r_raw`` is optional, as in the anchor-form trainers: without it the
        run is not on the two-time flow map, ``S`` is identically zero and the
        prediction degenerates to ``x_t - t * u_theta``, which is then correct
        because ``u`` is already instantaneous.
        """
        cfg = self._rl_cfg()
        dtype = x_0.dtype
        batch_size = x_0.shape[0]
        T = float(self.scheduler.config.num_train_timesteps)
        wdt = self._rl_work_dtype

        # 1. Forward noising, the same op sequence as the anchor-form trainers.
        # t_normed is derived in the network dtype because x_t is built from it;
        # computing it in fp32 instead would change x_t itself.
        bcast = (batch_size, *([1] * (x_0.ndim - 1)))
        t_raw = t_raw.to(device=x_0.device)
        t_normed = t_raw.to(dtype) / T
        t_broadcast = t_normed.view(bcast)
        eps = self._draw_eps(x_0)
        x_t = (1.0 - t_broadcast) * x_0 + t_broadcast * eps
        t_for_model = t_raw.to(dtype=torch.float32)
        r_for_model = r_raw.to(dtype=torch.float32) if r_raw is not None else None
        # r == t is the S == 0 convention the base uses when the run is not on
        # the two-time flow map (see GeneralRLTrainer.compute_kl_term): it takes
        # _s_term's zero branch instead of tripping its None compare.
        # _pre_compute_v_hook still gets the raw r_for_model, because it uses
        # None to skip the CD.
        r_cd = r_for_model if r_for_model is not None else t_for_model

        # Which KL mode (if any) runs is decided here, before the anchor block,
        # because that block has to know whether to keep the shared dF/dt past
        # its ``finally``. Config-level, hence rank-uniform.
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

        # 2. One shared central difference on old_model, then both velocities.
        self._pre_compute_v_hook(
            x_t=x_t, t_for_model=t_for_model, r_for_model=r_for_model,
            prompt_embeds=prompt_embeds, pooled_embeds=pooled_embeds, x_0=x_0,
        )
        try:
            u_theta = self._compute_u(
                self.generator, x_t, t_for_model, prompt_embeds, pooled_embeds,
                r_for_model, with_grad=True,
            )
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

            # 3. Denoise to x_0 along the ODE. The identity has to be applied
            # BEFORE the jump: x_t - t * u_theta is only an x0 prediction where u
            # is instantaneous. S is a detached old_model constant, so this adds
            # no gradient path -- it moves u_theta's target to v_tgt - S.
            v_theta_f = self._instantaneous_velocity_from_u(
                u_theta_f, t_for_model, r_cd, self._shared_dF_dt,
            )
            x_0_pred_v = x_t_f - t_normed_b_f * v_theta_f          # carries grad

            # Packed once, checked once: AnchorInputs rejects a network-dtype
            # operand and an unbroadcast [B] timestep, the two substitutions
            # that are otherwise silent here.
            anchor_inputs = AnchorInputs(
                x_t=x_t_f,
                t_normed=t_normed_b_f,
                u_old=u_old_f,
                u_theta=u_theta_f.detach(),
                dF_dt=self._shared_dF_dt,
                t_cd=t_for_model,
                r_cd=r_cd,
                prompt_embeds=prompt_embeds,
                pooled_embeds=pooled_embeds,
            )
            x_anchor_1 = self._compute_x_anchor_1(anchor_inputs)
            x_anchor_2 = self._compute_x_anchor_2(anchor_inputs)

            # Reduced to a float inside the block so the latent-sized CD stays
            # freeable. Measures the correction in x0 units, i.e. the amount by
            # which the prediction differs from the plain ``x_t - t * u_theta``.
            s_term = self._s_term(t_for_model, r_cd, self._shared_dF_dt, like=x_t_f)
            s_x0_abs_mean = (
                0.0 if s_term is None
                else float((t_normed_b_f * s_term).abs().mean())
            )

            # ``finally`` below clears the attribute, so the v-mode KL -- the
            # only consumer outside this block -- needs a local reference.
            # Taken only when the v-mode KL runs, so the latent-sized CD stays
            # freeable otherwise. Reusing the same tensor rather than
            # recomputing also keeps the KL's S identical to the prediction's
            # S.
            shared_dF_dt = self._shared_dF_dt if use_v_kl else None
        finally:
            self._post_compute_v_hook()

        adv = advantages.float().clamp(-adv_clip_max, adv_clip_max)

        # The ratio temperature. Derived from the network-dtype t_normed rather
        # than from t_for_model / T so it is bit-identical to the t that built
        # x_t and scaled the jump: that identity is what makes the t**2
        # cancellation exact rather than merely close. [B], never [B, 1, 1, 1]
        # -- decoupled_weights checks.
        sigma = t_normed.to(wdt).reshape(batch_size).clamp(min=self._SIGMA_MIN)

        # AWM reuses the decoupled kernel with the reward half only:
        # ``x_anchor_1=None`` switches self-regression off, ``x_anchor_2`` carries
        # pi_old's x0 prediction so decoupled_weights can form the ratio, and the
        # ``r_hat`` / ``beta`` slots take the clipped advantage and the
        # temperature.
        unscaled, form_info = self.decoupled_rl_loss(
            x_0_pred_v, x_anchor_1, x_anchor_2, x_0_f, adv, sigma, spatial,
        )
        # Not rescaled by adv_clip_max: the anchor-form trainers do that to undo
        # the ``r_hat = adv / (2 * adv_clip_max) + 0.5`` squash, which maps the
        # advantage into [0, 1]. Here omega carries the clipped advantage in raw
        # units, so the same multiply would silently inflate the effective
        # learning rate by adv_clip_max (5x at adv_clip_max=5).
        policy_loss = unscaled.mean()
        loss = policy_loss

        with torch.no_grad():
            # Recomputed rather than plumbed out of decoupled_weights: that
            # method's return signature is fixed by the base's three-slot
            # contract, and these are the standard GRPO health metrics that
            # flow-GRPO logs. Cheap -- two per-sample MSEs on tensors already in
            # cache.
            clip_range = float(cfg.clip_range)
            l_pred_d = self._per_sample_mse(x_0_pred_v.detach(), x_0_f, spatial)
            l_old_d = self._per_sample_mse(x_anchor_2, x_0_f, spatial)
            log_rho = (l_old_d - l_pred_d) / (2.0 * sigma**2)
            rho_d = torch.exp(log_rho)
            info = {
                "loss_policy": float(policy_loss.detach()),
                "loss_unscaled": float(unscaled.mean().detach()),
                "t_mean": float(t_raw.mean().item()),
                # r_cd, not r_raw: on a run without the two-time flow map this
                # reports t, which is the r == t convention actually in force.
                "r_mean": float(r_cd.mean().item()),
                # Post-clamp, so a value pinned at _SIGMA_MIN says the batch was
                # dominated by the (very rare) small-t slice.
                "sigma_mean": float(sigma.mean().item()),
                "advantage_mean": float(advantages.mean().item()),
                "advantage_std": float(advantages.std().item()),
                "x0_norm_mean": float((x_0_f ** 2).mean().item()),
                # u-space divergence between the policy and the behaviour policy
                # at this query point. S cancels in the difference, so this is
                # policy drift alone.
                "old_deviate": float(((u_theta_f - u_old_f) ** 2).mean().item()),
                # |t * S|: how much the MeanFlow correction moves the prediction
                # away from the plain ``x_t - t * u`` one. Exactly 0 on the
                # r == t slice by construction.
                "s_x0_abs_mean": s_x0_abs_mean,
                # The trust-region health panel. ``ratio_mean`` drifting off 1.0
                # or ``clipfrac`` approaching 1.0 both say the region is binding
                # on most samples, i.e. clip_range is too tight for the drift the
                # old-model decay is producing.
                "ratio_mean": float(rho_d.mean()),
                "ratio_max_dev": float((rho_d - 1.0).abs().max()),
                "clipfrac": float((rho_d - 1.0).abs().gt(clip_range).float().mean()),
                # 0.5*E[(log rho)^2], the standard PPO KL proxy.
                "approx_kl": float(0.5 * (log_rho**2).mean()),
                **form_info,
            }
        kl, kl_info = self.compute_kl_term(
            x_t, t_for_model, prompt_embeds, pooled_embeds, r_for_model,
            shared_dF_dt, u_theta_f, spatial,
        )
        loss = loss + kl_weight * kl
        info["loss_kl"] = float(kl.detach())
        # Merged unconditionally: compute_kl_term pre-seeds every key with its
        # inactive value, so the key set is identical for the u-mode KL, the
        # v-mode KL and the KL-off case. _reduce_metrics all-reduces per key
        # with no reconciliation, so a conditional key hangs the job.
        info.update(kl_info)

        info["loss_total"] = float(loss.detach())
        return loss, info

    def reward_augmentation(
        self,
        x_pred: torch.Tensor,
        x_pos: torch.Tensor,
        x_neg: torch.Tensor,  # noqa: ARG002 - see below: the center is x_pred, not x_neg
        omega: torch.Tensor,
        weight: torch.Tensor,
        spatial: tuple[int, ...],
    ) -> torch.Tensor:
        """Injects the gradient of ``weight * omega * mse(x_pred, x_pos)`` -> ``[B]``.

        Same stop-grad probe-center device as the base, aimed at a different
        objective. The base's direction is ``x_pos - x_neg``, two detached
        constants, which injects a constant gradient. AWM's target is a real MSE
        against the rewarded sample, whose gradient is proportional to
        ``x_pred - x_pos``, so the direction has to be measured from ``x_pred``
        itself and ``x_neg`` goes unread. It stays in the signature because
        :meth:`GeneralRLTrainer.decoupled_rl_loss` calls this positionally.

        With ``x_target = x_pred.detach() + omega * (x_pos - x_pred.detach())``
        the residual is ``-norm_grad`` in value while its Jacobian w.r.t.
        ``x_pred`` is the identity, so::

            dL/dx_pred = weight * (2/N) * (x_pred - x_target)
                       = -weight * omega * (2/N) * (x_pos - x_pred)
                       = d/dx_pred [ weight * omega * mean_sp((x_pred - x_pos)^2) ]

        which is the intended gradient exactly, not to first order. The loss
        value is ``weight * mean_sp[norm_grad ** 2]``, i.e. it carries ``omega``
        squared -- so ``loss_policy`` is not comparable with a directly-written
        ``ratio * adv * mse``, and a negative advantage still shows up as a
        positive loss. Gate on gradients, never on this value. It is also why
        :attr:`_SIGMA_MIN` exists: ``omega`` scales as ``1 / t**2`` here, so the
        value (though not the gradient) diverges as ``t -> 0``.

        Sign: ``omega > 0`` must pull ``x_pred`` toward ``x_pos``. That needs
        the residual to be ``-norm_grad``, i.e. ``norm_grad`` added to the
        detached center and the direction taken as ``x_pos - x_pred``. Writing
        ``x_pred.detach() - norm_grad`` flips the gradient and trains the
        policy to flee the rewarded sample, while leaving every logged number
        bit-identical.
        """
        with torch.no_grad():
            norm_grad = (omega * (x_pos.double() - x_pred.double())).to(x_pred.dtype)
        x_target = (x_pred.detach() + norm_grad).to(x_pred.dtype)
        return weight * self._per_sample_mse(x_pred, x_target, spatial)


class AWMTrainer(MeanFlowAWMTrainer):
    """Ablation of :class:`MeanFlowAWMTrainer` with the MeanFlow identity dropped.

    The flow map's average velocity is used as the instantaneous velocity
    (``v := u``, ``u := v``, i.e. ``S = 0`` for every sample, including in the
    v-mode KL target), so the prediction is ``x_0_pred = x_t - t * u_theta``
    and ``u_theta`` regresses onto ``v_tgt = eps - x_0`` directly instead of
    onto ``v_tgt - S``. The central difference is never computed, since nothing
    reads it.

    Everything else -- sampler, (t, r) draw, ratio, PPO clip, KL -- is inherited
    unchanged, so the only variable against ``meanflowawm_rl`` is the S term.
    """

    TRAINER_KEY = "awm_rl"

    def _pre_compute_v_hook(self, **_kwargs) -> None:
        # Class-level, hence rank-uniform: no rank enters the CD's collectives.
        return None

    def _s_term(
        self,
        t_cd: torch.Tensor,
        r_cd: torch.Tensor,
        dF_dt: Optional[torch.Tensor],
        like: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        return None

    def _instantaneous_velocity_from_u(
        self,
        u_old: torch.Tensor,
        t_cd: torch.Tensor,
        r_cd: torch.Tensor,
        dF_dt: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``v(x_t, t) := u(x_t, r, t)`` -- no S term."""
        return u_old

    def _average_velocity_from_v(
        self,
        v_inst: torch.Tensor,
        t_cd: torch.Tensor,
        r_cd: torch.Tensor,
        dF_dt: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``u(x_t, r, t) := v(x_t, t)`` -- no S term."""
        return v_inst
