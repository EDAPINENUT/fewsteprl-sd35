"""MeanFlowGRPO: flow-GRPO's clipped likelihood-ratio loss on a two-time flow map.

:class:`MeanFlowGRPOTrainer` replays the SDE rollout's own transitions
``(x_t, t) -> (x_r, r)`` (``rollout_type=sde``, ``reuse_x_t=true``). For each
one it queries the flow map over ``[s, t]`` at a freshly drawn ``s <= t``,
lifts that average velocity to the instantaneous one with the MeanFlow
identity ``v = u(x_t, t, s) + (t - s) * sg(dF/dt)`` (``dF/dt`` is a central
difference on ``old_model``, shared by both policies), and scores the replayed
action ``x_r`` under flow-GRPO's Euler-Maruyama Gaussian kernel on the grid
``(t, r)``. The per-step log-likelihood ratio ``rho`` between the policy and
``old_model`` enters the PPO surrogate ``max{-A * rho, -A * clip(rho, 1 - eps,
1 + eps)}``, with the sample's advantage ``A`` shared by all of its steps; the
optional KL term is :meth:`GeneralRLTrainer.compute_kl_term`.

:class:`FlowGRPOTrainer` is the S-term ablation: it uses ``u`` directly as the
instantaneous velocity (``S = 0``) and never computes the central difference.
"""

from __future__ import annotations

from typing import Optional

import torch

from fewsteprl.trainers.rl_general_base import AnchorInputs, GeneralRLTrainer


class MeanFlowGRPOTrainer(GeneralRLTrainer):
    """Flow-GRPO on replayed SDE-rollout transitions; see the module docstring."""

    TRAINER_KEY = "meanflowgrpo_rl"

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

    #: GRPO diagnostics, in place of the base's anchor-form keys: those measure
    #: ``x_anchor_1 - x_anchor_2``, and this class passes ``x_anchor_1=None`` and
    #: uses the ``x_anchor_2`` slot for pi_old's transition mean, so that
    #: difference is not defined. ``_run_rl_epoch`` logs only allowlisted keys
    #: and silently drops the rest, so every diagnostic :meth:`_rl_loss` emits
    #: must be listed here.
    #:
    #: ``clipfrac`` / ``ratio_max_dev`` show whether ``fewstep_rl.clip_range``
    #: binds: near 0 the trust region never binds, near 1 it binds everywhere
    #: and learning stalls.
    #:
    #: A class constant, hence rank-uniform: :meth:`BaseTrainer._reduce_metrics`
    #: all-reduces per key with no key-set reconciliation.
    _EXTRA_DIAGNOSTIC_KEYS: tuple[str, ...] = (
        "s_mean",           # mean flow-map jump target s (redrawn per transition)
        "r_mean",           # mean renoise endpoint (rollout grid)
        "tau_mean",         # mean kernel std, std_dev_t * sqrt(t - r)
        "ratio_mean",       # E[rho]; 1.0 at zero drift
        "ratio_max_dev",    # max|rho - 1|; the exp-runaway detector
        "clipfrac",         # fraction of transitions outside the trust region
        "approx_kl",        # 0.5*E[(log rho)^2], the PPO KL proxy
    )

    #: ``sample_pack`` keys this trainer requires from the rollout. All five
    #: are None unless the rollout ran with ``return_intermediate``, which
    #: ``fewstep_rl.reuse_x_t`` controls. Four are replayed; ``mid_timesteps``
    #: is only presence- and shape-checked, since the training ``s`` is
    #: redrawn -- see the load block in :meth:`_rl_train_phase`.
    _REPLAY_KEYS: tuple[str, ...] = (
        "x_t", "x_r", "start_timesteps", "mid_timesteps", "end_timesteps",
    )

    #: Not read by this class. The ratio's temperature is flow-GRPO's kernel
    #: std ``tau = std_dev_t * sqrt(t - r)`` (see :meth:`_rl_loss`), a smooth
    #: function of the grid ``(t, r)`` that is positive on every node, so no
    #: floor is applied.
    _TAU_MIN: float = 0.5

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
        """``v(x_t, t) = u(x_t, r, t) + (t - r) * dF/dt`` -- the identity solved for v."""
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
    def _compute_x_anchor_1(self, inputs: AnchorInputs) -> torch.Tensor:
        """``x_anchor_1 = x_t - t * u_old``.

        Not called by this class's :meth:`_rl_loss`, which builds the
        transition means directly.
        """
        return inputs.x_t - inputs.t_normed * inputs.u_old

    @torch.no_grad()
    def _compute_x_anchor_2(self, inputs: AnchorInputs) -> torch.Tensor:
        """``x_anchor_2 = x_t - t * v_inst``, with ``v_inst = u_old + S``.

        Not called by this class's :meth:`_rl_loss`, which builds the
        transition means directly.
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
        x_r_pred_u: torch.Tensor,
        x_anchor_1: Optional[torch.Tensor],  # noqa: ARG002 - see below: always None here
        x_anchor_2: torch.Tensor,   # pi_old's mean for this transition
        x_r_rwd: torch.Tensor,      # the action pi_old actually sampled
        adv: torch.Tensor,          # occupies the anchor form's ``r_hat`` slot
        tau_r: torch.Tensor,        # occupies the anchor form's ``beta`` slot
        spatial: tuple[int, ...]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """GRPO weights ``(ratio, ratio, omega)`` for the base's reward term.

        The anchor-form trainers read these three slots as affine probe
        coefficients; here they weight a policy-gradient loss

            L = gate * ratio * adv / (2 * tau_r**2) * mse(x_r_pred_u, x_r_rwd)

        with ``ratio`` detached. ``ratio`` is the Gaussian-kernel likelihood
        ratio between the current policy and the behaviour policy on the action
        the rollout took, with means ``x_r_pred_u`` and ``x_anchor_2``. Its
        temperature is the transition's own std: the kernel draws ``x_r`` around
        its mean with std ``tau_r``, so ``log p = -mse / (2 * tau_r**2)``. The
        anchor form passes a scalar ``beta`` in this slot; here it is a
        per-sample ``[B]`` tensor, which works because the base only forwards
        the slot to this method.

        Sign of the log-ratio
        ---------------------
        ``log rho = log p_theta - log p_old = (l_old - l_pred) / (2 tau^2)``,
        i.e. the old loss leads. The reverse order yields ``1/rho``, which
        inverts the PPO weighting (a transition the policy made more likely is
        down-weighted). It is easy to miss: ``rho`` enters as a detached
        coefficient, so the update direction is unchanged and only its
        magnitude is wrong, by ``rho**2``.

        The ``1/(2 tau^2)`` on the loss
        -------------------------------
        ``omega`` carries ``adv / (2 tau_r**2)``, not bare ``adv``. With ``rho``
        detached, ``grad(-A * rho) = -A * rho * grad(log p_theta) = A * rho *
        grad(mse) / (2 tau^2)``, and :meth:`reward_augmentation` injects the
        gradient of ``weight * omega * mse``, so the ``1/(2 tau^2)`` must come
        from ``omega``. Without it each transition's gradient is off by its own
        ``2 tau^2``, a per-transition learning-rate error that a global lr
        cannot absorb. It goes on ``omega`` rather than on the weights because
        ``W * omega`` is the reward term's whole coefficient and ``omega`` is the
        slot that already carries per-sample scale.

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
        :meth:`_rl_loss` passes None so the base zeroes that half.
        ``reg_weight`` is still returned equal to ``rwd_weight`` rather than as
        zeros: it reaches the logs as ``affine_coeff_mean``, and the term it
        would weight is switched off by ``x_anchor_1=None`` anyway.

        ``omega`` is reshaped to ``[B, 1, 1, 1]`` because
        :meth:`reward_augmentation` multiplies it against a spatial direction;
        a ``[B]`` tensor would broadcast onto the width axis of
        ``[B, C, H, W]``.
        """
        l_pred = self._per_sample_mse(x_r_pred_u, x_r_rwd, spatial)
        l_old = self._per_sample_mse(x_anchor_2, x_r_rwd, spatial)
        # tau_r is [B] to match the per-sample losses. A [B, 1, 1, 1] tau would
        # silently broadcast to [B, 1, 1, B] here instead of failing.
        two_tau2 = 2.0 * tau_r**2
        # (l_old - l_pred), not the reverse: see "Sign of the log-ratio" above.
        ratio = torch.exp((l_old - l_pred) / two_tau2)

        # PPO gate, elementwise over the batch. Reproduces torch.maximum's
        # gradient routing on max{-A*rho, -A*clip(rho)} without building either
        # branch as a tensor op the autograd graph would have to carry.
        clip_range = float(self._rl_cfg().clip_range)
        unclipped = -adv * ratio
        clipped = -adv * ratio.clamp(1.0 - clip_range, 1.0 + clip_range)
        inside = (ratio > 1.0 - clip_range) & (ratio < 1.0 + clip_range)
        gate = (inside | (unclipped >= clipped)).to(ratio.dtype)

        bcast = (-1, *([1] * (x_r_pred_u.ndim - 1)))
        omega = (gate * adv / two_tau2).view(*bcast)
        return ratio, ratio, omega

    def _rl_train_phase(self, sample_pack: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Replay the rollout's own states and grid; redraw only the jump target s.

        The forward-noising trainers draw a fresh ``(t, r)`` per inner
        iteration and rebuild ``x_t`` by noising ``x_0``. This one does not:
        the SDE rollout already produced ``K = sampling_num_steps`` transitions
        per sample, each an ``(x_t, t) -> (x_s, s) -> (x_r, r)``
        jump-then-renoise step, and those are what the policy is scored on. So
        the inner loop indexes rollout steps instead of drawing timesteps, and
        ``_num_training_timesteps`` / ``_prepare_inner_epoch_tr_state`` /
        ``_draw_training_tr`` are not used here.

        What is redrawn is the jump target ``s``. The rollout picked it off the
        inference grid (``timesteps[i + 1:]``, at most ``K - i`` distinct values
        on step ``i``), so the flow map's second argument would only ever take a
        handful of values per step.
        :meth:`GeneralRLTrainer._draw_training_s_given_t` replaces that with a
        continuous draw that has ``r``'s conditional law given ``t``, so each
        inner epoch queries ``u_theta(x_t, t, s)`` at a different jump length
        while ``(x_t, x_r, t, r)`` -- the state, the action, and the two
        endpoints the action was produced between -- stay exactly as sampled.
        ``s`` is bounded by ``t`` but not by ``r``; the transition kernel
        depends on ``(t, r)`` alone (see :meth:`_rl_loss`), so ``s`` only
        selects which average velocity is queried and lifted.

        The optimizer / grad-accumulation / EMA block is the same as the base's;
        only what feeds :meth:`_rl_loss` differs.
        """
        cfg = self._rl_cfg()
        device = torch.device("cuda")

        missing = [k for k in self._REPLAY_KEYS if sample_pack.get(k) is None]
        if missing:
            # Config-level and therefore rank-uniform: every rank reaches this
            # with the same verdict, so raising cannot desync the collectives.
            raise ValueError(
                f"{type(self).__name__} replays the rollout's own states, so it "
                f"requires fewstep_rl.reuse_x_t=true; sample_pack is missing "
                f"{missing}."
            )

        prompt_embeds = sample_pack["prompt_embeds"].to(device)
        pooled_embeds = sample_pack["pooled_embeds"].to(device)
        advantages = sample_pack["advantages"].to(device)
        # [n_local, C, H, W], the trajectory's TERMINAL latent -- not a
        # per-step quantity. Only the central difference reads it, and only to
        # point ``v_dir``; see the caveat in :meth:`_rl_loss`. Shared by all K
        # steps of a sample for want of a per-step x_0.
        x_0_all = sample_pack["x_0"].to(device)
        # [n_local, K, C, H, W]: x_t[:, k] is the state ENTERING step k and
        # x_r[:, k] the state leaving it, so x_t[:, k + 1] == x_r[:, k].
        x_t_all = sample_pack["x_t"].to(device)
        x_r_all = sample_pack["x_r"].to(device)
        # [n_local, K], raw timestep units. t -> s is the flow-map jump the
        # sampler took; s -> r is the renoise back up.
        #
        # Only (t, r), the grid endpoints x_r was produced between, are
        # replayed: the training s is redrawn per transition (see the loop
        # below). ``s_all`` is still required and shape-checked because the SDE
        # rollout emits all five keys together under ``reuse_x_t``, so a
        # sample_pack without it did not come from the rollout this class
        # assumes.
        t_all = sample_pack["start_timesteps"].to(device=device, dtype=torch.float32)
        s_all = sample_pack["mid_timesteps"].to(device=device, dtype=torch.float32)
        r_all = sample_pack["end_timesteps"].to(device=device, dtype=torch.float32)

        n_local, n_steps = t_all.shape
        if n_local == 0:
            return {}
        for name, tensor in (
            ("x_t", x_t_all), ("x_r", x_r_all),
            ("mid_timesteps", s_all), ("end_timesteps", r_all),
        ):
            if tensor.shape[:2] != (n_local, n_steps):
                raise ValueError(
                    f"sample_pack['{name}'] has leading shape "
                    f"{tuple(tensor.shape[:2])}, expected {(n_local, n_steps)}. "
                    "The rollout must stack samples on dim 0 and rollout steps "
                    "on dim 1."
                )
        # Checked separately: x_0 has no step axis, so it is [n_local, ...].
        if x_0_all.shape[0] != n_local:
            raise ValueError(
                f"sample_pack['x_0'] has leading dim {x_0_all.shape[0]}, expected "
                f"{n_local}. It is indexed by the same row permutation as the "
                "per-step tensors, so a mismatch would pair a state with "
                "another sample's endpoint."
            )

        inner_bs = int(cfg.rl_inner_batch_size) if cfg.rl_inner_batch_size > 0 else n_local
        inner_bs = max(1, min(inner_bs, n_local))
        n_inner_epochs = max(1, int(cfg.num_inner_epochs))
        max_grad_norm = float(self.config.solver.generator.max_grad_norm)
        grad_accum = max(1, int(self.config.train.gradient_accumulation_steps))
        # One backward per rollout step, so the rollout length takes the place
        # of ``num_training_timesteps_per_sample`` in the accumulation window.
        effective_grad_accum_steps = grad_accum * n_steps

        self.generator.train()
        accumulator: dict[str, float] = {}
        n_backward = 0       # cumulative backwards across the whole train phase
        n_opt_steps = 0
        current_accumulated_steps = 0
        self.optimizers["generator"].zero_grad(set_to_none=True)
        last_grad_norm: float = 0.0

        # Only reaches the (t, s) sampler, which draws its uniform in the
        # network dtype to consume the RNG the way the base's t1/t2 do.
        dtype = next(self.generator.parameters()).dtype

        # inner_epoch_idx keys the s|t RNG, so two inner epochs over the same
        # transition draw different jump targets.
        for inner_epoch_idx in range(n_inner_epochs):
            # Per-sample shuffle, rows only: the step axis indexes physical
            # transitions, so permuting it would pair a state with another
            # step's timesteps.
            perm = torch.randperm(n_local, device=device)
            sx0 = x_0_all[perm]
            sxt = x_t_all[perm]
            sxr = x_r_all[perm]
            st = t_all[perm]
            sr = r_all[perm]
            spe = prompt_embeds[perm]
            spp = pooled_embeds[perm]
            sadv = advantages[perm]

            for start in range(0, n_local, inner_bs):
                # Clamped, unlike the base's ``start + inner_bs``: ``end`` is
                # not just a slice bound here, it is the chunk size the s|t
                # sampler partitions by global index, and on a ragged final
                # chunk an unclamped ``end`` would ask for more draws than the
                # chunk has rows.
                end = min(start + inner_bs, n_local)
                pe_b = spe[start:end]
                pp_b = spp[start:end]
                adv_b = sadv[start:end]
                for k in range(n_steps):
                    t_b = st[start:end, k]
                    r_b = sr[start:end, k]
                    # Fresh jump target for this transition, drawn with r's
                    # own conditional law given t. ``k`` keys the RNG
                    # because the rollout step is this loop's draw axis -- the
                    # slot the base fills with its ``j_idx``.
                    s_b = self._draw_training_s_given_t(
                        t_b,
                        state={},   # this trainer carries no per-inner-epoch draw state
                        chunk_start=start,
                        chunk_end=end,
                        j_idx=k,
                        inner_epoch_idx=inner_epoch_idx,
                        device=device,
                        dtype=dtype,
                    )
                    loss, info = self._rl_loss(
                        # No step axis: one terminal latent per sample, reused
                        # by its K transitions (see the load block above).
                        x_0=sx0[start:end],
                        x_t=sxt[start:end, k],
                        x_r=sxr[start:end, k],
                        prompt_embeds=pe_b,
                        pooled_embeds=pp_b,
                        # One advantage per SAMPLE, shared by its K steps: the
                        # reward scored the trajectory, not the transition.
                        advantages=adv_b,
                        t_raw=t_b,
                        s_raw=s_b,
                        r_raw=r_b,
                    )
                    # Scale loss for grad accumulation; gradients accumulate
                    # across `effective_grad_accum_steps` backward calls.
                    (loss / float(effective_grad_accum_steps)).backward()
                    current_accumulated_steps += 1
                    n_backward += 1

                    # Not ``k``: that is the rollout-step index of the enclosing
                    # loop.
                    for key, val in info.items():
                        accumulator[key] = accumulator.get(key, 0.0) + float(val)

                    if current_accumulated_steps % effective_grad_accum_steps == 0:
                        last_grad_norm = self._compute_grad_norm(self.generator)
                        if max_grad_norm > 0:
                            self._clip_grad_norm(self.generator, max_grad_norm)
                        self.optimizers["generator"].step()
                        self.schedulers["generator"].step()
                        self.optimizers["generator"].zero_grad(set_to_none=True)

                        # EMA per opt step, with the warmup-aware decay of
                        # _current_ema_decay.
                        if (
                            self.generator_ema is not None
                            and self.global_step % self.config.train.ema_every == 0
                        ):
                            self.ema_update(
                                self.generator, self.generator_ema,
                                self._current_ema_decay(),
                            )

                        # ``global_step`` increments per opt step (DiffusionNFT
                        # convention); _current_decay and _current_ema_decay
                        # read it.
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
        # Backwards per sample per inner epoch, which here is the rollout length.
        averaged["num_train_timesteps"] = float(n_steps)
        averaged["effective_grad_accum_steps"] = float(effective_grad_accum_steps)
        return averaged


    def _rl_loss(
        self,
        x_0: torch.Tensor,
        x_t: torch.Tensor,
        x_r: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_embeds: torch.Tensor,
        advantages: torch.Tensor,
        t_raw: torch.Tensor,
        s_raw: torch.Tensor,
        r_raw: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """One GRPO step over one replayed rollout transition.

        No forward noising: ``x_t`` and ``x_r`` are the states the sampler
        visited and ``(t, r)`` the grid endpoints it visited them between.
        ``s`` is the flow map's jump target -- the interval ``u_theta`` is
        queried over -- and is a fresh per-transition draw rather than the
        rollout's own grid pick; see :meth:`_rl_train_phase`.

        Kernel
        ------
        ``s`` enters only through the MeanFlow lift ``v = u + (t - s) * dF/dt``,
        which turns the network's average velocity over ``[s, t]`` into the
        instantaneous ``v(x_t, t)`` that flow-GRPO's kernel consumes. The
        transition is then scored with flow-GRPO's Euler-Maruyama coefficients
        (``flow_grpo/grpo_loss.py::transition_coeffs``), which depend on the
        grid ``(t, r)`` alone::

            std_dev_t = noise_level * sqrt(t / (1 - t))
            a         = 1 - std_dev_t^2 / (2 t) * h,     h = t - r
            b         = (1 + std_dev_t^2 (1 - t) / (2 t)) * h
            mu        = a * x_t - b * v
            tau       = std_dev_t * sqrt(h)

        and the objective is flow-GRPO's::

            L     = gate * ratio * adv / (2*tau**2) * mse(mu_theta, x_r),
            ratio = exp((mse(mu_old, x_r) - mse(mu_theta, x_r)) / (2*tau**2))
            gate  = 0 where the PPO clip kills the gradient, else 1

        The old loss leads in the ratio because ``log p = -mse/(2 tau^2)``. All
        three factors are assembled by :meth:`decoupled_weights` and handed to
        :meth:`decoupled_rl_loss`, whose reward half injects exactly this
        gradient (see :meth:`reward_augmentation`). ``fewstep_rl.beta`` is
        unread here; ``clip_range`` and ``grpo_noise_level`` are the knobs that
        matter.

        Approximation
        -------------
        The rollout is jump-and-renoise, whose own kernel is centred at
        ``alpha * (x_t + (s - t) * u)`` with ``alpha = (1 - r) / (1 - s)``; its
        actions are scored here with the v-form kernel above instead. The ratio
        is still exactly 1 at zero policy drift, since both means are built by
        the same expression and the mismatch cancels in their difference, but
        the per-transition weighting is approximate. A v-space Euler-Maruyama
        rollout would remove the gap.

        ``r_raw`` is required: ``h``, and hence every coefficient, is undefined
        without it.

        ``x_0`` is the rollout's terminal latent, passed only to
        :meth:`GeneralRLTrainer._pre_compute_v_hook`, which under the default
        ``cd_velocity_source="noise_minus_x0"`` recovers ``eps`` from
        ``(x_t, x_0, t)`` to point the central difference. That assumes ``x_t``
        and ``x_0`` lie on one straight path, which holds when ``x_t`` is the
        forward-noised ``x_0`` but only approximately here: the SDE renoise
        injects fresh noise at every step. It biases the CD's direction, not
        the identity; ``cd_velocity_source="u_self"`` avoids it by taking the
        tangent from the model (``u(x_t, t, t)``) and ignoring ``x_0``.
        """
        cfg = self._rl_cfg()
        batch_size = x_r.shape[0]
        T = float(self.scheduler.config.num_train_timesteps)
        wdt = self._rl_work_dtype

        device = x_r.device
        t_for_model = t_raw.to(device=device, dtype=torch.float32)
        # The flow map is queried at (t, s), not (t, r): the lift below carries
        # (t - s), so u must be the average velocity over the same [s, t].
        s_for_model = s_raw.to(device=device, dtype=torch.float32)
        r_for_model = r_raw.to(device=device, dtype=torch.float32)

        # Normalized time in the fp32 work dtype, as _rollout_samples_sde uses.
        # The forward-noising trainers round t_normed to the network dtype
        # because x_t is built from it; nothing is rebuilt here, and bf16
        # (8 mantissa bits) would put a ~1e-3 relative error on the kernel
        # coefficients.
        bcast = (batch_size, *([1] * (x_r.ndim - 1)))
        t_normed_b_f = (t_for_model / T).view(bcast).to(wdt)
        r_normed_b_f = (r_for_model / T).view(bcast).to(wdt)

        # Which KL mode (if any) runs is decided here, before the velocity
        # block, because that block has to know whether to keep the shared
        # dF/dt past its ``finally``. Config-level, hence rank-uniform. Both
        # modes run with ``s`` in the ``r`` slot: the v-mode target is
        # ``v_ref - (t - s) * dF/dt``, the average velocity over the same
        # ``[s, t]`` u_theta is queried for.
        _, use_v_kl = self._kl_usage()
        kl_weight = float(cfg.kl_weight)

        if self._shared_dF_dt is not None:
            raise RuntimeError(
                "stale _shared_dF_dt on entry to _rl_loss -- a previous step "
                "leaked its central difference (missing _post_compute_v_hook)."
            )

        # 1. Average velocities only; S is never folded into these. The flow
        # map's second argument is ``s_for_model``: every ``r`` in the MeanFlow
        # identity is this class's ``s``, the endpoint u_theta is queried over
        # (``r`` is where the renoise lands, and the flow map never sees it).
        # So the CD is taken at fixed ``s`` and the lift below carries
        # ``(t - s)``.
        self._pre_compute_v_hook(
            x_t=x_t, t_for_model=t_for_model, r_for_model=s_for_model,
            prompt_embeds=prompt_embeds, pooled_embeds=pooled_embeds, x_0=x_0,
        )
        try:
            u_theta = self._compute_u(
                self.generator, x_t, t_for_model, prompt_embeds, pooled_embeds,
                s_for_model, with_grad=True,
            )
            u_old = self._compute_u(
                self.old_model, x_t, t_for_model, prompt_embeds, pooled_embeds,
                s_for_model, with_grad=False,
            ).detach()

            u_theta_f = u_theta.to(wdt)
            u_old_f = u_old.to(wdt)
            x_t_f = x_t.to(wdt)
            x_r_f = x_r.to(wdt)
            spatial = tuple(range(1, x_r.ndim))

            # 2. Lift both to instantaneous velocity with the same S:
            # ``v = u + (t - s) * dF/dt``. At zero policy drift
            # (u_theta == u_old) a per-network S would still offset the two
            # transition means, so the ratio would report CD noise instead of
            # exactly 1.0. ``_pre_compute_v_hook`` refuses to run with
            # ``share_cd_with_old=False``, which guarantees the sharing.
            #
            # dF/dt is a no-grad old_model constant, so v_theta_f carries
            # exactly u_theta_f's gradient and S rides along as a stop-grad
            # additive term; the diffusion slice gets S == 0 arithmetically
            # through ``(t - s)`` at no collective cost.
            v_theta_f = self._instantaneous_velocity_from_u(
                u_theta_f, t_for_model, s_for_model, self._shared_dF_dt,
            )
            v_old_f = self._instantaneous_velocity_from_u(
                u_old_f, t_for_model, s_for_model, self._shared_dF_dt,
            )
            # ``finally`` below clears the attribute, so the v-mode KL -- the
            # only consumer outside this block -- needs a local reference.
            # Taken only when the v-mode KL runs, so the latent-sized CD stays
            # freeable otherwise. Reusing the same tensor rather than
            # recomputing also keeps the KL's S identical to the policy term's
            # S.
            shared_dF_dt = self._shared_dF_dt if use_v_kl else None
        finally:
            self._post_compute_v_hook()


        adv_clip_max = float(cfg.adv_clip_max)
        if adv_clip_max <= 0:
            raise ValueError(f"adv_clip_max must be > 0, got {adv_clip_max}")

        # flow-GRPO's transition coefficients, term for term as in
        # ``flow_grpo/grpo_loss.py::transition_coeffs`` (the ``sde`` branch) and
        # ``sd3_sde_with_logprob.py``:
        #
        #     std_dev_t = noise_level * sqrt(sigma / (1 - sigma))
        #     a         = 1 + std_dev_t^2 / (2 sigma) * dt          (dt = -h < 0)
        #     b         = (1 + std_dev_t^2 (1 - sigma) / (2 sigma)) * h
        #     mean      = a * x_t - b * v
        #     tau       = std_dev_t * sqrt(h)
        #
        # with ``sigma = t``, ``sigma_prev = r``, ``h = t - r``.
        #
        # The naive Euler mean ``x_t + (r - t) * v`` is only the
        # ``noise_level -> 0`` limit of this (``a -> 1``, ``b -> h``); at a
        # nonzero noise level ``a < 1``, and dropping it would misplace the
        # kernel centre by a fraction of ``x_t``, far more than the policy
        # difference the ratio measures.
        h = t_normed_b_f - r_normed_b_f                    # > 0, the step length
        # flow-GRPO's ``sigma == 1`` substitution applies to the denominator
        # only:
        #
        #     std_dev_t = noise_level * sqrt(sigma / (1 - where(sigma == 1, sigma_max, sigma)))
        #
        # At t == 1 the raw ``1 - sigma`` is 0 and the variance would diverge,
        # so ``sigma_max`` (the first grid node below 1) is used there; the
        # numerator and ``a`` / ``b`` keep the raw sigma. Substituting in both
        # places would change ``tau`` on the first step.
        sigma = t_normed_b_f
        sigma_max = float(self.flowmap_scheduler.timesteps[1]) / T
        denom = 1.0 - torch.where(
            sigma >= 1.0, torch.full_like(sigma, sigma_max), sigma,
        )
        std_dev_t = float(cfg.grpo_noise_level) * torch.sqrt(sigma / denom)
        a_coef = 1.0 - std_dev_t**2 / (2.0 * sigma) * h     # dt = -h
        b_coef = (1.0 + std_dev_t**2 * (1.0 - sigma) / (2.0 * sigma)) * h

        # Both means of the step's transition kernel. The renoise adds the same
        # tau * eps to either, so it drops out of their difference and only the
        # means are needed to score the action the rollout took. The rollout's
        # own kernel is jump-and-renoise, not this one; see "Approximation" in
        # the docstring.
        x_r_pred_u = a_coef * x_t_f - b_coef * v_theta_f    # carries grad
        x_r_old_u = a_coef * x_t_f - b_coef * v_old_f

        # ``tau = std_dev_t * sqrt(h)``, flow-GRPO's own sampling/scoring std
        # (it draws with ``std_dev_t * sqrt(-dt)`` and scores with the same).
        # Flattened to [B] to line up with the per-sample losses downstream.
        tau_r = (std_dev_t * torch.sqrt(h)).reshape(batch_size)
        # No floor: tau is a smooth function of the grid (t, r) alone and
        # never reaches 0, because h > 0 and std_dev_t > 0 on every node.

        adv = advantages.float().clamp(-adv_clip_max, adv_clip_max)

        # GRPO reuses the decoupled kernel with the reward half only:
        # ``x_anchor_1=None`` switches self-regression off (there is no
        # policy-improvement anchor here), ``x_anchor_2`` carries pi_old's mean
        # so :meth:`decoupled_weights` can form the likelihood ratio, and the
        # ``r_hat`` / ``beta`` slots take the advantage and the ratio
        # temperature -- this step's tau_r rather than the anchor form's fixed
        # beta. What comes back has the gradient of
        # ``gate * ratio * adv / (2*tau_r**2) * mse(x_r_pred_u, x_r_rwd)``,
        # i.e. flow-GRPO's clipped-surrogate gradient.
        x_anchor_1 = None
        x_anchor_2 = x_r_old_u
        unscaled, form_info = self.decoupled_rl_loss(
            x_r_pred_u, x_anchor_1, x_anchor_2, x_r_f, adv, tau_r, spatial,
        )
        # Not rescaled by adv_clip_max: the anchor-form trainers do that to undo
        # the ``r_hat = adv / (2 * adv_clip_max) + 0.5`` squash, which maps the
        # advantage into [0, 1]. Here omega carries the clipped advantage in raw
        # units, so the same multiply would silently inflate the effective
        # learning rate by adv_clip_max (5x at adv_clip_max=5).
        policy_loss = unscaled.mean()
        loss = policy_loss

        # ``old_deviate`` is the u-space divergence between the policy and the
        # behaviour policy at the replayed query point.
        with torch.no_grad():
            # Recomputed rather than plumbed out of decoupled_weights: that
            # method's return signature is fixed by the base's three-slot
            # contract, and these are the standard GRPO health metrics that
            # flow-GRPO logs. Cheap -- two per-sample MSEs on tensors already in
            # cache.
            clip_range = float(cfg.clip_range)
            l_pred_d = self._per_sample_mse(x_r_pred_u.detach(), x_r_f, spatial)
            l_old_d = self._per_sample_mse(x_r_old_u, x_r_f, spatial)
            log_rho = (l_old_d - l_pred_d) / (2.0 * tau_r**2)
            rho_d = torch.exp(log_rho)
            info = {
                "loss_policy": float(policy_loss.detach()),
                "loss_unscaled": float(unscaled.mean().detach()),
                # ``t`` and ``r`` are the rollout grid endpoints; ``s`` is the
                # per-transition redraw (see _rl_train_phase).
                "t_mean": float(t_raw.mean().item()),
                "s_mean": float(s_raw.mean().item()),
                "r_mean": float(r_raw.mean().item()),
                # flow-GRPO's kernel std, a function of the grid (t, r) only.
                "tau_mean": float(tau_r.mean().item()),
                "advantage_mean": float(advantages.mean().item()),
                "advantage_std": float(advantages.std().item()),
                # Keeps the key name (it is in the log allowlist) but measures
                # the replayed x_r, which is what this trainer regresses against.
                "x0_norm_mean": float((x_r_f ** 2).mean().item()),
                "old_deviate": float(((u_theta_f - u_old_f) ** 2).mean().item()),
                # The GRPO health panel. ``ratio_mean`` drifting off 1.0 or
                # ``clipfrac`` approaching 1.0 both say the trust region is
                # binding on most transitions, i.e. clip_range is too tight for
                # the drift the old-model decay is producing.
                "ratio_mean": float(rho_d.mean()),
                "ratio_max_dev": float((rho_d - 1.0).abs().max()),
                "clipfrac": float((rho_d - 1.0).abs().gt(clip_range).float().mean()),
                # 0.5*E[(log rho)^2], the standard PPO KL proxy.
                "approx_kl": float(0.5 * (log_rho**2).mean()),
                **form_info,
            }
        kl, kl_info = self.compute_kl_term(
            x_t, t_for_model, prompt_embeds, pooled_embeds, s_for_model,
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
        constants, which injects a constant gradient. GRPO's target is a real
        MSE against the sampled action, whose gradient is proportional to
        ``x_pred - x_pos``, so the direction has to be measured from ``x_pred``
        itself and ``x_neg`` goes unread. It stays in the signature because
        :meth:`GeneralRLTrainer.decoupled_rl_loss` calls this positionally.

        With ``x_target = x_pred.detach() + omega * (x_pos - x_pred.detach())``
        the residual is ``-norm_grad`` in value while its Jacobian w.r.t.
        ``x_pred`` is the identity, so::

            dL/dx_pred = weight * (2/N) * (x_pred - x_target)
                       = -weight * omega * (2/N) * (x_pos - x_pred)
                       = d/dx_pred [ weight * omega * mean_sp((x_pred - x_pos)^2) ]

        which is the GRPO gradient exactly, not to first order. The loss value
        is ``weight * mean_sp[norm_grad ** 2]``, i.e. it carries ``omega``
        squared -- so ``loss_policy`` is not comparable with a directly-written
        ``ratio * adv * mse``, and a negative advantage still shows up as a
        positive loss. Gate on gradients, never on this value.

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


class FlowGRPOTrainer(MeanFlowGRPOTrainer):
    """Ablation of :class:`MeanFlowGRPOTrainer` with the MeanFlow identity dropped.

    The flow map's average velocity over ``[s, t]`` is fed to flow-GRPO's
    Euler-Maruyama kernel as the instantaneous velocity (``v := u``, ``u := v``,
    i.e. ``S = 0`` for every transition, including in the v-mode KL target),
    instead of being lifted with ``v = u + (t - s) * dF/dt``. The central
    difference is never computed, since nothing reads it.

    Everything else -- SDE rollout replay, s|t redraw, kernel coefficients,
    ratio, PPO clip, KL -- is inherited unchanged, so the only variable against
    ``meanflowgrpo_rl`` is the S term.
    """

    TRAINER_KEY = "flowgrpo_rl"

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
