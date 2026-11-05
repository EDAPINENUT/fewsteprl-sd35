"""DMDNFT: the decoupled NFT policy loss with an auxiliary-network probe center.

:class:`DMDNFTTrainer` (``dmdnft_rl``) keeps the S-free
``x_anchor_1 = x_t - t * u_old`` of
:class:`~fewsteprl.trainers.meanflownft.MeanFlowNFTTrainer`, but takes the
instantaneous velocity for the probe center from :attr:`aux_v_model`::

    x_anchor_2 = x_t - t * v_fake(x_t, t)

``aux_v_model`` is a single-time v-predictor trained online with a plain
flow-matching loss on the generator's own rollouts
(:meth:`GeneralRLTrainer._run_aux_epoch`, run on each epoch's rollout pack
before the policy update). ``v_fake`` is therefore an amortized estimate of the
marginal ``E[eps - x_0 | x_t, t]`` over the rollout distribution, whereas
MeanFlowNFTTrainer's ``u_old + (t - r) * sg(dF/dt)`` is a finite-difference
probe of one trajectory; the two agree only in expectation.

Not DMD, despite the name. Full DMD injects a stop-grad ``(fake - real)``
gradient through a surrogate on the rollout output and therefore needs a second
reference network. Here ``v_fake`` only supplies the probe center of
:meth:`GeneralRLTrainer.decoupled_rl_loss`; the target-distribution side is
already provided by ``x_0_rwd`` and ``x_anchor_1``, so there is no real-score
network and no backpropagation through the rollout.

The aux network's update budget is not a per-step aux/policy ratio but the
pair ``fewstep_rl.max_aux_sample`` (rollout samples per GPU in the aux pack)
and ``renoise_per_epoch / aux_v_gradient_accum`` (optimizer steps over them).
"""

from __future__ import annotations

from typing import Optional

import torch

from fewsteprl.trainers.rl_general_base import AnchorInputs, GeneralRLTrainer


class DMDNFTTrainer(GeneralRLTrainer):
    """Decoupled NFT loss whose probe center comes from the online auxiliary v-predictor."""

    TRAINER_KEY = "dmdnft_rl"

    def setup_models(self) -> None:
        """Require ``fewstep_rl.aux_v_enabled``, then run the base setup.

        Checked here, before ``super()``, because the condition is fixed at
        config-load time: a misconfigured run fails before any transformer is
        loaded instead of at the first ``_rl_loss``, after a full
        ``_rl_sampling_phase`` (rollouts plus reward scoring, the most
        expensive part of the epoch).

        ``aux_v_model`` is mandatory for this trainer: it supplies
        ``x_anchor_2``, the probe center ``decoupled_rl_loss`` builds both
        ``w_pos`` and ``w_neg`` from, so without it there is no loss to compute.
        """
        if not self._rl_cfg().aux_v_enabled:
            raise ValueError(
                f"{type(self).__name__} reads x_anchor_2 off aux_v_model, but "
                "fewstep_rl.aux_v_enabled=false, so setup_models would never "
                "build that model and the first policy step would fail -- after "
                "a full rollout and reward-scoring phase. Set "
                "fewstep_rl.aux_v_enabled=true, or use the meanflow_nft trainer, "
                "which derives x_anchor_2 from old_model instead."
            )
        super().setup_models()

    def _s_term(
        self,
        t_cd: torch.Tensor,
        r_cd: torch.Tensor,
        dF_dt: Optional[torch.Tensor],
        like: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """``S = (t - r) * dF/dt`` in ``like``'s dtype, or None when S is exactly 0.

        Neither anchor of this trainer contains S; it is used only by
        :meth:`_average_velocity_from_v` for the v-mode KL target.
        """
        if dF_dt is None:
            # Legal only when every sample has r == t, so S is identically zero.
            # Anything else means the shared central difference is missing.
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
        # (t - r) in raw timestep units, matching the CD denominator (2*eps_cd,
        # also raw), so the time units cancel.
        dt_cd = (t_cd - r_cd).view(
            like.shape[0], *([1] * (like.ndim - 1))
        ).to(wdt)
        return dt_cd * dF_dt.to(wdt)

    def _average_velocity_from_v(
        self,
        v_inst: torch.Tensor,
        t_cd: torch.Tensor,
        r_cd: torch.Tensor,
        dF_dt: Optional[torch.Tensor]=None,
    ) -> torch.Tensor:
        """``u(x_t, r, t) = v(x_t, t) - (t - r) * dF/dt`` -- the identity solved for u.

        The MeanFlow regression target of the v-mode KL: given a reference's
        instantaneous velocity at t, this is what the generator's average
        velocity over ``[r, t]`` should equal. Keeping the whole correction on
        the target side is what makes the KL a MeanFlow loss rather than a
        v-space MSE -- algebraically the same residual
        (``(u + S) - v_ref == u - (v_ref - S)``), but the gradient path then
        carries no S at all.
        """
        s = self._s_term(t_cd, r_cd, dF_dt, like=v_inst)
        return v_inst if s is None else v_inst - s

    @torch.no_grad()
    def _compute_x_anchor_1(self, inputs: AnchorInputs) -> torch.Tensor:
        """``x_anchor_1 = x_t - t * u_old``: the S-free anchor, as in ``MeanFlowNFTTrainer``.

        Self-regression target and origin of the probe direction in
        :meth:`GeneralRLTrainer.decoupled_rl_loss`.
        """
        return inputs.x_t - inputs.t_normed * inputs.u_old

    @torch.no_grad()
    def _compute_x_anchor_2(self, inputs: AnchorInputs) -> torch.Tensor:
        """``x_anchor_2 = x_t - t * v_fake(x_t, t)``, with ``v_fake`` from :attr:`aux_v_model`.

        Unlike ``MeanFlowNFTTrainer``, whose instantaneous velocity is
        self-induced from ``old_model`` as ``u_old + (t - r) * dF/dt``, no
        central difference enters this anchor, and ``inputs.dF_dt`` is
        deliberately unread (the module docstring explains how the two
        estimates differ).

        ``model_u_pred`` stays at its default False: ``aux_v_model`` gets neither
        the flow-map wrapper nor an ``r`` input in ``setup_models``, so it is
        already instantaneous by construction and the ``u -> v`` lift must not
        run. ``r_timesteps`` is passed only to satisfy the signature; that branch
        never forwards it.
        """
        aux = self.aux_v_model
        if aux is None:
            raise RuntimeError(
                "DMDNFTTrainer needs aux_v_model for x_anchor_2; setup_models "
                "only builds it when fewstep_rl.aux_v_enabled is set."
            )
        # Back to the network's dtype for the forward. Exact, not lossy:
        # AnchorInputs.x_t is the widened copy of the network-dtype x_t that
        # _rl_loss built, so narrowing recovers the same values the other
        # velocity queries in this step were evaluated at.
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
        # Cast, not check. MeanFlowNFTTrainer's anchor rejects a non-fp32
        # velocity because S is added to it, and adding S in bf16 would cancel
        # the deviation the anchors measure. Nothing is added here -- v_fake is
        # one raw network output, so the network dtype is expected -- and the
        # only requirement is that the x0 algebra below runs in the work dtype,
        # like x_anchor_1's u_old.
        v_fake = v_fake.to(inputs.x_t.dtype)
        # Explicit raise, not assert: assert vanishes under `python -O`.
        if v_fake.requires_grad:
            raise RuntimeError(
                "v_fake carries grad; x_anchor_2 is a stop-grad probe center, "
                "and aux_v_model has its own optimizer -- a gradient path from "
                "the policy loss into it would train it on the wrong objective."
            )
        return inputs.x_t - inputs.t_normed * v_fake

    def _rl_loss(
        self,
        x_0: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_embeds: torch.Tensor,
        advantages: torch.Tensor,
        t_raw: torch.Tensor,
        r_raw: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Policy loss on one microbatch: ``(loss, info)``.

        Forward-noises ``x_0`` at ``t_raw``, queries the generator and
        ``old_model`` for ``u(x_t, r, t)`` and ``aux_v_model`` for ``v_fake``,
        builds the two anchors and returns the decoupled NFT loss (scaled by
        ``adv_clip_max``) plus ``kl_weight`` times the KL term.
        """
        cfg = self._rl_cfg()
        dtype = x_0.dtype
        batch_size = x_0.shape[0]
        T = float(self.scheduler.config.num_train_timesteps)

        # 1. Forward noising. t_normed is derived in the network dtype, the
        # dtype x_t is built in; computing it in fp32 would change x_t itself.
        t_raw = t_raw.to(device=x_0.device)
        t_normed = t_raw.to(dtype) / T
        t_broadcast = t_normed.view(batch_size, *([1] * (x_0.ndim - 1)))
        eps = self._draw_eps(x_0)
        x_t = (1.0 - t_broadcast) * x_0 + t_broadcast * eps
        t_for_model = t_raw.to(dtype=torch.float32)
        r_for_model = r_raw.to(dtype=torch.float32) if r_raw is not None else None

        # The KL branch is decided before the anchor block because it decides
        # whether the central difference is computed at all. Config-level, hence
        # rank-uniform -- required, since the CD issues FSDP collectives and a
        # rank that skips them deadlocks.
        _, use_v_kl = self._kl_usage()
        kl_weight = float(cfg.kl_weight)

        if self._shared_dF_dt is not None:
            raise RuntimeError(
                "stale _shared_dF_dt on entry to _rl_loss -- a previous step "
                "leaked its central difference (missing _post_compute_v_hook)."
            )

        # Gated on use_v_kl, unlike MeanFlowNFTTrainer, which needs dF/dt
        # unconditionally for x_anchor_2. Here both anchors are S-free (u_old and
        # aux_v_model's own output), so the only consumer is the v-mode KL target
        # ``u_tgt = v_ref - (t - r) * dF/dt``. When that KL runs the hook cannot
        # be skipped: _s_term rejects ``dF_dt=None`` whenever any sample has
        # r != t.
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
            u_old = self._compute_u(
                self.old_model, x_t, t_for_model, prompt_embeds, pooled_embeds,
                r_for_model, with_grad=False,
            ).detach()
            
            wdt = self._rl_work_dtype
            u_theta_f = u_theta.to(wdt)
            u_old_f = u_old.to(wdt)
            x_t_f = x_t.to(wdt)
            x_0_f = x_0.to(wdt)
            t_normed_b_f = t_broadcast.to(wdt)
            spatial = tuple(range(1, x_0.ndim))

            beta = float(cfg.beta)
            adv_clip_max = float(cfg.adv_clip_max)
            if adv_clip_max <= 0:
                raise ValueError(f"adv_clip_max must be > 0, got {adv_clip_max}")

            x_0_pred_u = x_t_f - t_normed_b_f * u_theta_f          # carries grad

            # Packed once, checked once: AnchorInputs rejects a network-dtype
            # operand and an unbroadcast [B] timestep, the two substitutions
            # that are otherwise silent here.
            anchor_inputs = AnchorInputs(
                x_t=x_t_f,
                t_normed=t_normed_b_f,
                u_old=u_old_f,
                u_theta=u_theta_f.detach(),
                # None even when the hook above just populated
                # ``self._shared_dF_dt``: that CD exists for the v-mode KL target
                # only. Handing it to the anchors would let a future edit fold S
                # into x_anchor_2 and quietly turn this into MeanFlowNFTTrainer's
                # anchor.
                dF_dt=None,
                t_cd=t_for_model,
                r_cd=r_for_model,
                prompt_embeds=prompt_embeds,
                pooled_embeds=pooled_embeds,
            )
            x_anchor_1 = self._compute_x_anchor_1(anchor_inputs)
            x_anchor_2 = self._compute_x_anchor_2(anchor_inputs)
            # ``finally`` clears ``_shared_dF_dt``, so the v-mode KL, its only
            # consumer here, needs a local reference. The same guard decides to
            # compute it and to hold it, so the latent-sized CD is freed
            # otherwise.
            shared_dF_dt = self._shared_dF_dt if use_v_kl else None
        finally:
            self._post_compute_v_hook()

        # 3. r_hat: the clipped advantage mapped linearly onto [0, 1].
        adv = advantages.float().clamp(-adv_clip_max, adv_clip_max)
        r_hat = ((adv / adv_clip_max) / 2.0 + 0.5).clamp(0.0, 1.0)

        unscaled, form_info = self.decoupled_rl_loss(
            x_0_pred_u, x_anchor_1, x_anchor_2, x_0_f, r_hat, beta, spatial,
        )
        policy_loss = (unscaled * adv_clip_max).mean()
        loss = policy_loss

        # ``old_deviate`` is the mean squared deviation from old_model in u-space.
        with torch.no_grad():
            s_x0 = (x_anchor_1 - x_anchor_2)
            dev_abs = (x_0_pred_u.detach() - x_anchor_1).abs().mean()
            info = {
                "loss_policy": float(policy_loss.detach()),
                "loss_unscaled": float(unscaled.mean().detach()),
                "t_mean": float(t_raw.mean().item()),
                "r_hat_mean": float(r_hat.mean().item()),
                "advantage_mean": float(advantages.mean().item()),
                "advantage_std": float(advantages.std().item()),
                "x0_norm_mean": float((x_0_f ** 2).mean().item()),
                "old_deviate": float(((u_theta_f - u_old_f) ** 2).mean().item()),
                # Diagnostics are emitted unconditionally, on every rank and for
                # every trainer: _reduce_metrics all-reduces per key with no
                # key-set reconciliation, so a conditional key deadlocks until
                # the NCCL timeout instead of raising.
                # Here s_x0 = x_anchor_1 - x_anchor_2 = t * (v_fake - u_old), the
                # gap between the anchors (there is no S term in them).
                "s_x0_abs_mean": float(s_x0.abs().mean()),
                # mean|s_x0| / mean|dev| (a ratio of means).
                "s_rel_median": float(
                    (s_x0.abs().mean() / dev_abs.clamp(min=1e-12))
                ),
                **form_info,
            }

        # 4. KL against the frozen reference, via the shared compute_kl_term so
        # the Stage-1 weighting and the ``kl_*`` metric keys match every other
        # trainer's.
        kl, kl_info = self.compute_kl_term(
            x_t, t_for_model, prompt_embeds, pooled_embeds, r_for_model,
            shared_dF_dt, u_theta_f, spatial,
        )
        loss = loss + kl_weight * kl
        info["loss_kl"] = float(kl.detach())
        # Merged unconditionally: compute_kl_term pre-seeds every key with its
        # inactive value, so the key set is the same for the u-mode KL, the
        # v-mode KL and KL off, as _reduce_metrics requires.
        info.update(kl_info)

        info["loss_total"] = float(loss.detach())
        return loss, info

