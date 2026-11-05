"""MeanFlowNFT: the decoupled NFT policy loss on a MeanFlow flow map.

:class:`MeanFlowNFTTrainer` (``meanflow_nft``) builds both anchors from
``old_model``'s average velocity ``u_old = u(x_t, r, t)``::

    x_anchor_1 = x_t - t * u_old                            # S-free
    x_anchor_2 = x_t - t * (u_old + (t - r) * sg(dF/dt))    # self-induced v

``x_anchor_2`` uses the instantaneous velocity recovered through the MeanFlow
identity ``v(x_t, t) = u(x_t, r, t) + (t - r) * dF/dt``, where ``dF/dt`` is a
stop-grad central-difference estimate of the time derivative of
``old_model``'s ``u`` along the trajectory, computed once per step. The outer
``t`` is ``t/T``; ``(t - r)`` is in raw timestep units, matching ``dF/dt``.

The policy loss is :meth:`GeneralRLTrainer.decoupled_rl_loss`, which splits the
fused NFT MSE into ``self_regression + reward_augmentation``. The two forms are
gradient-identical, but their loss values differ by a cross term, so compare
runs on gradients or rewards, not on loss values. The optional KL term comes
from :meth:`GeneralRLTrainer.compute_kl_term`.

:class:`NFTTrainer` (``nft_rl``) is the S-term ablation: the average velocity
is used as the instantaneous one (``S = 0``, no central difference), so both
anchors reduce to ``x_t - t * u_old``.
"""

from __future__ import annotations

from typing import Optional

import torch

from fewsteprl.trainers.rl_general_base import AnchorInputs, GeneralRLTrainer


class MeanFlowNFTTrainer(GeneralRLTrainer):
    """Decoupled NFT loss with a self-induced instantaneous-velocity probe center."""

    TRAINER_KEY = "meanflow_nft"

    #: Apply the loss weighting of Stage 1 (flow-map pretraining) to the v-mode
    #: KL, which regresses ``u_theta`` onto the PDE-residual target
    #: ``u_tgt = v_ref - (t - r) * dF/dt``, i.e. Stage 1's flow-map objective.
    #: Two layers, applied per sample in :meth:`GeneralRLTrainer.compute_kl_term`:
    #:
    #:   1. ``* get_train_weight(t)``: the beta08 curve ``t * (1 - t)^0.5``,
    #:      normalized so ``sum(weights) == T``. It zeroes the degenerate
    #:      ``t == T`` endpoint, where a velocity target is meaningless.
    #:   2. Mode rebalance: every non-diffusion (``r != t``) sample is rescaled
    #:      to the global mean of the diffusion (``r == t``) slice. Without it
    #:      the ``r == 0`` slice, which has the largest ``(t - r)`` and hence
    #:      the largest S, dominates the batch mean, and the KL reports
    #:      S-estimation noise rather than policy divergence.
    #:
    #: False leaves the v-mode KL a plain per-sample MSE. The u-mode KL is
    #: never weighted.
    _kl_use_stage1_weighting: bool = True

    #: Weight curve for layer 1. :meth:`GeneralRLTrainer._v_kl_reweight` builds
    #: its own FlowMapScheduler with this type: the shared ``flowmap_scheduler``
    #: uses ``weight_type="uniform"`` (a flat curve, which would make layer 1 a
    #: no-op) and also serves the rollout grid, so it is left untouched.
    _kl_weight_type: str = "beta08"

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
        regression target of the v-mode KL: given a reference's instantaneous
        velocity at t, this is what the generator's average velocity over
        ``[r, t]`` should equal. Keeping the whole correction on the target side
        is what makes the KL a MeanFlow loss rather than a v-space MSE --
        algebraically the same residual (``(u + S) - v_ref == u - (v_ref - S)``),
        but the gradient path then carries no S at all.
        """
        s = self._s_term(t_cd, r_cd, dF_dt, like=v_inst)
        return v_inst if s is None else v_inst - s

    @torch.no_grad()
    def _compute_x_anchor_1(self, inputs: AnchorInputs) -> torch.Tensor:
        """``x_anchor_1 = x_t - t * u_old``: the S-free anchor.

        Self-regression target and origin of the probe direction in
        :meth:`GeneralRLTrainer.decoupled_rl_loss`.
        """
        return inputs.x_t - inputs.t_normed * inputs.u_old

    @torch.no_grad()
    def _compute_x_anchor_2(self, inputs: AnchorInputs) -> torch.Tensor:
        """``x_anchor_2 = x_t - t * v_inst`` with ``v_inst = u_old + (t - r) * dF/dt``.

        The probe center of :meth:`GeneralRLTrainer.decoupled_rl_loss`, built
        from the instantaneous velocity that the MeanFlow identity recovers
        from ``old_model``.
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
        ``old_model`` for ``u(x_t, r, t)``, builds the two anchors and returns
        the decoupled NFT loss (scaled by ``adv_clip_max``) plus ``kl_weight``
        times the KL term.
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

        # The KL branch is decided before the anchor block, which must know
        # whether to keep the shared dF/dt past its ``finally``. Config-level,
        # hence rank-uniform.
        _, use_v_kl = self._kl_usage()
        kl_weight = float(cfg.kl_weight)


        # Explicit raise, not assert: assert vanishes under `python -O`.
        if self._shared_dF_dt is not None:
            raise RuntimeError(
                "stale _shared_dF_dt on entry to _rl_loss -- a previous step "
                "leaked its central difference (missing _post_compute_v_hook)."
            )

        # 2. Average velocities and anchors. S is never folded into u_theta or u_old.
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
                dF_dt=self._shared_dF_dt,
                t_cd=t_for_model,
                r_cd=r_for_model,
                prompt_embeds=prompt_embeds,
                pooled_embeds=pooled_embeds,
            )
            x_anchor_1 = self._compute_x_anchor_1(anchor_inputs)
            x_anchor_2 = self._compute_x_anchor_2(anchor_inputs)
            # ``finally`` clears ``_shared_dF_dt``, so the v-mode KL, its only
            # consumer outside this block, needs a local reference. Held only
            # when that KL runs, so the latent-sized CD is freed otherwise;
            # reusing it keeps the KL's S identical to the anchors' S.
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

        # ``old_deviate`` is the mean squared deviation from old_model in u-space,
        # so it contains no S term.
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
                "s_x0_abs_mean": float(s_x0.abs().mean()),
                # mean|S_x0| / mean|dev| (a ratio of means): the size of S
                # relative to the policy deviation, i.e. how much cancellation
                # adding S to the velocity in low precision would cause.
                "s_rel_median": float(
                    (s_x0.abs().mean() / dev_abs.clamp(min=1e-12))
                ),
                **form_info,
            }
        kl, kl_info = self.compute_kl_term(
            x_t, t_for_model, prompt_embeds, pooled_embeds, r_for_model, shared_dF_dt, u_theta_f, spatial,
        )
        loss = loss + kl_weight * kl
        info["loss_kl"] = float(kl.detach())
        # Merged unconditionally: compute_kl_term pre-seeds every key with its
        # inactive value, so the key set is the same for the u-mode KL, the
        # v-mode KL and KL off, as _reduce_metrics requires.
        info.update(kl_info)

        info["loss_total"] = float(loss.detach())
        return loss, info


class NFTTrainer(MeanFlowNFTTrainer):
    """S-term ablation of :class:`MeanFlowNFTTrainer`: the MeanFlow identity is dropped.

    The flow map's average velocity is used as the instantaneous velocity
    (``v := u`` and ``u := v``), i.e. ``S = (t - r) * dF/dt`` is taken to be 0
    for every sample. Both anchors then collapse to ``x_t - t * u_old`` and the
    loss is plain DiffusionNFT on ``x_0 = x_t - t * u``. The central difference
    is never computed, since nothing reads it.

    Everything else -- sampler, (t, r) draw, KL, optimizer loop -- is inherited
    unchanged, so the only difference from ``meanflow_nft`` is the S term.
    """

    TRAINER_KEY = "nft_rl"

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
