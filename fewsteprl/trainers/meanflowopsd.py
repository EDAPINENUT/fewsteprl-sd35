"""MeanFlowOPSDTrainer: DiffusionOPSD-style reward-gradient fitting on a MeanFlow flow map.

Each RL epoch has two phases.

Sampling (:meth:`_rl_sampling_phase`). Rollout, reward scoring and group
advantages are :class:`GeneralRLTrainer`'s. Then, per trajectory:

* :meth:`_resolve_query` draws a query noise level ``sigma_q`` (``opsd_sigma_mode``)
  and builds the query state ``z_q``, by default by forward-noising the rollout
  endpoint ``x_0``;
* :meth:`_opsd_build_anchor` draws ``r`` given ``t = sigma_q * T`` (``opsd_r_mode``)
  and builds the lifted v-form anchor
  ``anchor_v = z_q - sigma_q * (u_old(z_q, r, t) + S)``, ``S = (t - r) * sg(dF/dt)``
  (``S = 0`` under ``disable_cd``), together with
  ``anchor_u = z_q - sigma_q * u_old(z_q, 0, t)``, the old model's direct jump to
  the clean end;
* :meth:`_opsd_tr_step` builds the two reward targets with a trust-region walk:
  ``opsd_n_ascent`` unit-normalized steps along ``+grad R`` (``xp``) or
  ``-grad R`` (``xm``), each projected back into the ball of radius
  ``opsd_rho * ||x_start||``. The walk starts at ``anchor_u``, so ``grad R`` is
  taken around an image the flow map produces; the resulting displacement is
  then applied to ``anchor_v``.

Fitting (:meth:`_rl_train_phase`). At the packed ``(z_q, t, r)`` the live
prediction is ``x_0_pred_v = z_q - sigma_q * (u_theta + S)``, and
:meth:`decoupled_opsd_loss` sums a proximal pull toward the lifted anchor, a pull
toward ``xp`` and a push away from ``xm``, weighted per sample by
:meth:`opsd_weights` from the advantage-derived ``omega``; the KL term is added.

Tensor names:

    x_0                rollout endpoint
    z_q, sigma_q       query state and its noise level
    x_0_pred_v         live prediction, the only tensor carrying grad
    x_anchor_1         lifted anchor z_q - sigma_q * (u_old + S), recomputed in the
                       loss: self-regression target and origin of the probe direction
    x_anchor_2         probe centre; the same point as x_anchor_1 in this trainer
    anchor_v           the target builder's name for that anchor (packed as opsd_y_0)
    anchor_u           z_q - sigma_q * u_old(z_q, 0, t); shift = anchor_u - anchor_v
    xp / xm            reward-ascent / reward-descent targets
    probe_pos / _neg   x_anchor_2 +- beta * (x_0_pred_v - x_anchor_1), in opsd_weights
    x_start            origin of a trust-region walk
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from fewsteprl.trainers.rl_general_base import AnchorInputs, GeneralRLTrainer

_DIFFERENTIABLE_REWARDS = frozenset({'aesthetic', 'clipscore', 'hpsv2', 'pickscore'})


class MeanFlowOPSDTrainer(GeneralRLTrainer):
    """DiffusionOPSD on a MeanFlow flow-map generator, v-form (see the module docstring)."""

    TRAINER_KEY = "meanflowopsd_rl"

    #: Must be identical on every rank: _reduce_metrics all-reduces per key with
    #: no reconciliation, so a rank-dependent key set hangs instead of raising.
    #: The ``opsd_sigma_q_{std,min,max}`` entries are therefore emitted on every
    #: sigma mode; on "fixed" they read (0, sigma*, sigma*).
    _EXTRA_DIAGNOSTIC_KEYS: tuple[str, ...] = ('s_x0_abs_mean', 's_rel_median', 'opsd_delta_res', 'opsd_sigma_q', 'opsd_r_mean', 'opsd_r_frac_zero', 'opsd_frame_shift', 'opsd_sigma_q_std', 'opsd_sigma_q_min', 'opsd_sigma_q_max')

    _EXTRA_LOSS_KEYS: tuple[str, ...] = ('self_regression_loss', 'reward_augmentation_loss', 'loss_kl_unweighted', 'kl_t_weight_mean', 'kl_rebalance_mean', 'opsd_pull_pos', 'opsd_push_neg', 'opsd_omega_pos_mean', 'opsd_omega_neg_mean')

    #: Sample-pack keys written by the sampling phase and required by the
    #: fitting phase (not defined on GeneralRLTrainer): the query state ``z_q``
    #: (``opsd_z_q``), the anchor (``opsd_y_0``) and the reward targets ``xp`` /
    #: ``xm`` (``opsd_y_plus`` / ``opsd_y_minus``).
    _OPSD_KEYS: tuple[str, ...] = ('opsd_z_q', 'opsd_y_0', 'opsd_y_plus', 'opsd_y_minus')

    _kl_use_stage1_weighting = True

    _kl_weight_type = 'beta08'

    def __init__(self, config) -> None:
        super().__init__(config)
        # Resident differentiable scorer, distinct from the base's
        # ``_reward_scorer``. See :meth:`_ensure_grad_scorer` for why the base
        # one cannot be reused.
        self._opsd_grad_scorer: Optional[Any] = None
        # Populated by :meth:`_rl_sampling_phase`, read by :meth:`_rl_loss`.
        self._opsd_last_stats: dict[str, float] = {}

    def _average_velocity_from_v(
        self,
        v_inst: torch.Tensor,
        t_cd: torch.Tensor,
        r_cd: torch.Tensor,
        dF_dt: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``u(x_t, r, t) = v(x_t, t) - (t - r) * dF/dt`` -- exact inverse of the lift."""
        s = self._s_term(t_cd, r_cd, dF_dt, like=v_inst)
        return v_inst if s is None else v_inst - s

    @staticmethod
    def _bcast(sigma_q: Any, like: torch.Tensor) -> torch.Tensor:
        """Broadcast a ``[B]`` (or scalar) noise level to ``like``'s rank as ``[B, 1, ...]``.

        A bare ``[B]`` tensor multiplied against ``[B, C, H, W]`` right-aligns
        onto the width axis: it raises only when ``B != W`` and silently scales
        per column when ``B == W``. A Python float or 0-d tensor (e.g. the fixed
        query sigma passed to :meth:`_opsd_query_state`) is expanded to ``[B]``
        first.
        """
        if not torch.is_tensor(sigma_q):
            sigma_q = torch.as_tensor(
                sigma_q, device=like.device, dtype=torch.float32,
            )
        if sigma_q.ndim == 0:
            sigma_q = sigma_q.expand(like.shape[0])
        if sigma_q.ndim != 1 or sigma_q.shape[0] != like.shape[0]:
            raise ValueError(
                f"sigma_q must be [B] with B == {like.shape[0]}, got "
                f"{tuple(sigma_q.shape)}."
            )
        return sigma_q.to(device=like.device, dtype=like.dtype).view(
            -1, *([1] * (like.ndim - 1))
        )

    @torch.no_grad()
    def _build_opsd_targets(
        self,
        x_0: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_embeds: torch.Tensor,
        prompts: list[str],
        z_q: torch.Tensor,
        sigma_q: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
        """Build the anchor and the two reward targets for every sample.

        Returns ``(anchor, xp, xm, stats)``; the tensors are fp32 and shaped
        like ``x_0``. Works in chunks of ``opsd_target_chunk`` samples: per
        chunk, :meth:`_opsd_build_anchor` builds the anchor (and the frame
        shift), the reward gradient at the anchor is computed once and passed
        to both directions as ``first_grad`` (ignored on the shifted path, see
        :meth:`_opsd_tr_step`), and :meth:`_opsd_tr_step` returns ``xp``
        (``+1``) and ``xm`` (``-1``). The reward-gradient modules are offloaded
        at the end.

        The loop count is a config constant times a rank-uniform sample count,
        so no rank can skip a forward.
        """
        cfg = self._rl_cfg()
        device = next(self.generator.parameters()).device
        n = x_0.shape[0]
        chunk = max(1, int(cfg.opsd_target_chunk))
        dir_mode = str(cfg.opsd_dir_mode)
        need_grad = dir_mode == "grad" and int(cfg.opsd_n_ascent) > 0
        if need_grad:
            self._ensure_grad_scorer()

        T = float(self.scheduler.config.num_train_timesteps)
        sigma_q = sigma_q.to(device=device, dtype=torch.float32)

        anchor_out = torch.empty_like(x_0, dtype=torch.float32)
        xp_out = torch.empty_like(x_0, dtype=torch.float32)
        xm_out = torch.empty_like(x_0, dtype=torch.float32)
        res_sum = 0.0

        net_dtype = next(self.generator.parameters()).dtype
        self.old_model.eval()

        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            zq_c = z_q[start:end].to(device=device, dtype=net_dtype)
            x0_c = x_0[start:end].to(device=device, dtype=torch.float32)
            pe_c = prompt_embeds[start:end].to(device=device, dtype=net_dtype)
            pp_c = pooled_embeds[start:end].to(device=device)
            prm_c = list(prompts[start:end])
            s_c = sigma_q[start:end]

            # Per-sample query timesteps. ``r_c`` is a placeholder:
            # :meth:`_opsd_build_anchor` draws its own ``r``.
            t_c = (s_c * T).to(torch.float32)
            r_c = torch.zeros_like(t_c)

            # The anchor must be the lifted v-form one, so the targets live in
            # the same coordinate the loss fits in (a u-form
            # ``z_q - sigma * u_old`` here would not).
            x_anchor = self._opsd_build_anchor(
                zq_c, t_c, r_c, pe_c, pp_c, s_c, start, end, x0_c,
            )

            g0 = None
            if need_grad:
                with torch.enable_grad():
                    x_g = x_anchor.detach().requires_grad_(True)
                    rg = self._reward_of_latents_grad(x_g, prm_c)
                    (g0,) = torch.autograd.grad(rg.sum(), x_g)
                g0 = g0.detach()

            with torch.enable_grad():
                xp = self._opsd_tr_step(
                    x_anchor, prm_c, +1.0, x_end=x0_c, first_grad=g0,
                )
                xm = self._opsd_tr_step(
                    x_anchor, prm_c, -1.0, x_end=x0_c, first_grad=g0,
                )

            anchor_out[start:end] = x_anchor
            xp_out[start:end] = xp
            xm_out[start:end] = xm
            res_sum += float(
                (
                    (xp - x_anchor).flatten(1).norm(dim=1)
                    / (x_anchor.flatten(1).norm(dim=1) + 1e-8)
                ).sum()
            )
            del g0

        self._opsd_release_grad_memory()
        torch.cuda.empty_cache()

        return anchor_out, xp_out, xm_out, {
            "opsd_delta_res": res_sum / max(1, n),
            # Mean of the per-sample query sigma; its spread is logged as
            # opsd_sigma_q_{std,min,max}.
            "opsd_sigma_q": float(sigma_q.mean()),
        }

    @torch.no_grad()
    def _compute_x_anchor_1(self, inputs: AnchorInputs) -> torch.Tensor:
        """``x_anchor_1 = z_q - sigma_q * v_old`` -- the lifted anchor.

        ``v_old = u_old + S`` comes from :meth:`_instantaneous_velocity_from_u`,
        the same lift as :meth:`_opsd_clean_output` and
        :meth:`_opsd_build_anchor`, so the anchor is in the coordinate of
        ``x_0_pred_v`` and ``x_0_pred_v - x_anchor_1 = -sigma_q * (u_theta - u_old)``
        exactly: a pure policy deviation that vanishes at initialisation.

        A u-form anchor ``z_q - sigma_q * u_old`` would leave a residual of
        ``-sigma_q * S`` even when ``u_theta == u_old``, so the self-regression
        term ``W * mean[(x_0_pred_v - x_anchor_1)^2]`` would measure S instead of
        policy drift.
        """
        v_inst = self._instantaneous_velocity_from_u(
            inputs.u_old, inputs.t_cd, inputs.r_cd, inputs.dF_dt,
        )
        if v_inst.dtype not in (torch.float32, torch.float64):
            raise TypeError(
                f"the lift must stay in fp32/fp64, got {v_inst.dtype}."
            )
        if v_inst.requires_grad:
            raise RuntimeError(
                "_compute_x_anchor_1 must be detached; it is a stop-grad "
                "reference point."
            )
        return inputs.x_t - inputs.t_normed * v_inst

    @torch.no_grad()
    def _compute_x_anchor_2(self, inputs: AnchorInputs) -> torch.Tensor:
        """``x_anchor_2 = z_q - sigma_q * v_old`` -- the probe center.

        The same point as :meth:`_compute_x_anchor_1` in this trainer.
        """
        v_inst = self._instantaneous_velocity_from_u(
            inputs.u_old, inputs.t_cd, inputs.r_cd, inputs.dF_dt,
        )
        if v_inst.dtype not in (torch.float32, torch.float64):
            raise TypeError(
                f"_instantaneous_velocity_from_u must return fp32/fp64, got "
                f"{v_inst.dtype}. Adding S in the network's bf16 loses precision "
                "to cancellation."
            )
        if v_inst.requires_grad:
            raise RuntimeError(
                "_instantaneous_velocity_from_u must be detached; x_anchor_2 is "
                "a stop-grad probe center."
            )
        return inputs.x_t - inputs.t_normed * v_inst

    def _decode01_grad(self, latents: torch.Tensor) -> torch.Tensor:
        """Differentiable latent -> image in [0, 1].

        Same arithmetic as ``fewsteprl.utils.image.decode_latents_to_tensor``
        (and DiffusionOPSD's ``_decode01``): the VAE scaling/shift factors, then
        ``/2 + 0.5``, so the images scored here are on the same scale as the
        ones that produced the rewards. Restated because that helper is
        ``@torch.no_grad()``. ``clamp(0, 1)`` zeroes the gradient on saturated
        pixels, which cannot move in the direction the reward wants anyway.

        The decode runs in fp32, not the pipeline dtype. The VAE is loaded in
        ``model.dtype`` (bf16 in the shipped configs), but only the DIRECTION of
        the reward gradient survives into the target (each ascent step is
        normalized to a unit vector), and a bf16 backward through the decoder
        rotates that direction substantially (~21 degrees against fp32 on a
        decoder-shaped stack) without changing any logged metric.
        :meth:`_opsd_vae_decode_fp32` decodes through a cached fp32 copy, so
        ``self.vae`` -- which the rollout/eval decode uses in the pipeline dtype
        under ``no_grad`` -- is left untouched, and :meth:`_opsd_autocast_off`
        keeps an enclosing autocast region from pulling the decode back to bf16.
        """
        vae = self.vae
        scaling = getattr(vae.config, "scaling_factor", 1.0)
        shift = getattr(vae.config, "shift_factor", 0.0)
        scaled = latents.float() / scaling + shift
        with self._opsd_autocast_off():
            images = self._opsd_vae_decode_fp32(scaled)
        return (images / 2 + 0.5).clamp(0, 1).float()

    def _ensure_grad_scorer(self) -> None:
        """Build (or move back to the accelerator) the fp32 scorer used for reward gradients.

        Separate from the base's ``_reward_scorer``, a ``MultiScorer`` that
        lives on CPU between calls and quantizes its input. This one is fp32
        with frozen (``requires_grad_(False)``) weights but a live graph through
        the INPUT, which is what ``torch.autograd.grad(R, x)`` needs.

        Built once; :meth:`_opsd_release_grad_memory` offloads it to CPU after
        each target build and this call moves it back.
        """
        cfg = self._rl_cfg()
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        if self._opsd_grad_scorer is not None:
            self._opsd_move_grad_modules(device)
            return

        kind = str(cfg.opsd_reward_kind)
        # Honour the same checkpoint root the base scorer uses, so both find
        # the same frozen weights.
        if cfg.reward_ckpt_path:
            from fewsteprl.rewards.reward_ckpt_path import set_ckpt_path
            set_ckpt_path(cfg.reward_ckpt_path)

        self._opsd_grad_scorer = self._load_grad_scorer(kind, device)

    def _instantaneous_velocity_from_u(
        self,
        u_old: torch.Tensor,
        t_cd: torch.Tensor,
        r_cd: torch.Tensor,
        dF_dt: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``v(x_t, t) = u(x_t, r, t) + (t - r) * dF/dt``."""
        s = self._s_term(t_cd, r_cd, dF_dt, like=u_old)
        return u_old if s is None else u_old + s

    def _load_grad_scorer(self, kind: str, device: torch.device) -> Any:
        """Load the differentiable fp32 scorer for ``opsd_reward_kind`` (as in DiffusionOPSD).

        ``kind="mixed"`` loads one scorer per ``reward_fn`` entry, so the
        gradient follows the same weighted sum as ``reward_fn``; it costs one
        reward backward per component.
        """
        if kind == "mixed":
            weights = {k: float(v) for k, v in dict(self._rl_cfg().reward_fn).items()}
            # Only the differentiable backends can contribute a direction; a
            # non-differentiable one (ocr, geneval2) would silently contribute
            # nothing, so refuse rather than train on a partial gradient.
            bad = [k for k in weights if k not in _DIFFERENTIABLE_REWARDS]
            if bad:
                raise ValueError(
                    f"opsd_reward_kind='mixed' needs every reward in reward_fn "
                    f"to be differentiable, but {bad} are not "
                    f"(supported: {sorted(_DIFFERENTIABLE_REWARDS)}). Set "
                    "fewstep_rl.opsd_reward_kind to a single differentiable "
                    "reward instead."
                )
            if not weights:
                raise ValueError("opsd_reward_kind='mixed' needs a non-empty reward_fn.")
            return {
                "weights": weights,
                "scorers": {k: self._load_grad_scorer(k, device) for k in weights},
            }

        if kind == "hpsv2":
            from fewsteprl.rewards.hpsv2_scorer import HPSv2Scorer
            scorer = HPSv2Scorer(dtype=torch.float32, device=device)
        elif kind == "clipscore":
            from fewsteprl.rewards.clip_scorer import ClipScorer
            scorer = ClipScorer(device=device)
            scorer.dtype = torch.float32
        elif kind == "pickscore":
            from fewsteprl.rewards.pickscore_scorer import PickScoreScorer
            scorer = PickScoreScorer(device=device, dtype=torch.float32)
        elif kind == "aesthetic":
            from fewsteprl.rewards.aesthetic_scorer import AestheticScorer
            scorer = AestheticScorer(dtype=torch.float32, device=device)
        else:
            raise ValueError(
                f"fewstep_rl.opsd_reward_kind={kind!r} has no differentiable "
                f"scorer. Supported: {sorted(_DIFFERENTIABLE_REWARDS)} (or "
                "'mixed'). Non-differentiable rewards (ocr, geneval2) cannot "
                "supply a target direction -- use opsd_dir_mode='rand'/'noop' "
                "if you want the reward-free controls."
            )
        scorer.requires_grad_(False)   # freeze WEIGHTS; the input graph stays live
        return scorer

    @staticmethod
    def _opsd_autocast_off():
        """Force fp32 math regardless of any enclosing autocast region.

        A null context without CUDA, where there is no autocast region to
        suppress.
        """
        if not torch.cuda.is_available():
            from contextlib import nullcontext
            return nullcontext()
        return torch.autocast(device_type="cuda", enabled=False)

    @torch.no_grad()
    def _opsd_build_anchor(
        self,
        z_q: torch.Tensor,
        t_raw: torch.Tensor,
        r_raw: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_embeds: torch.Tensor,
        sigma_q: Any,
        chunk_start: int,
        chunk_end: int,
        x_0: torch.Tensor,
    ) -> torch.Tensor:
        """Draw ``r``, build the v-form anchor ``anchor_v``, and measure the frame shift.

        * ``r`` comes from :meth:`_opsd_draw_r` and is stashed so the fitting
          phase replays the same ``(t, r)``; ``r_raw`` is not used.
        * ``anchor_v = z_q - sigma * (u_old(z_q, r, t) + S)``, the lifted anchor,
          with the same central difference the fitting phase lifts with.
        * ``shift = anchor_u - anchor_v`` with ``anchor_u = z_q - sigma * u_old(z_q, 0, t)``
          is stored on ``self._opsd_frame_shift`` for :meth:`_opsd_tr_step`,
          which builds the targets around ``anchor_u`` and moves the resulting
          displacement back onto ``anchor_v``.

        Returns ``anchor_v``, so the packed ``opsd_y_0``, ``opsd_delta_res`` and
        the loss's ``x_anchor_1`` / ``x_anchor_2`` all refer to the same point.

        ``u_old(z_q, 0, t)`` costs one extra no-grad forward instead of being
        inferred from the central difference: the MeanFlow identity that would
        give the shift in closed form holds only for an exact flow map, and a
        trained network's residual grows with S.
        """
        net_dtype = next(self.generator.parameters()).dtype
        device = z_q.device

        r_draw = self._opsd_draw_r(
            t_raw,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
            inner_epoch_idx=int(self._rl_epoch),
            device=device,
        )
        # Persist so the fitting phase queries the SAME (t, r) pair.
        self._opsd_stash_r(chunk_start, chunk_end, r_draw)

        u_old = self._compute_u(
            self.old_model, z_q, t_raw, prompt_embeds, pooled_embeds, r_draw,
            with_grad=False,
        ).detach().float()

        # The same central difference the fitting phase lifts with, computed
        # here because the target builder runs outside any fitting step's
        # _pre_compute_v_hook scope.
        self._pre_compute_v_hook(
            x_t=z_q.to(net_dtype), t_for_model=t_raw, r_for_model=r_draw,
            prompt_embeds=prompt_embeds, pooled_embeds=pooled_embeds,
            x_0=x_0.to(net_dtype),
        )
        try:
            v_inst = self._instantaneous_velocity_from_u(
                u_old, t_raw, r_draw, self._shared_dF_dt,
            )
        finally:
            self._post_compute_v_hook()

        if v_inst.requires_grad:
            raise RuntimeError(
                "_opsd_build_anchor must stay stop-grad: anchor_v is the centre the "
                "reward targets are constructed around."
            )
        anchor_v = z_q.float() - self._opsd_sigma_scale_build(sigma_q, v_inst) * v_inst

        # The flow map's direct jump to the clean end, r = 0 exactly. The
        # few-step sampler's last step also ends at r = 0 (t = 0.5 -> 0 on the
        # 4-step, shift = 3.0 grid), so grad R taken around this point is the
        # gradient of the reward of an image the model can produce.
        u_r0 = self._compute_u(
            self.old_model, z_q, t_raw, prompt_embeds, pooled_embeds,
            torch.zeros_like(t_raw), with_grad=False,
        ).detach().float()
        anchor_u = z_q.float() - self._opsd_sigma_scale_build(sigma_q, u_r0) * u_r0

        shift = (anchor_u - anchor_v).detach()
        if shift.requires_grad:
            raise RuntimeError(
                "_opsd_build_anchor's frame shift must be stop-grad: it is a "
                "frozen reparametrisation of the target space, not a trainable "
                "quantity."
            )
        self._opsd_frame_shift = shift
        self._opsd_shift_stat = float(shift.abs().mean())
        return anchor_v

    def _opsd_clean_output(
        self,
        z_q_f: torch.Tensor,
        sigma_scale: Any,
        u_theta_f: torch.Tensor,
        *,
        t_for_model: torch.Tensor,
        r_for_model: torch.Tensor,
        dF_dt: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """v-form live prediction: ``x_0_pred_v = z_q - sigma_q * (u + (t - r) * sg(dF/dt))``.

        The lift goes through :meth:`_instantaneous_velocity_from_u`, the same
        method :meth:`_opsd_build_anchor` and the anchors use, so the S
        convention and units cannot drift between the live prediction and the
        reference points it is compared against. ``dF_dt`` arrives detached, so
        the correction is a constant offset and the gradient path is the same
        as for the u-form ``z_q - sigma_q * u``.

        When ``dF_dt is None``, :meth:`_s_term` returns None and this reduces to
        the u-form. Two configs reach that: ``r == t`` for every sample
        (``diffusion_ratio=1.0``), where S is arithmetically zero; and
        ``disable_cd=true``, where S is defined to be zero with ``r`` still random.
        """
        v_inst = self._instantaneous_velocity_from_u(
            u_theta_f, t_for_model, r_for_model, dF_dt,
        )
        if v_inst.dtype not in (torch.float32, torch.float64):
            raise TypeError(
                f"the lift must stay in fp32/fp64, got {v_inst.dtype}: folding "
                "S in the network's bf16 destroys the S cancellation between "
                "x_0_pred_v and its anchor."
            )
        return z_q_f - sigma_scale * v_inst

    def _opsd_collect_r(self, n: int) -> Optional[torch.Tensor]:
        """Assemble the stashed per-chunk ``r`` into one ``[n]`` tensor and clear the stash.

        Returns None unless the stashed chunks cover all ``n`` rows.
        """
        buf = getattr(self, "_opsd_r_chunks", None)
        if not buf:
            return None
        out = torch.zeros(n, dtype=torch.float32)
        covered = 0
        for (s, e), v in buf.items():
            if e <= n and v.shape[0] == e - s:
                out[s:e] = v
                covered += e - s
        self._opsd_r_chunks = {}
        return out if covered == n else None

    def _opsd_draw_r(
        self,
        t_raw: torch.Tensor,
        *,
        chunk_start: int,
        chunk_end: int,
        inner_epoch_idx: int,
        device: torch.device,
    ) -> torch.Tensor:
        """``r`` for a chunk: drawn given ``t``, or replayed from the anchor build.

        * From :meth:`_opsd_build_anchor` (sampling phase) no replay buffer is
          set, so ``r`` is drawn from ``_draw_training_s_given_t``.
        * From the fitting loop ``_opsd_r_replay`` holds the ``r`` each anchor
          was built at, so the matching slice is replayed.

        The replay keeps the anchor and the live prediction at the identical
        ``(t, r)``; otherwise the two differ by the model's MeanFlow-identity
        residual, which grows with S. ``opsd_r_mode="zero"`` gives ``r = 0`` in
        both phases.
        """
        if str(self._rl_cfg().opsd_r_mode) == "zero":
            return torch.zeros_like(t_raw)

        replay = getattr(self, "_opsd_r_replay", None)
        if replay is not None and replay.shape[0] >= chunk_end:
            return replay[chunk_start:chunk_end].to(
                device=device, dtype=torch.float32,
            )

        dtype = next(self.generator.parameters()).dtype
        return self._draw_training_s_given_t(
            t_raw,
            state={},          # no per-inner-epoch draw state
            chunk_start=chunk_start,
            chunk_end=chunk_end,
            j_idx=0,           # one fitting query per sample, so no j axis
            inner_epoch_idx=inner_epoch_idx,
            device=device,
            dtype=dtype,
        )

    def _opsd_grad_modules(self) -> list[Any]:
        """Every module kept for the reward-gradient path (grad scorers, fp32 VAE copy)."""
        mods: list[Any] = []
        sc = self._opsd_grad_scorer
        if isinstance(sc, dict):
            mods.extend(sc.get("scorers", {}).values())
        elif sc is not None:
            mods.append(sc)
        fp32_vae = getattr(self, "_opsd_vae_fp32", None)
        if fp32_vae is not None:
            mods.append(fp32_vae)
        return mods

    def _opsd_move_grad_modules(self, device: torch.device) -> None:
        """Move the reward-gradient modules to ``device`` in place."""
        for m in self._opsd_grad_modules():
            try:
                m.to(device)
            except Exception:      # noqa: BLE001 - a non-Module scorer
                continue
            # Several scorers cache their own device and use it to place
            # inputs (e.g. ClipScorer._process(...).to(self.device)), so the
            # attribute has to follow the weights. On diffusers modules
            # (AutoencoderKL) ``device`` is a read-only property derived from
            # the parameters, so the assignment raises AttributeError; those
            # already report the right device after ``.to()``.
            if hasattr(m, "device"):
                try:
                    m.device = device
                except AttributeError:
                    pass

    def _opsd_pack_sigma(
        self, sample_pack: dict[str, Any], sigma_q: torch.Tensor,
    ) -> None:
        """Store ``sigma_q`` in the sample pack as a ``[B]`` CPU tensor."""
        sample_pack["opsd_sigma_q"] = sigma_q.detach().cpu()

    def _opsd_permute_r(self, perm: torch.Tensor) -> None:
        """Apply the fitting loop's row permutation ``perm`` to the replayed ``r``."""
        r = getattr(self, "_opsd_r_replay", None)
        if r is not None:
            self._opsd_r_replay = r.to(perm.device)[perm]

    @torch.no_grad()
    def _opsd_query_state(
        self, x_0: torch.Tensor, sigma_q: torch.Tensor,
    ) -> torch.Tensor:
        """Forward-noise ``x_0`` to the query state: ``z_q = (1 - s) * x_0 + s * eps``.

        ``s = sigma_q`` per sample (a scalar is broadcast too); ``eps`` comes
        from :meth:`_draw_eps`.
        """
        eps = self._draw_eps(x_0)
        s = self._bcast(sigma_q, x_0)
        return (1.0 - s) * x_0 + s * eps

    def _opsd_release_grad_memory(self) -> None:
        """Offload the reward-gradient modules to CPU.

        Called at the end of target construction, so they do not hold
        accelerator memory through the fitting phase. No-op without CUDA.
        """
        if not torch.cuda.is_available():
            return
        self._opsd_move_grad_modules(torch.device("cpu"))

    def _opsd_sigma_scale(
        self, sigma_q: Any, like: torch.Tensor,
    ) -> torch.Tensor:
        """``sigma_q`` broadcast over ``like``: the multiplier in the loss's ``z_q - sigma * v``."""
        return self._bcast(sigma_q, like)

    @staticmethod
    def _opsd_sigma_scale_build(sigma_q: Any, like: torch.Tensor) -> Any:
        """``sigma_q`` as a multiplier inside the target builder.

        A tensor is viewed as ``[B, 1, ...]`` against ``like``; a scalar is
        returned as a float.
        """
        if torch.is_tensor(sigma_q):
            return sigma_q.to(device=like.device, dtype=like.dtype).view(
                -1, *([1] * (like.ndim - 1))
            )
        return float(sigma_q)

    def _opsd_stash_r(self, start: int, end: int, r: torch.Tensor) -> None:
        """Record the ``r`` used to build this chunk's anchor."""
        buf = getattr(self, "_opsd_r_chunks", None)
        if buf is None:
            buf = {}
            self._opsd_r_chunks = buf
        buf[(int(start), int(end))] = r.detach().float().cpu()

    def _opsd_t_raw(
        self, sigma_q: Any, batch_size: int, device: torch.device,
    ) -> torch.Tensor:
        """Raw model timesteps ``sigma_q * T`` (``[B]``, fp32) from a ``[B]`` sigma."""
        T = float(self.scheduler.config.num_train_timesteps)
        return (
            sigma_q.to(device=device, dtype=torch.float32) * T
        ).to(torch.float32)

    def _opsd_tform(self, key: str, image_processor: Any):
        """Cached tensor-native replacement for an HF image processor."""
        cache = getattr(self, "_opsd_tform_cache", None)
        if cache is None:
            cache = {}
            self._opsd_tform_cache = cache
        if key not in cache:
            # get_image_transform reads resize/crop/normalize straight off the
            # processor config, so it matches what the frozen weights expect.
            from fewsteprl.rewards.clip_scorer import get_image_transform
            cache[key] = get_image_transform(image_processor)
        return cache[key]

    def _opsd_tr_step(
        self,
        x_anchor: torch.Tensor,
        prompts: list[str],
        direction: float,
        *,
        x_end: Optional[torch.Tensor] = None,
        first_grad: Optional[torch.Tensor] = None,   # noqa: ARG002 - see below
    ) -> torch.Tensor:
        """Trust-region target around ``x_anchor``, with the walk run at ``x_anchor + shift``.

        ``x_anchor`` is ``anchor_v`` and ``x_anchor + shift`` is ``anchor_u``
        (see :meth:`_opsd_build_anchor`), so everything position-sensitive in
        :meth:`_opsd_tr_step_base` happens around ``anchor_u``: the budget
        ``rho * ||x_start||``, the ball projection, and every ``grad R`` along
        the walk.

        The result is mapped back by ``- shift``, so the caller receives
        ``x_pm = anchor_v + Delta_pm``: the displacement is kept and the target
        sits around the loss's own anchor. Returning the absolute u-frame point
        instead would offset every target from the loss's probe centre by
        ``shift``.

        ``first_grad`` is ignored on the shifted path: it was computed at
        ``anchor_v``, not ``anchor_u``, so each direction recomputes its first
        reward gradient at ``anchor_u``.

        Without a measured shift (no preceding :meth:`_opsd_build_anchor`) this
        runs the unshifted walk around ``x_anchor``.
        """
        shift = getattr(self, "_opsd_frame_shift", None)
        if shift is None:
            return self._opsd_tr_step_base(
                x_anchor, prompts, direction, x_end=x_end, first_grad=first_grad,
            )
        if shift.shape != x_anchor.shape:
            raise RuntimeError(
                f"frame shift shape {tuple(shift.shape)} does not match the "
                f"anchor's {tuple(x_anchor.shape)}: the chunk loop and the anchor "
                "builder have gone out of step."
            )

        sh = shift.to(device=x_anchor.device, dtype=x_anchor.dtype)
        out = self._opsd_tr_step_base(
            x_anchor + sh,
            prompts,
            direction,
            # shifted too, so dir_mode="residual" keeps meaning the same vector
            x_end=None if x_end is None else x_end + sh,
            first_grad=None,
        )
        return (out - sh).detach()

    def _opsd_tr_step_base(
        self,
        x_anchor: torch.Tensor,
        prompts: list[str],
        direction: float,
        *,
        x_end: Optional[torch.Tensor] = None,
        first_grad: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """One trust-region target around ``x_anchor``: ``xp`` (direction=+1) or ``xm`` (-1).

        DiffusionOPSD's trust-region step: ``M_tgt = opsd_n_ascent`` steps along
        the unit-normalized reward gradient, each followed by projecting the
        displacement back into the ball of radius ``rho * ||x_anchor||``. The
        step length is ``h = eta * rho * ||x_anchor|| / M_tgt``, so ``M_tgt``
        steps travel at most ``eta * rho * ||x_anchor||``; with ``eta <= 1`` the
        projection is inactive in exact arithmetic and only acts as a guard.
        Either way ``||x_pm - x_anchor|| <= rho * ||x_anchor||``.

        ``opsd_dir_mode`` picks the step direction: ``"grad"`` (reward
        gradient), ``"rand"`` (one Gaussian direction per call), ``"residual"``
        (along ``x_end - x_anchor``, ``x_end`` being the rollout endpoint) or
        ``"noop"`` (no displacement).

        ``first_grad`` is the reward gradient at ``x_anchor``; both directions
        start there, so it can be shared between them to save one reward
        backward per chunk.

        Returns a detached tensor: the reward graph ends here, so the fitting
        phase back-propagates through the diffusion backbone only, never
        through the reward.
        """
        cfg = self._rl_cfg()
        rho = float(cfg.opsd_rho)
        n_ascent = int(cfg.opsd_n_ascent)
        eta = float(cfg.opsd_eta_tgt)
        eps_g = float(cfg.opsd_eps_grad)
        dir_mode = str(cfg.opsd_dir_mode)

        x_start = x_anchor.detach().float()
        if dir_mode == "noop" or n_ascent == 0:
            # No displacement: xp == xm == x_anchor, so the reward terms exert
            # no force (the reward-free control).
            return x_start

        budget = (rho * x_start.flatten(1).norm(dim=1)).view(-1, *([1] * (x_start.ndim - 1)))
        step_len = eta * budget / max(n_ascent, 1)

        fixed_dir = None
        if dir_mode == "rand":
            fixed_dir = torch.randn_like(x_start)
        elif dir_mode == "residual":
            if x_end is None:
                raise ValueError(
                    "opsd_dir_mode='residual' needs the rollout endpoint x_0."
                )
            fixed_dir = x_end.detach().float() - x_start

        x = x_start.clone()
        for i in range(n_ascent):
            if dir_mode == "grad":
                if i == 0 and first_grad is not None:
                    g = first_grad
                else:
                    x = x.detach().requires_grad_(True)
                    r = self._reward_of_latents_grad(x, prompts)
                    (g,) = torch.autograd.grad(r.sum(), x)
            else:
                g = fixed_dir
            gn = g.flatten(1).norm(dim=1).view(-1, *([1] * (x_start.ndim - 1))) + eps_g
            x = x.detach() + float(direction) * step_len * (g / gn)
            # Project the DISPLACEMENT (not the point) back into the ball.
            d = x - x_start
            dn = d.flatten(1).norm(dim=1).view(-1, *([1] * (x_start.ndim - 1)))
            x = (x_start + d * torch.clamp(budget / (dn + eps_g), max=1.0)).detach()
        return x.detach()

    def _opsd_unpack_sigma(
        self, sample_pack: dict[str, Any], device: torch.device,
    ) -> torch.Tensor:
        """Read ``sigma_q`` back from the sample pack as a ``[B]`` fp32 tensor on ``device``."""
        return sample_pack["opsd_sigma_q"].to(device=device, dtype=torch.float32)

    def _opsd_vae_decode_fp32(self, scaled: torch.Tensor) -> torch.Tensor:
        """Decode ``scaled`` through an fp32 copy of the frozen VAE.

        The copy (the whole VAE, ~4 bytes per parameter; only its decoder is
        used) is made once and cached, so the per-step cost is the decode
        itself. Uses the shared module directly when it is already fp32.
        """
        vae = self.vae
        if next(vae.parameters()).dtype == torch.float32:
            return vae.decode(scaled).sample
        cached = getattr(self, "_opsd_vae_fp32", None)
        if cached is None:
            import copy
            cached = copy.deepcopy(vae).to(dtype=torch.float32)
            cached.requires_grad_(False)
            cached.eval()
            self._opsd_vae_fp32 = cached
        # _opsd_release_grad_memory parks this on CPU between target builds;
        # _ensure_grad_scorer normally brings it back, but guard anyway so a
        # decode can never hit a device mismatch.
        if next(cached.parameters()).device != scaled.device:
            cached.to(scaled.device)
        return cached.decode(scaled).sample

    @torch.no_grad()
    def _resolve_query(
        self, sample_pack: dict[str, Any], x_0: torch.Tensor, cfg: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw the query noise level ``sigma_q`` and build the query state ``z_q``.

        Returns ``(sigma_q [B] fp32, z_q [B,C,H,W])``. ``opsd_sigma_mode``:

        * ``"fixed"`` -- ``sigma_q = opsd_query_sigma`` for every sample. ``z_q``
          is forward-noised (``opsd_state_mode="forward"``) or, for
          ``"rollout"``, taken from the recorded rollout step whose start sigma
          is nearest, with ``sigma_q`` snapped to that grid value.
        * ``"lognormal"`` -- ``sigmoid(opsd_logit_mean + opsd_logit_std * N(0, 1))``
          per sample, through the flow-map scheduler's shift.
        * ``"base_sampler"`` -- ``t`` from the training ``(t, r)`` sampler
          (``_sample_training_tr_three_mode``).

        The two random modes require ``opsd_state_mode="forward"``. The spread
        ``(std, min, max)`` of ``sigma_q`` is recorded for the
        ``opsd_sigma_q_*`` diagnostics.
        """
        if str(cfg.opsd_sigma_mode) == "fixed":
            sigma_star = float(cfg.opsd_query_sigma)
            mode = str(cfg.opsd_state_mode)

            if mode == "forward":
                sigma_f, z_q = sigma_star, self._opsd_query_state(x_0, sigma_star)
            else:
                # "rollout": take the state entering the step whose start-sigma
                # is nearest sigma*, straight off the recorded trajectory.
                x_t_all = sample_pack.get("x_t")
                start_ts = sample_pack.get("start_timesteps")
                if x_t_all is None or start_ts is None:
                    raise ValueError(
                        "opsd_state_mode='rollout' needs the per-step rollout tensors; "
                        "set fewstep_rl.reuse_x_t=true (the config validator normally "
                        "catches this)."
                    )
                T = float(self.scheduler.config.num_train_timesteps)
                # Row 0 is representative: every sample shares the inference grid.
                grid = start_ts[0].to(torch.float64) / T
                q = int(torch.argmin((grid - sigma_star).abs()).item())
                sigma_f = float(grid[q])
                z_q = x_t_all[:, q].to(device=x_0.device)
            n = x_0.shape[0]
            sigma_q = torch.full(
                (n,), float(sigma_f), device=x_0.device, dtype=torch.float32,
            )
            # Zero spread by construction; recorded anyway because the
            # opsd_sigma_q_* keys are emitted on every mode (see
            # _EXTRA_DIAGNOSTIC_KEYS).
            self._opsd_sigma_spread = (0.0, float(sigma_f), float(sigma_f))
            return sigma_q, z_q

        if str(cfg.opsd_state_mode) != "forward":
            # "rollout" snaps the state to an on-grid sigma, which is by
            # definition not the drawn one -- the two would disagree.
            raise ValueError(
                f"opsd_sigma_mode={str(cfg.opsd_sigma_mode)!r} requires "
                "opsd_state_mode='forward': a randomized sigma cannot be taken "
                "off the fixed rollout grid."
            )

        T = float(self.scheduler.config.num_train_timesteps)
        device = x_0.device
        dtype = next(self.generator.parameters()).dtype

        if str(cfg.opsd_sigma_mode) == "lognormal":
            # A logit-normal query sigma centred on ``opsd_query_sigma`` by the
            # default parameters. Its own key and parameters, separate from
            # ``fewstep_rl.renoise_distribution`` (the law of the training t).
            # Drawn once per phase from the global RNG: only t is drawn here
            # (r comes from _opsd_draw_r), and the fitting phase reads the
            # packed sigma_q rather than redrawing it.
            n = x_0.shape[0]
            logit_mean = float(cfg.opsd_logit_mean)
            logit_std = float(cfg.opsd_logit_std)

            # fp32 regardless of the network dtype: the shift is a division and
            # bf16's 8 mantissa bits would put ~1e-3 relative error on a
            # quantity the anchors and the lift all key off.
            t01 = torch.sigmoid(
                logit_mean
                + logit_std * torch.randn(n, device=device, dtype=torch.float32)
            )
            sigma_q = self.flowmap_scheduler.apply_shift(t01).to(torch.float32)
            sigma_q = sigma_q.clamp(1.0 / T, 1.0 - 1.0 / T)

            self._opsd_sigma_spread = (
                float(sigma_q.std()) if n > 1 else 0.0,
                float(sigma_q.min()),
                float(sigma_q.max()),
            )
            return sigma_q, self._opsd_query_state(x_0, sigma_q)

        # The training (t, r) law. ``j_idx``/``chunk_start`` key the
        # per-iteration RNG; the sampling phase is one draw per epoch, so
        # ``_rl_epoch`` is the axis that has to advance or every epoch would
        # reuse the same t. ``_r_raw`` is discarded: r is drawn by
        # :meth:`_opsd_draw_r`.
        t_raw, _r_raw = self._sample_training_tr_three_mode(
            batch_size=x_0.shape[0],
            chunk_start=0,
            j_idx=0,
            inner_epoch_idx=int(self._rl_epoch),
            device=device,
            dtype=dtype,
        )
        sigma_q = (t_raw.to(torch.float32) / T).clamp(1.0 / T, 1.0 - 1.0 / T)
        self._opsd_sigma_spread = (
            float(sigma_q.std()) if sigma_q.numel() > 1 else 0.0,
            float(sigma_q.min()),
            float(sigma_q.max()),
        )
        return sigma_q, self._opsd_query_state(x_0, sigma_q)

    def _reward_of_latents_grad(
        self, latents: torch.Tensor, prompts: list[str],
    ) -> torch.Tensor:
        """Differentiable reward of latents: ``R(decode(latents), prompts)``."""
        kind = str(self._rl_cfg().opsd_reward_kind)
        return self._reward_scores_grad(
            self._opsd_grad_scorer, kind, self._decode01_grad(latents), prompts,
        )

    def _reward_scores_grad(
        self, scorer: Any, kind: str, images01: torch.Tensor, prompts: list[str],
    ) -> torch.Tensor:
        """Differentiable reward for ``images01`` ``[B,3,H,W]`` in [0,1].

        As in DiffusionOPSD, each branch replicates the corresponding scorer's
        ``__call__`` body minus its ``@torch.no_grad``, and routes image
        preprocessing through tensor ops rather than the HF processor (which
        returns detached tensors). ``kind="mixed"`` returns the
        ``reward_fn``-weighted sum of its components.
        """
        if kind == "mixed":
            weights, scorers = scorer["weights"], scorer["scorers"]
            total = None
            for sub_kind, weight in weights.items():
                sub = self._reward_scores_grad(
                    scorers[sub_kind], sub_kind, images01, prompts,
                )
                term = float(weight) * sub
                total = term if total is None else total + term
            return total.float()

        device = scorer.device
        if kind == "hpsv2":
            # ``preprocess_val`` is the tensor-native transform the scorer
            # already builds, so this IS its own preprocessing, just outside
            # no_grad.
            image = scorer.preprocess_val(images01.to(scorer.dtype).to(device))
            text = scorer.processor(prompts).to(device)
            out = scorer.model(image, text)
            return torch.diagonal(
                out["image_features"] @ out["text_features"].T, 0,
            ).float()

        if kind == "clipscore":
            texts = scorer.processor(
                text=prompts, padding="max_length", truncation=True,
                return_tensors="pt",
            ).to(device)
            pixels = scorer._process(images01).to(device)
            out = scorer.model(pixel_values=pixels, **texts)
            return (out.logits_per_image.diagonal() / 100).float()

        if kind == "pickscore":
            # The HF image processor is replaced by the equivalent tensor
            # transform; the text processor is fine (token ids carry no grad).
            tform = self._opsd_tform("pickscore", scorer.processor.image_processor)
            pixels = tform(images01).to(dtype=scorer.dtype, device=device)
            text_inputs = scorer.processor(
                text=list(prompts), padding=True, truncation=True,
                max_length=77, return_tensors="pt",
            ).to(device)
            img_e = scorer.model.get_image_features(pixel_values=pixels)
            img_e = img_e / img_e.norm(p=2, dim=-1, keepdim=True)
            txt_e = scorer.model.get_text_features(**text_inputs)
            txt_e = txt_e / txt_e.norm(p=2, dim=-1, keepdim=True)
            # /26 matches PickScoreScorer's scaling, so the gradient is on the
            # same scale as the reward that set the advantages.
            return (
                (scorer.model.logit_scale.exp() * (txt_e @ img_e.T)).diag() / 26
            ).float()

        if kind == "aesthetic":
            tform = self._opsd_tform("aesthetic", scorer.processor.image_processor)
            pixels = tform(images01).to(dtype=scorer.dtype, device=device)
            embed = scorer.clip.get_image_features(pixel_values=pixels)
            embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
            # ``.mlp.layers``, NOT ``.mlp(...)``: MLP.forward carries its own
            # @torch.no_grad, so calling the module would detach even here.
            return scorer.mlp.layers(embed).squeeze(1).float()

        raise ValueError(f"OPSD differentiable ascent: unsupported reward {kind!r}.")

    def _rl_loss(
        self,
        x_0: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_embeds: torch.Tensor,
        advantages: torch.Tensor,
        t_raw: torch.Tensor,
        r_raw: Optional[torch.Tensor] = None,
        *,
        z_q: torch.Tensor,
        x_anchor_pre: torch.Tensor,   # noqa: ARG002 -- unused, the anchors are recomputed below
        xp_pre: torch.Tensor,
        xm_pre: torch.Tensor,
        sigma_q: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """One OPSD fitting step at the query state, v-form.

        Forwards the generator and the old model at ``(z_q, t, r)``, lifts both
        with the shared central difference, and fits ``x_0_pred_v`` with
        :meth:`decoupled_opsd_loss` against the packed targets ``xp_pre`` /
        ``xm_pre``, plus the KL term. Compared with ``MeanFlowNFTTrainer._rl_loss``:

        * There is **no forward noising**. MeanFlowNFT draws a fresh ``(t, r)``
          per inner iteration and rebuilds ``x_t`` from ``x_0``; OPSD fits at
          ONE query state per trajectory, already built in the sampling phase.
        * The target is the ``(xp, xm)`` pair rather than ``x_0``.

        ``x_anchor_pre`` is not used: both anchors are recomputed here from
        ``u_old`` at the same ``(t, r)``. ``opsd_frame_shift`` logs the mean
        ``|shift|`` of the last anchor build; it grows with ``t``, and 0 means no
        shift was measured.
        """
        cfg = self._rl_cfg()
        wdt = self._rl_work_dtype
        spatial = tuple(range(1, x_0.ndim))
        net_dtype = next(self.generator.parameters()).dtype

        beta = float(cfg.beta)
        adv_clip_max = float(cfg.adv_clip_max)
        if adv_clip_max <= 0:
            raise ValueError(f"adv_clip_max must be > 0, got {adv_clip_max}")

        _, use_v_kl = self._kl_usage()
        kl_weight = float(cfg.kl_weight)

        if self._shared_dF_dt is not None:
            raise RuntimeError(
                "stale _shared_dF_dt on entry to _rl_loss -- a previous step "
                "leaked its central difference (missing _post_compute_v_hook)."
            )

        x_t = z_q.to(net_dtype)
        t_for_model = t_raw.to(dtype=torch.float32)
        r_for_model = (
            r_raw.to(dtype=torch.float32) if r_raw is not None
            else torch.zeros_like(t_for_model)
        )
        # The sigma multiplier for ``z_q - sigma * v``, routed through
        # :meth:`_opsd_sigma_scale` so it has the same fp32 values the target
        # builder used (``_opsd_sigma_scale_build``): ``x_anchor_2`` then equals
        # the builder's ``anchor_v``, the centre the targets were built around.
        # ``t_for_model.to(net_dtype) / T`` would round it through bf16 and
        # shift the probe centre away from that point.
        t_broadcast = self._opsd_sigma_scale(sigma_q, x_0.to(wdt))

        # The CD is a collective, so the hook runs unconditionally and is
        # cleared in ``finally``.
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
            t_normed_b_f = t_broadcast.to(wdt)

            # The live clean-output prediction in the lifted coordinate
            # z_q - sigma*(u + S); the ONLY tensor on the graph.
            x_0_pred_v = self._opsd_clean_output(
                x_t_f, t_normed_b_f, u_theta_f,
                t_for_model=t_for_model,
                r_for_model=r_for_model,
                dF_dt=self._shared_dF_dt,
            )

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
            shared_dF_dt = self._shared_dF_dt if use_v_kl else None
        finally:
            self._post_compute_v_hook()

        # The fitting weight omega (``r_hat``): clip the advantage to
        # +-adv_clip_max and rescale to [0, 1] -- the same arithmetic as
        # MeanFlowNFTTrainer's ``r_hat``. adv_clip_max is both the clip range
        # and the loss scale.
        adv = advantages.float().clamp(-adv_clip_max, adv_clip_max)
        omega = ((adv / adv_clip_max) / 2.0 + 0.5).clamp(0.0, 1.0)

        unscaled, form_info = self.decoupled_opsd_loss(
            x_0_pred_v, x_anchor_1, x_anchor_2, xp_pre.to(wdt), xm_pre.to(wdt),
            omega, beta, spatial,
        )
        policy_loss = (unscaled * adv_clip_max).mean()
        loss = policy_loss

        with torch.no_grad():
            s_x0 = x_anchor_1 - x_anchor_2
            dev_abs = (x_0_pred_v.detach() - x_anchor_1).abs().mean()
            info = {
                "loss_policy": float(policy_loss.detach()),
                "loss_unscaled": float(unscaled.mean().detach()),
                # Mean of the per-sample query timesteps.
                "t_mean": float(t_raw.float().mean().item()),
                "r_hat_mean": float(omega.mean().item()),
                "advantage_mean": float(advantages.mean().item()),
                "advantage_std": float(advantages.std().item()),
                "x0_norm_mean": float((x_0.to(wdt) ** 2).mean().item()),
                "old_deviate": float(((u_theta_f - u_old_f) ** 2).mean().item()),
                "s_x0_abs_mean": float(s_x0.abs().mean()),
                "s_rel_median": float(
                    (s_x0.abs().mean() / dev_abs.clamp(min=1e-12))
                ),
                # Carried from the sampling phase so the trust-region
                # saturation is visible next to the loss it produced.
                "opsd_delta_res": float(
                    self._opsd_last_stats.get("opsd_delta_res", 0.0)
                ),
                "opsd_sigma_q": float(
                    sigma_q.mean() if torch.is_tensor(sigma_q) else sigma_q
                ),
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

        # The ``r`` diagnostics, emitted unconditionally on every rank (see
        # _EXTRA_DIAGNOSTIC_KEYS).
        with torch.no_grad():
            if r_raw is None:
                info["opsd_r_mean"] = 0.0
                info["opsd_r_frac_zero"] = 1.0
            else:
                r_f = r_raw.float()
                info["opsd_r_mean"] = float(r_f.mean())
                info["opsd_r_frac_zero"] = float((r_f == 0).float().mean())

        # Mean |frame shift| of the last anchor build; 0 before the first one.
        info["opsd_frame_shift"] = float(getattr(self, "_opsd_shift_stat", 0.0))

        # Spread of the query sigma from the last _resolve_query, emitted on
        # every mode (see _EXTRA_DIAGNOSTIC_KEYS); on "fixed" it is
        # (0, sigma*, sigma*).
        _std, _lo, _hi = getattr(self, "_opsd_sigma_spread", (0.0, 0.0, 0.0))
        info["opsd_sigma_q_std"] = _std
        info["opsd_sigma_q_min"] = _lo
        info["opsd_sigma_q_max"] = _hi
        return loss, info

    def _rl_sampling_phase(
        self, return_rewards: bool = True, return_intermediate: bool = False,
    ) -> tuple[dict[str, Any], Optional[tuple[torch.Tensor, list[str]]]]:
        """Base rollout and advantages, then the OPSD target stage.

        ``super()`` owns the rollout, the reward scoring, the cross-rank
        advantage computation and the stat tracker, so the advantages (and
        hence ``omega``) are the same as MeanFlowNFTTrainer's for the same
        rollout. This adds, per sample, the query state ``z_q`` and
        ``sigma_q``, the anchor, the two reward targets, and the ``r`` each
        anchor was built at (``opsd_r``), which :meth:`_opsd_draw_r` replays in
        the fitting phase.

        Skipped when ``return_rewards=False`` (rollouts only) or when OPSD is
        disabled -- both are config- or caller-level and therefore rank-uniform,
        so no rank can take a different branch and desync the collectives
        inside the target stage.
        """
        sample_pack, preview = super()._rl_sampling_phase(
            return_rewards=return_rewards,
            return_intermediate=return_intermediate,
        )

        cfg = self._rl_cfg()
        # Rank-uniform condition, so returning early cannot desync.
        if not bool(cfg.opsd_enabled) or not return_rewards:
            return sample_pack, preview

        device = next(self.generator.parameters()).device
        x_0 = sample_pack["x_0"].to(device)
        prompt_embeds = sample_pack["prompt_embeds"].to(device)
        pooled_embeds = sample_pack["pooled_embeds"].to(device)
        prompts = sample_pack["prompts"]

        sigma_q, z_q = self._resolve_query(sample_pack, x_0, cfg)

        anchor_out, xp_out, xm_out, stats = self._build_opsd_targets(
            x_0=x_0,
            prompt_embeds=prompt_embeds,
            pooled_embeds=pooled_embeds,
            prompts=prompts,
            z_q=z_q,
            sigma_q=sigma_q,
        )

        # Back to CPU to match the pack's convention (the base stores CPU
        # tensors and _rl_train_phase moves them to the device).
        sample_pack["opsd_z_q"] = z_q.detach().cpu()
        sample_pack["opsd_y_0"] = anchor_out.detach().cpu()
        sample_pack["opsd_y_plus"] = xp_out.detach().cpu()
        sample_pack["opsd_y_minus"] = xm_out.detach().cpu()
        self._opsd_pack_sigma(sample_pack, sigma_q)
        self._opsd_last_stats = stats

        # The r each anchor was built at, replayed by _opsd_draw_r in the
        # fitting phase.
        if sample_pack.get("opsd_y_0") is not None:
            r_all = self._opsd_collect_r(sample_pack["opsd_y_0"].shape[0])
            if r_all is not None:
                sample_pack["opsd_r"] = r_all
        return sample_pack, preview

    def _rl_train_phase(self, sample_pack: dict[str, Any]) -> dict[str, float]:
        """The OPSD fitting phase, with ``r`` replayed into it.

        ``num_inner_epochs`` passes over the packed samples, each in a fresh row
        permutation and in inner batches of ``rl_inner_batch_size``: one
        :meth:`_rl_loss` and backward per inner batch, an optimizer step every
        ``gradient_accumulation_steps`` backwards. The packed ``r`` is published
        as ``_opsd_r_replay`` for :meth:`_opsd_draw_r` and cleared in
        ``finally``.
        """
        self._opsd_r_replay = sample_pack.get("opsd_r")
        try:
            cfg = self._rl_cfg()
            device = next(self.generator.parameters()).device

            if not bool(cfg.opsd_enabled):
                raise ValueError(
                    f"{type(self).__name__} requires fewstep_rl.opsd_enabled=true. "
                    "With it off no reward-gradient targets are built and this trainer "
                    "has nothing to fit; run --trainer meanflow_nft instead."
                )
            missing = [k for k in self._OPSD_KEYS if sample_pack.get(k) is None]
            if missing:
                # Config-level and therefore rank-uniform: every rank agrees, so
                # raising here cannot desync a collective.
                raise ValueError(
                    f"sample_pack is missing {missing}; the OPSD target stage did "
                    "not run. It is skipped when return_rewards=False."
                )

            x_0 = sample_pack["x_0"].to(device)
            prompt_embeds = sample_pack["prompt_embeds"].to(device)
            pooled_embeds = sample_pack["pooled_embeds"].to(device)
            advantages = sample_pack["advantages"].to(device)
            z_q_all = sample_pack["opsd_z_q"].to(device)
            anchor_all = sample_pack["opsd_y_0"].to(device)
            xp_all = sample_pack["opsd_y_plus"].to(device)
            xm_all = sample_pack["opsd_y_minus"].to(device)
            sigma_q = self._opsd_unpack_sigma(sample_pack, device)

            n_local = x_0.shape[0]
            if n_local == 0:
                return {}
            for name, tensor in (
                ("opsd_z_q", z_q_all), ("opsd_y_0", anchor_all),
                ("opsd_y_plus", xp_all), ("opsd_y_minus", xm_all),
            ):
                if tensor.shape[0] != n_local:
                    raise ValueError(
                        f"sample_pack['{name}'] has leading dim {tensor.shape[0]}, "
                        f"expected {n_local}. It is indexed by the same row "
                        "permutation as x_0, so a mismatch would pair a query "
                        "state with another sample's target."
                    )

            inner_bs = int(cfg.rl_inner_batch_size) if cfg.rl_inner_batch_size > 0 else n_local
            inner_bs = max(1, min(inner_bs, n_local))
            n_inner_epochs = max(1, int(cfg.num_inner_epochs))
            max_grad_norm = float(self.config.solver.generator.max_grad_norm)
            grad_accum = max(1, int(self.config.train.gradient_accumulation_steps))
            # One fitting query (one backward) per sample per inner epoch, so the
            # base's ``* num_train_timesteps`` factor is 1 here.
            effective_grad_accum_steps = grad_accum

            # ``sigma_q`` is per-sample ([B]), so it must be permuted with the
            # other per-sample tensors below, or a query state would be paired
            # with another sample's noise level. A non-[B] ``sigma_q`` is used
            # unpermuted.
            sigma_is_per_sample = torch.is_tensor(sigma_q) and sigma_q.ndim == 1
            if sigma_is_per_sample and sigma_q.shape[0] != n_local:
                raise ValueError(
                    f"opsd_sigma_q has leading dim {sigma_q.shape[0]}, expected "
                    f"{n_local}: it is indexed by the same row permutation as x_0."
                )

            self.generator.train()
            accumulator: dict[str, float] = {}
            n_backward = 0
            n_opt_steps = 0
            current_accumulated_steps = 0
            self.optimizers["generator"].zero_grad(set_to_none=True)
            last_grad_norm: float = 0.0

            for inner_epoch_idx in range(n_inner_epochs):
                perm = torch.randperm(n_local, device=device)
                sx0 = x_0[perm]
                spe = prompt_embeds[perm]
                spp = pooled_embeds[perm]
                sadv = advantages[perm]
                szq = z_q_all[perm]
                s_anchor = anchor_all[perm]
                sxp = xp_all[perm]
                sxm = xm_all[perm]
                ssig = sigma_q[perm] if sigma_is_per_sample else sigma_q
                # The replayed ``r`` must follow the same row permutation, or a
                # query state would be paired with another sample's r.
                self._opsd_permute_r(perm)

                for start in range(0, n_local, inner_bs):
                    end = min(start + inner_bs, n_local)
                    bsz = end - start
                    sig_b = ssig[start:end] if sigma_is_per_sample else ssig
                    t_b = self._opsd_t_raw(sig_b, bsz, device)
                    # r is REPLAYED here: _opsd_draw_r returns the r each anchor
                    # was built at, so the live prediction is queried at the
                    # identical (t, r). A random r is legitimate only under the
                    # MeanFlow lift, which makes x_0_pred_v r-invariant.
                    # opsd_r_mode=zero gives r = 0.
                    r_b = self._opsd_draw_r(
                        t_b,
                        chunk_start=start,
                        chunk_end=end,
                        inner_epoch_idx=inner_epoch_idx,
                        device=device,
                    )

                    loss, info = self._rl_loss(
                        x_0=sx0[start:end],
                        prompt_embeds=spe[start:end],
                        pooled_embeds=spp[start:end],
                        advantages=sadv[start:end],
                        t_raw=t_b,
                        r_raw=r_b,
                        z_q=szq[start:end],
                        x_anchor_pre=s_anchor[start:end],
                        xp_pre=sxp[start:end],
                        xm_pre=sxm[start:end],
                        sigma_q=sig_b,
                    )
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

                        if (
                            self.generator_ema is not None
                            and self.global_step % self.config.train.ema_every == 0
                        ):
                            self.ema_update(
                                self.generator, self.generator_ema,
                                self._current_ema_decay(),
                            )

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
            # One fitting query per sample, so the key's meaning ("backwards per
            # sample per inner epoch") is preserved at 1.
            averaged["num_train_timesteps"] = 1.0
            averaged["effective_grad_accum_steps"] = float(effective_grad_accum_steps)
            return averaged
        finally:
            self._opsd_r_replay = None

    def _s_term(
        self,
        t_cd: torch.Tensor,
        r_cd: torch.Tensor,
        dF_dt: Optional[torch.Tensor],
        like: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """``S = (t - r) * dF/dt`` in ``like``'s dtype, or None when S is exactly 0."""
        if dF_dt is None:
            # ``disable_cd`` skips the central difference on purpose, so S is
            # DEFINED to be zero even where r != t (e.g. opsd_r_mode=given_t).
            # Checked before the r == t guard, which is exactly what it opts
            # out of; config-level, hence rank-uniform. Not a free speed-up:
            # S cancels in x_0_pred_v - x_anchor_1, but the stop-grad
            # coefficients (trust-region radius, W, the reward force) are built
            # from anchor positions, which all shift by sigma*S, so the
            # objective changes slightly. See FewstepRLTrainConfig.disable_cd.
            if bool(getattr(self._rl_cfg(), "disable_cd", False)):
                return None
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
        dt_cd = (t_cd - r_cd).view(
            like.shape[0], *([1] * (like.ndim - 1))
        ).to(wdt)
        return dt_cd * dF_dt.to(wdt)

    def decoupled_opsd_loss(
        self,
        x_0_pred_v: torch.Tensor,  # fp32, the ONLY tensor carrying grad
        x_anchor_1: torch.Tensor,  # fp32 detached: anchor / self-regression target
        x_anchor_2: torch.Tensor,  # fp32 detached: probe center
        xp: torch.Tensor,          # fp32 detached: reward-ascent target
        xm: torch.Tensor,          # fp32 detached: reward-descent target
        omega: torch.Tensor,       # [B] in [0,1], the fitting weight r_hat
        beta: float,
        spatial: tuple[int, ...],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """The three-term decoupled OPSD loss.

        Built from the base's two loss primitives::

            (iii) proximal   self_regression(x_0_pred_v, x_anchor_1, W)
            (i)   attract    reward_augmentation(x_0_pred_v, xp, x_anchor_2, omega_p, W)
            (ii)  repel      reward_augmentation(x_0_pred_v, x_anchor_2, xm, omega_m, W)

        ``reward_augmentation`` injects ``grad = -(2W/N) * om * (x_pos - x_neg)``
        for arguments ``(x_pos, x_neg)``. So:

        * term (i) passes ``(xp, x_anchor_2)`` -> gradient ``-(xp - x_anchor_2)``,
          i.e. the update moves ``x_0_pred_v`` TOWARD ``xp``;
        * term (ii) passes ``(x_anchor_2, xm)`` -- **swapped** -> gradient
          ``-(x_anchor_2 - xm) = +(xm - x_anchor_2)``, i.e. the update moves
          ``x_0_pred_v`` AWAY from ``xm``.

        The argument order in term (ii) is the critical detail. Passing
        ``(xm, x_anchor_2)`` instead leaves every logged loss value
        bit-identical (the kernel's value is ``W * mean[norm_grad**2]``, which
        is even in the sign) while training the policy to CHASE the
        reward-descent target.

        As with :meth:`GeneralRLTrainer.decoupled_rl_loss`, the gradient matches
        the fused (single-MSE) OPSD loss exactly but the loss VALUE does not (it
        differs by cross terms of indefinite sign): compare gradients, never
        loss values.
        """
        with torch.no_grad():
            W, omega_p, omega_m = self.opsd_weights(
                x_0_pred_v, x_anchor_1, x_anchor_2, xp, xm, omega, beta, spatial,
            )

        # (iii) proximal anchor term -- the implicit trust region.
        l_reg = self.self_regression(x_0_pred_v, x_anchor_1, W, spatial)
        # (i) reward attraction toward xp.
        l_pos = self.reward_augmentation(x_0_pred_v, xp, x_anchor_2, omega_p, W, spatial)
        # (ii) reward repulsion away from xm. Arguments SWAPPED; see above.
        l_neg = self.reward_augmentation(x_0_pred_v, x_anchor_2, xm, omega_m, W, spatial)

        return l_reg + l_pos + l_neg, {
            # The standard affine_* loss keys of GeneralRLTrainer's logging.
            "affine_coeff_mean": float(W.mean().item()),
            "affine_adv_coeff_mean": float((omega_p - omega_m).mean().item()),
            "self_regression_loss": float(l_reg.detach().mean()),
            # The base's key, here the SUM of both reward terms, so it stays
            # directly comparable with self_regression_loss.
            "reward_augmentation_loss": float((l_pos + l_neg).detach().mean()),
            "opsd_pull_pos": float(l_pos.detach().mean()),
            "opsd_push_neg": float(l_neg.detach().mean()),
            "opsd_omega_pos_mean": float(omega_p.mean().item()),
            "opsd_omega_neg_mean": float(omega_m.mean().item()),
        }

    @torch.no_grad()
    def opsd_weights(
        self,
        x_0_pred_v: torch.Tensor,
        x_anchor_1: torch.Tensor,
        x_anchor_2: torch.Tensor,
        xp: torch.Tensor,
        xm: torch.Tensor,
        omega: torch.Tensor,
        beta: float,
        spatial: tuple[int, ...],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(W, omega_p, omega_m)`` -- the three frozen per-sample weights.

        ``omega`` is the fitting weight ``r_hat``, already in [0, 1]. With
        ``probe_pm = x_anchor_2 pm beta*(x_0_pred_v - x_anchor_1)`` the adaptive
        normalizers are::

            gamma_+ = max(mean|probe_pos - xp|, eps_gamma)
            gamma_- = max(mean|probe_neg - xm|, eps_gamma)
            a_+ = omega / gamma_+          a_- = (1 - omega) / gamma_-

        ``W = beta**2 * (a_+ + a_-)``, not ``beta * (a_+ + a_-)``. DiffusionOPSD's
        loss differentiates the probe branches, so the chain rule contributes
        one factor of beta from the branch and the outer MSE another via the
        coefficient, giving
        ``grad = (2*beta/N) [beta*(a_+ + a_-)*delta - (a_+ e_+ - a_- e_-)]`` with
        ``delta = x_0_pred_v - x_anchor_1`` and ``e_pm = x_pm - x_anchor_2``. The
        affine form of :meth:`GeneralRLTrainer.decoupled_rl_loss` carries only
        ONE beta, so without the square the two would differ by a factor of
        beta whenever ``beta != 1``.

        ``omega_p`` / ``omega_m`` are defined so that ``W * omega_pm ==
        a_pm * beta``: the reward terms' coefficient is then exactly the
        bracket's, and both sub-losses can share the single weight ``W`` the
        way :meth:`GeneralRLTrainer.decoupled_rl_loss` requires.

        fp64 throughout: these are per-sample reciprocals of a mean absolute
        residual, which can be small.
        """
        eps_gamma = float(self._rl_cfg().opsd_eps_gamma)
        bc = (-1, *([1] * (x_0_pred_v.ndim - 1)))

        # Probe DIRECTION from the S-free deviation, CENTER at x_anchor_2 --
        # the same split as GeneralRLTrainer.decoupled_weights.
        dev = x_0_pred_v.detach() - x_anchor_1
        probe_pos = x_anchor_2 + beta * dev
        probe_neg = x_anchor_2 - beta * dev

        gamma_p = (
            (probe_pos.double() - xp.double())
            .abs().mean(dim=spatial, keepdim=True).clip(min=eps_gamma)
        )
        gamma_m = (
            (probe_neg.double() - xm.double())
            .abs().mean(dim=spatial, keepdim=True).clip(min=eps_gamma)
        )

        om_b = omega.view(bc).double()
        a_p = om_b / gamma_p
        a_m = (1.0 - om_b) / gamma_m
        if not bool(self._rl_cfg().opsd_dual_neg):
            # Positive-branch-only ablation (DiffusionOPSD's opa_dual_neg=0).
            # Zeroing a_- removes BOTH negative forces -- the repulsion from
            # xm and the proximal pull it carries -- which is exactly what
            # dropping the branch means.
            a_m = torch.zeros_like(a_m)
        ab = a_p + a_m

        W = (beta * beta * ab).reshape(-1)          # beta**2, see the docstring
        omega_p = a_p / (beta * ab)                 # W * omega_p == a_p * beta
        omega_m = a_m / (beta * ab)                 # W * omega_m == a_m * beta
        return W, omega_p, omega_m
