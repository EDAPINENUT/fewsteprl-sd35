"""
ImageReward v1.0 scorer.

Adapted from DiffusionNFT's ImageReward scorer.
"""

import importlib
import logging
import os
import sys
import torch
from fewsteprl.rewards.reward_ckpt_path import CKPT_PATH

logger = logging.getLogger(__name__)


def _ensure_pkg_resources(retries: int = 8, base_delay: float = 0.5) -> bool:
    """Make ``import pkg_resources`` succeed before ImageReward/clip need it.

    ImageReward imports ``clip``, which does ``from pkg_resources import
    packaging``. Under a multi-process torchrun launch on a shared filesystem
    (e.g. CephFS), the site-packages directory listing can be stale for some
    ranks, so Python's PathFinder misses the installed ``pkg_resources`` and
    raises ``ModuleNotFoundError`` although the package is present.

    Drops any ``None`` entry from ``sys.modules``, then retries the import,
    calling :func:`importlib.invalidate_caches` between attempts to force a
    fresh directory re-scan. Returns True once importable.
    """
    # A ``None`` entry in sys.modules makes every import of the name fail.
    if sys.modules.get("pkg_resources", False) is None:
        del sys.modules["pkg_resources"]
    if "pkg_resources" in sys.modules:
        return True

    import time

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            import pkg_resources  # type: ignore  # noqa: F401 - cache the real module
            if attempt > 0:
                logger.info(
                    "[ImageReward] pkg_resources became importable after %d "
                    "retries (stale directory listing).", attempt,
                )
            return True
        except Exception as e:  # noqa: BLE001 - retry stale-finder failures
            last_err = e
            importlib.invalidate_caches()
            time.sleep(base_delay * (attempt + 1))

    logger.warning(
        "[ImageReward] pkg_resources still not importable after %d retries: "
        "%r", retries, last_err,
    )
    return False


# Alias of _ensure_pkg_resources() that discards the result.
def _repair_pkg_resources() -> None:
    _ensure_pkg_resources()


def _load_imagereward_module():
    """Load ImageReward with compatibility patch for new transformers versions."""
    _ensure_pkg_resources()
    try:
        from transformers import modeling_utils as hf_modeling_utils
    except Exception:
        hf_modeling_utils = None

    if hf_modeling_utils is not None:
        try:
            from transformers import pytorch_utils as hf_pytorch_utils
        except Exception:
            hf_pytorch_utils = None
        if hf_pytorch_utils is not None:
            for symbol_name in (
                "apply_chunking_to_forward",
                "find_pruneable_heads_and_indices",
                "prune_linear_layer",
            ):
                if not hasattr(hf_modeling_utils, symbol_name) and hasattr(hf_pytorch_utils, symbol_name):
                    # ImageReward imports these symbols from the old module path.
                    setattr(hf_modeling_utils, symbol_name, getattr(hf_pytorch_utils, symbol_name))

    try:
        return importlib.import_module("ImageReward")
    except Exception:
        # Log the full traceback: MultiScorer reports only "<type>: <message>"
        # for an unavailable reward, which hides which nested import failed.
        # Re-raise so MultiScorer's unavailable-reward handling still applies.
        import traceback
        logger.error(
            "[ImageReward] import failed; pkg_resources in sys.modules=%r\n%s",
            sys.modules.get("pkg_resources", "ABSENT"),
            traceback.format_exc(),
        )
        raise


class ImageRewardScorer(torch.nn.Module):
    def __init__(self, device="cuda", dtype=torch.float32):
        super().__init__()
        self.device = torch.device(device)
        self.dtype = dtype
        rm_module = _load_imagereward_module()
        self.model = (
            rm_module.load(
                "ImageReward-v1.0",
                device=str(self.device),
                # Keep ImageReward cache/checkpoints under unified reward_ckpt_path.
                download_root=os.path.join(os.path.expanduser(CKPT_PATH), "ImageReward"),
            )
            .eval()
            .to(dtype=dtype)
        )
        self.model.requires_grad_(False)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        device = kwargs.get("device")
        if device is None and args:
            first = args[0]
            if isinstance(first, (str, int, torch.device)):
                device = first
        if device is not None:
            self.device = torch.device(device)
            if hasattr(self.model, "to"):
                self.model.to(self.device)
            if hasattr(self.model, "device"):
                # ImageReward.inference_rank relies on this attribute to place inputs.
                self.model.device = str(self.device)
        return self

    @torch.no_grad()
    def __call__(self, prompts, images):
        if hasattr(self.model, "device"):
            # Keep inference input placement aligned with model weights.
            self.model.device = str(self.device)
        _, rewards = self.model.inference_rank(prompts, images)
        raw = torch.as_tensor(rewards, device=self.device)
        n = len(prompts)
        try:
            rewards = torch.diagonal(raw.reshape(n, n), 0).contiguous()
        except Exception as e:  # noqa: BLE001 - log the shapes, then re-raise
            print(
                f"[ImageReward] reshape failed: n_prompts={n} "
                f"n_images={len(images)} raw_numel={raw.numel()} "
                f"raw_shape={tuple(raw.shape)} err={type(e).__name__}: {e}",
                flush=True,
            )
            raise
        return rewards
