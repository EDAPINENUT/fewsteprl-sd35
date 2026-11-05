"""FewStepRL training entry point (Stage-3 RL on a few-step SD3.5-M flow map).

Every trainer here reads the ``fewstep_rl`` config section and is launched by
one of the ``scripts/train/<arm>.sh`` wrappers, which pin the trainer key:

- ``meanflow_nft`` / ``nft_rl``: MeanFlowNFT and its S-term ablation.
- ``dmdnft_rl`` / ``dmdawm_rl``: NFT / AWM variants built on an online-trained
  auxiliary v-predictor (``fewstep_rl.aux_v_enabled``).
- ``meanflowgrpo_rl`` / ``flowgrpo_rl``: MeanFlowGRPO and its S-term ablation.
- ``meanflowawm_rl`` / ``awm_rl``: MeanFlowAWM and its S-term ablation.
- ``meanflowopsd_rl``: DiffusionOPSD under MeanFlow.
- ``meanflowrefl_rl``: the ReFL corner of few-step OPSD.
"""

# pkg_resources must be imported before the heavy trainer stack (torch /
# transformers / diffusers): imported after it, the ``import pkg_resources``
# inside ImageReward can fail with a spurious ModuleNotFoundError at
# reward-loading time. Importing it first caches it in ``sys.modules``.
try:
    import pkg_resources  # type: ignore  # noqa: F401
except Exception:
    pass

import argparse
import yaml

from fewsteprl.config import FewStepRLConfig
from fewsteprl.trainers.dmdawm import DMDAWMTrainer
from fewsteprl.trainers.dmdnft import DMDNFTTrainer
from fewsteprl.trainers.meanflowawm import AWMTrainer, MeanFlowAWMTrainer
from fewsteprl.trainers.meanflowgrpo import FlowGRPOTrainer, MeanFlowGRPOTrainer
from fewsteprl.trainers.meanflownft import MeanFlowNFTTrainer, NFTTrainer
from fewsteprl.trainers.meanflowopsd import MeanFlowOPSDTrainer
from fewsteprl.trainers.meanflowrefl import MeanFlowReFLTrainer


TRAINER_REGISTRY = {
    # x_anchor_2's instantaneous velocity is old_model's u + (t - r) * dF/dt
    # (central difference); the NFT loss is evaluated in decoupled form,
    # self_regression + reward_augmentation.
    "meanflow_nft": MeanFlowNFTTrainer,
    # x_anchor_2's instantaneous velocity comes from the online-trained
    # aux_v_model instead of old_model's central difference.
    # Requires fewstep_rl.aux_v_enabled.
    "dmdnft_rl": DMDNFTTrainer,
    # AWM with no central difference (x_anchor_2 = x_t - t * u_old);
    # aux_v_model supplies the reward term's probe center.
    # Requires fewstep_rl.aux_v_enabled.
    "dmdawm_rl": DMDAWMTrainer,
    "meanflowgrpo_rl": MeanFlowGRPOTrainer,
    "meanflowawm_rl": MeanFlowAWMTrainer,
    # Both require fewstep_rl.opsd_enabled.
    "meanflowopsd_rl": MeanFlowOPSDTrainer,
    "meanflowrefl_rl": MeanFlowReFLTrainer,
    # S-term ablations of meanflow_nft / meanflowawm_rl / meanflowgrpo_rl: the
    # flow map's average velocity is used as the instantaneous one, and the
    # central difference is never computed. Same configs and loop otherwise.
    "nft_rl": NFTTrainer,
    "awm_rl": AWMTrainer,
    "flowgrpo_rl": FlowGRPOTrainer,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FewStepRL Training")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    parser.add_argument(
        "--trainer",
        type=str,
        required=True,
        choices=list(TRAINER_REGISTRY.keys()),
        help="Trainer key (pinned by scripts/train/<arm>.sh)",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Config overrides in key=value format, e.g. train.seed=123",
    )
    return parser.parse_args()


def parse_overrides(override_list: list[str]) -> dict:
    """Parse CLI overrides such as ``train.seed=123`` into a dictionary."""
    overrides = {}
    for item in override_list:
        if "=" not in item:
            raise ValueError(f"Invalid override format: {item}. Expected key=value.")
        key, raw_value = item.split("=", 1)
        value = "" if raw_value == "" else yaml.safe_load(raw_value)
        overrides[key] = value
    return overrides


def main():
    args = parse_args()

    overrides = parse_overrides(args.override) if args.override else None
    config = FewStepRLConfig.from_yaml(args.config, overrides=overrides)

    trainer_cls = TRAINER_REGISTRY[args.trainer]
    trainer = trainer_cls(config)
    trainer.train()


if __name__ == "__main__":
    main()
