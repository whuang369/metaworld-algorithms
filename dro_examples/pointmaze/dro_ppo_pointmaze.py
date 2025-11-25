from dataclasses import dataclass
from pathlib import Path

import tyro

from metaworld_algorithms.config.networks import (
    ContinuousActionPolicyConfig,
    # ValueFunctionConfig,
)
from metaworld_algorithms.config.nn import VanillaNetworkConfig
from metaworld_algorithms.config.optim import OptimizerConfig
from metaworld_algorithms.config.rl import OnPolicyTrainingConfig
from metaworld_algorithms.envs import MetaworldConfig
from metaworld_algorithms.rl.algorithms import PPOConfig
from metaworld_algorithms.run import Run

from custom_envs.pointmaze import PointMazeConfig


@dataclass(frozen=True)
class Args:
    seed: int = 1
    track: bool = False
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_group: str | None = None
    data_dir: Path = Path("./run_results")
    resume: bool = False

    dro_learning_rate: float = 0.3
    dro_eps: float = 0.01
    dro_min_prob: float = 0.01
    rollout_steps: int = 10_000


def main() -> None:
    args = tyro.cli(Args)

    run = Run(
        run_name="pointmaze_ppo",
        seed=args.seed,
        data_dir=args.data_dir,
        env=PointMazeConfig(
            env_id="Pointmaze/dro_pointmaze",
            terminate_on_success=False,
            num_tasks=3,
        ),
        algorithm=PPOConfig(
            num_tasks=3,
            gamma=0.99,
            policy_config=ContinuousActionPolicyConfig(
                network_config=VanillaNetworkConfig(
                    optimizer=OptimizerConfig(max_grad_norm=1.0),
                ),
                squash_tanh=False,
            ),
            vf_config=None,
            baseline_type="linear",
            num_epochs=16,
            num_gradient_steps=32,
            gae_lambda=0.97,
            target_kl=None,
            clip_vf_loss=False,
            normalize_advantages=False,
            dro_upd_num_steps=args.rollout_steps,
        ),
        training_config=OnPolicyTrainingConfig(
            total_steps=int(2e7),
            rollout_steps=10_000,
            evaluation_frequency=1_000_000 // 500,
        ),
        checkpoint=True,
        resume=args.resume,
        dro=True,
        dro_learning_rate=args.dro_learning_rate,
        dro_eps=args.dro_eps,
        dro_min_prob=args.dro_min_prob,
    )

    if args.track:
        assert args.wandb_project is not None and args.wandb_entity is not None
        run.enable_wandb(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            config=run,
            resume="allow",
        )

    run.start()


if __name__ == "__main__":
    main()
