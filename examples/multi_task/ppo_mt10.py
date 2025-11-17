from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import tyro

from metaworld_algorithms.config.networks import (
    ContinuousActionPolicyConfig, ValueFunctionConfig,
    # ValueFunctionConfig,
)
from metaworld_algorithms.config.nn import VanillaNetworkConfig
from metaworld_algorithms.config.optim import OptimizerConfig
from metaworld_algorithms.config.rl import OnPolicyTrainingConfig
from metaworld_algorithms.envs import MetaworldConfig
from metaworld_algorithms.rl.algorithms import PPOConfig
from metaworld_algorithms.run import Run


def get_value_function_config(
    baseline_type: str = 'linear'
) -> ValueFunctionConfig:

    if baseline_type == 'linear':
        value_config = None
    elif baseline_type == 'mlp':
        value_config = ValueFunctionConfig(
            network_config=VanillaNetworkConfig(
                optimizer=OptimizerConfig(max_grad_norm=1.0),
            )
        )
    else:
        raise ValueError(f'Invalid baseline type: {baseline_type}')

    return value_config


@dataclass(frozen=True)
class Args:
    seed: int = 1
    resume: bool = False
    data_dir: Path = Path("./run_results")

    track: bool = True
    wandb_entity: str = 'nicholascorrado'
    wandb_project: str = 'metaworld'
    wandb_group: str = ""
    wandb_name: str = "lfb"

    env_id: str = 'MT10'
    total_steps: int = int(2e7)
    rollout_steps: int = 10_000
    evaluation_frequency: int = 1_000_000 // 500 # number of trajectories between each eval
    baseline_type: Literal["linear", "mlp"] = "linear"

def main() -> None:
    args = tyro.cli(Args)

    run = Run(
        run_name="mt10_ppo_lfb",
        seed=args.seed,
        data_dir=args.data_dir,
        env=MetaworldConfig(
            env_id="MT10",
            terminate_on_success=False,
        ),
        algorithm=PPOConfig(
            num_tasks=10,
            gamma=0.99,
            policy_config=ContinuousActionPolicyConfig(
                network_config=VanillaNetworkConfig(
                    optimizer=OptimizerConfig(max_grad_norm=1.0),
                ),
                squash_tanh=False,
            ),
            vf_config=get_value_function_config(args.baseline_type),
            baseline_type=args.baseline_type,
            num_epochs=16,
            num_gradient_steps=32,
            gae_lambda=0.97,
            target_kl=None,
            clip_vf_loss=False,
            normalize_advantages=False,
        ),
        training_config=OnPolicyTrainingConfig(
            total_steps=args.total_steps,
            rollout_steps=args.rollout_steps,
            evaluation_frequency=args.evaluation_frequency,
        ),
        checkpoint=True,
        resume=args.resume,
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
