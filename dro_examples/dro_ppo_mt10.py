from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import tyro

from metaworld_algorithms.config.networks import (
    ContinuousActionPolicyConfig, ValueFunctionConfig,
    # ValueFunctionConfig,
)
from metaworld_algorithms.config.nn import VanillaNetworkConfig, MOOREConfig
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

def get_actor_config(
    actor_type: str = 'vanilla'
):

    if actor_type == 'vanilla':
        actor_config = ContinuousActionPolicyConfig(
                network_config=MOOREConfig(
                    num_tasks=10, optimizer=OptimizerConfig(lr=3e-4, max_grad_norm=1.0)
                ),
                squash_tanh=False,
            )
    elif actor_type == 'moore':
        actor_config = ContinuousActionPolicyConfig(
            network_config=MOOREConfig(
                num_tasks=10, optimizer=OptimizerConfig(lr=3e-4, max_grad_norm=1.0)
            ),
            log_std_min=-10,
            log_std_max=2,
        )
    else:
        raise ValueError(f'Invalid actor type: {actor_type}')

    return actor_config


@dataclass(frozen=False)
class Args:
    seed: int = 1
    seed_offset: int = 0
    resume: bool = False
    data_dir: Path = Path("./run_results")

    track: bool = True
    wandb_entity: str = 'nicholascorrado'
    wandb_project: str = 'metaworld'
    wandb_group: str = ""
    wandb_name: str = "dro"

    env_id: str = 'MT10'
    total_steps: int = int(4e7)
    rollout_steps: int = 10_000
    evaluation_frequency: int = 1_000_000 // 500 # number of trajectories between each eval
    actor_type: Literal["vanilla", "moore", "paco", "soft"] = "vanilla"
    baseline_type: Literal["linear", "mlp"] = "linear"

    dro_learning_rate: float = 0.3
    dro_eps: float = 0.01
    dro_min_prob: float = 0.01

def main() -> None:
    args = tyro.cli(Args)
    args.seed += args.seed_offset

    run = Run(
        run_name="mt10_ppo_lfb",
        seed=args.seed,
        data_dir=args.data_dir,
        env=MetaworldConfig(
            env_id="DRO-MT10",
            terminate_on_success=False,
        ),
        eval_env=MetaworldConfig(
            env_id="MT10",
            terminate_on_success=False,
        ),
        algorithm=PPOConfig(
            num_tasks=10,
            gamma=0.99,
            policy_config=get_actor_config(args.actor_type),
            vf_config=get_value_function_config(args.baseline_type),
            baseline_type=args.baseline_type,
            num_epochs=8,
            num_gradient_steps=32,
            gae_lambda=0.97,
            target_kl=None,
            clip_vf_loss=False,
            normalize_advantages=False,
            dro_upd_num_steps=args.rollout_steps, # this is per environment, so this is 100k steps = 200 trajectories
            # dro_lr=0.1,
            # dro_eps=0.05,
            # dro_success_ref=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        ),
        training_config=OnPolicyTrainingConfig(
            total_steps=args.total_steps,
            rollout_steps=args.rollout_steps,
            evaluation_frequency=args.evaluation_frequency,
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
            config=run,
            resume="allow",
            group=args.wandb_group,
        )

    run.start()


if __name__ == "__main__":
    main()
