from dataclasses import dataclass
from pathlib import Path

import numpy as np
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


@dataclass(frozen=True)
class Args:
    seed: int = 1
    track: bool = True
    wandb_project: str = 'dro'
    wandb_entity: str = 'nicholascorrado'
    data_dir: Path = Path("./run_results")
    resume: bool = False
    wandb_group: str | None = None


def main() -> None:
    args = tyro.cli(Args)

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
            dro_upd_num_steps=10_000, # this is per environment, so this is 100k steps = 200 trajectories
            # dro_lr=0.1,
            # dro_eps=0.05,
            # dro_success_ref=np.array([1, 1, 1, 0, 1, 1, 1, 1, 1, 1])
        ),
        training_config=OnPolicyTrainingConfig(
            total_steps=int(1e6),
            rollout_steps=10_000,
            evaluation_frequency=1_000_000 // 2500, # is this a reasonable evaluation frequency?
        ),
        checkpoint=True,
        resume=args.resume,
        dro=True,
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
