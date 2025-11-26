from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import tyro

from metaworld_algorithms.config.networks import (
    ContinuousActionPolicyConfig, ValueFunctionConfig,
)
from metaworld_algorithms.config.nn import (
    VanillaNetworkConfig, MultiHeadConfig, MOOREConfig, PaCoConfig
)
from metaworld_algorithms.config.optim import OptimizerConfig
from metaworld_algorithms.config.rl import OnPolicyTrainingConfig
from metaworld_algorithms.rl.algorithms import PPOConfig
from metaworld_algorithms.run import Run
from custom_envs.pointmaze import PointMazeConfig


# ---------------------------------------------------------------------------
#  Shared helpers
# ---------------------------------------------------------------------------

def build_network_config(net_type: str, num_tasks: int):
    """Return appropriate network_config based on network type."""
    if net_type == "single_head":
        return VanillaNetworkConfig(
            optimizer=OptimizerConfig(lr=1e-3, max_grad_norm=1.0)
        )

    elif net_type == "multi_head":
        return MultiHeadConfig(
            num_tasks=num_tasks,
            optimizer=OptimizerConfig(lr=1e-3, max_grad_norm=1.0),
        )

    elif net_type == "moore":
        return MOOREConfig(
            num_tasks=num_tasks,
            optimizer=OptimizerConfig(lr=1e-3, max_grad_norm=1.0),
        )

    elif net_type == "paco":
        return PaCoConfig(
            num_tasks=num_tasks,
            num_parameter_sets=20,
            optimizer=OptimizerConfig(lr=1e-3, max_grad_norm=1.0),
        )

    else:
        raise ValueError(f"Invalid network type: {net_type}")


# ---------------------------------------------------------------------------
#  Actor config
# ---------------------------------------------------------------------------

def get_actor_config(actor_type: str, num_tasks: int) -> ContinuousActionPolicyConfig:
    net = build_network_config(actor_type, num_tasks)

    # special handling for MOORE (bounded log_std)
    if actor_type == "moore":
        return ContinuousActionPolicyConfig(
            network_config=net,
            log_std_min=-10,
            log_std_max=2,
            squash_tanh=False,
        )

    return ContinuousActionPolicyConfig(
        network_config=net,
        squash_tanh=False,
    )


# ---------------------------------------------------------------------------
#  Value function config (RESPECTS BASELINE CONSTRAINTS)
# ---------------------------------------------------------------------------

def get_value_function_config(
    value_type: str,
    baseline_type: str,
    num_tasks: int
) -> ValueFunctionConfig | None:
    """
    baseline_type ∈ {"linear", "mlp"}
    value_type    ∈ {"linear", "single_head", "multi_head", "mlp", "moore"}

    Constraint:
        - If value_type == "linear": baseline_type must be "linear"
        - Otherwise: baseline_type must be "mlp"
    """

    # ----------- enforce constraints -----------
    if value_type == "linear":
        if baseline_type != "linear":
            raise ValueError(
                "If value_type='linear', baseline_type must also be 'linear', "
                f"but got baseline_type='{baseline_type}'"
            )
        return None  # linear baseline = no network

    # non-linear value-function:
    if baseline_type != "mlp":
        raise ValueError(
            f"If value_type='{value_type}', baseline_type must be 'mlp', "
            f"but got baseline_type='{baseline_type}'"
        )

    # ----------- build network -----------
    net = build_network_config(value_type, num_tasks)
    return ValueFunctionConfig(network_config=net)


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

@dataclass
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

    env_id: str = 'dro_pointmaze'
    total_steps: int = int(10e7)
    evaluation_frequency: int = 2_000_000 // 500

    learning_rate: float = 3e-4
    num_epochs: int = 8
    num_gradient_steps: int = 32
    rollout_steps: int = 10_000
    normalize_advantages: int = 1
    reset_optimizer_steps: int = -1

    actor_type: Literal["single_head", "multi_head", "moore", "paco"] = "multi_head"
    value_type: Literal["linear", "single_head", "multi_head", "mlp", "moore"] = "multi_head"

    dro_rollout_steps: int = 10_000
    dro_learning_rate: float = 1.0
    dro_eps: float = 0.05
    dro_min_prob: float = 0.05


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = tyro.cli(Args)
    args.seed += args.seed_offset

    if args.env_id == 'dro_pointmaze':
        num_tasks = 3
    else:
        raise ValueError(f"Invalid env_id: {args.env_id}")

    args.baseline_type = 'linear' if args.value_type == 'linear' else 'mlp'

    run = Run(
        run_name=f"pointmaze_ppo_dro_{args.seed}",
        seed=args.seed,
        data_dir=args.data_dir,
        env=PointMazeConfig(env_id="Pointmaze/dro_pointmaze", terminate_on_success=False),
        eval_env=PointMazeConfig(env_id="Pointmaze/eval_pointmaze", terminate_on_success=True),
        algorithm=PPOConfig(
            num_tasks=num_tasks,
            gamma=0.99,
            policy_config=get_actor_config(args.actor_type, num_tasks),
            vf_config=get_value_function_config(args.value_type, args.baseline_type, num_tasks),
            baseline_type=args.baseline_type,
            num_epochs=8,
            num_gradient_steps=32,
            gae_lambda=0.97,
            target_kl=0.05,
            entropy_coefficient=1e-2,
            clip_vf_loss=False,
            reset_optimizer_steps=args.reset_optimizer_steps,
            normalize_advantages=args.normalize_advantages,
            dro_upd_num_steps=args.dro_rollout_steps,
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
        run.enable_wandb(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=run,
            resume="allow",
            group=args.wandb_group,
        )

    print(args)
    run.start()


if __name__ == "__main__":
    main()