import os

import gymnasium as gym

from gymnasium.envs.registration import register

import abc
import pickle
from collections import OrderedDict
from functools import partial
from typing import Any, Literal, Union

import gymnasium as gym  # type: ignore
import numpy as np
import numpy.typing as npt

import gymnasium_robotics

# noqa: D104
from gymnasium.envs.registration import register

import metaworld.env_dict as _env_dict
from metaworld.env_dict import (
    ALL_V3_ENVIRONMENTS,
    ALL_V3_ENVIRONMENTS_GOAL_HIDDEN,
    ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE,
)
from metaworld.sawyer_xyz_env import SawyerXYZEnv  # type: ignore
from metaworld.types import Task  # type: ignore
from metaworld.wrappers import (
    AutoTerminateOnSuccessWrapper,
    CheckpointWrapper,
    NormalizeRewardsExponential,
    OneHotWrapper,
    PseudoRandomTaskSelectWrapper,
    RandomTaskSelectWrapper,
    RNNBasedMetaRLWrapper,
    MultiTaskDROWrapper,
)

ENVS_DIR = os.path.join(os.path.dirname(__file__), 'envs')

############################################################################
### Toy

pointmaze_steps = [1000, 100, 200, 400, 400]
for i in range(1, 5):
    register(
        id=f"PointMaze{i}-v0",
        entry_point=f"custom_envs.envs.pointmaze:PointMazeEnv{i}",
        max_episode_steps=pointmaze_steps[i],
    )

def make_custom_pointmaze_envs(
    seed: int | None = None,
    vector_strategy: Literal["sync", "async"] = "async",
    autoreset_mode: gym.vector.AutoresetMode | str = gym.vector.AutoresetMode.SAME_STEP,
    use_one_hot: bool = False,
    terminate_on_success: bool = False,
    recurrent_info_in_obs: bool = False,
    normalize_reward_in_recurrent_info: bool = True,
    reward_normalization_method: Literal["gymnasium", "exponential"] | None = None,
    normalize_observations: bool = False,
    reward_alpha: float = 0.001,
    render_mode: Literal["human", "rgb_array", "depth_array"] | None = None,
    **kwargs,
) -> gym.vector.VectorEnv:
    """
    Create a vectorized EasyGridWorld environment containing:
        EasyGridWorldEnv1–4.
    Mirrors Meta-World's make_mt_envs but all logic is inline.
    """
    # Map each environment ID to its constructor
    env_classes = {
        "PointMaze1": lambda: gym.make("PointMaze1"),
        "PointMaze2": lambda: gym.make("PointMaze2"),
        "PointMaze3": lambda: gym.make("PointMaze3"),
        "PointMaze4": lambda: gym.make("PointMaze4"),
    }

    vectorizer: type[gym.vector.VectorEnv] = getattr(
        gym.vector, f"{vector_strategy.capitalize()}VectorEnv"
    )
    num_tasks = len(env_classes)

    # --- Inline environment factory for each sub-env ---
    def make_single_env(env_fn, env_id: int):
        env = env_fn()

        if seed is not None:
            env.reset(seed=seed + env_id)

        # Meta-World–style wrappers
        env = gym.wrappers.TimeLimit(env, getattr(env, "max_episode_steps", 200))
        env = AutoTerminateOnSuccessWrapper(env)
        env.toggle_terminate_on_success(terminate_on_success)

        if use_one_hot:
            env = OneHotWrapper(env, env_id, num_tasks)

        if recurrent_info_in_obs:
            env = RNNBasedMetaRLWrapper(env, normalize_reward=normalize_reward_in_recurrent_info)

        if reward_normalization_method == "gymnasium":
            env = gym.wrappers.NormalizeReward(env)
        elif reward_normalization_method == "exponential":
            env = NormalizeRewardsExponential(reward_alpha=reward_alpha, env=env)

        if normalize_observations:
            env = gym.wrappers.NormalizeObservation(env)

        env = gym.wrappers.RecordEpisodeStatistics(env)
        # env = CheckpointWrapper(env, f"EasyGridWorldEnv{env_id+1}")

        if seed is not None:
            env.action_space.seed(seed + env_id)

        return env

    # --- Build env constructors for AsyncVectorEnv/SyncVectorEnv ---
    env_fns = [
        partial(make_single_env, env_fn=env_fn, env_id=env_id)
        for env_id, env_fn in enumerate(env_classes.values())
    ]

    return vectorizer(env_fns, autoreset_mode=autoreset_mode)

def _custom_pointmaze_vector_entry_point(
        vector_strategy: str = "async",
        autoreset_mode: gym.vector.AutoresetMode | str = gym.vector.AutoresetMode.SAME_STEP,
        seed: int | None = None,
        use_one_hot: bool = False,
        num_envs: int | None = None,
        **kwargs,
):
    # This mirrors _dro_mt_bench_vector_entry_point in Meta-World

    return make_custom_pointmaze_envs(
        seed=seed,
        vector_strategy=vector_strategy,
        autoreset_mode=autoreset_mode,
        use_one_hot=use_one_hot,
        **kwargs,
    )

register(
    id="Pointmaze/custom_pointmaze",
    vector_entry_point=lambda vector_strategy="async",
                              autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
                              seed=None,
                              use_one_hot=False,
                              num_envs=None,
                              **kwargs: _custom_pointmaze_vector_entry_point(
        vector_strategy=vector_strategy,
        autoreset_mode=autoreset_mode,
        seed=seed,
        use_one_hot=use_one_hot,
        num_envs=num_envs,
        **kwargs,
    ),
    kwargs={},
)