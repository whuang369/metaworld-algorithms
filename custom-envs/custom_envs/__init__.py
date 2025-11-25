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

pointmaze_steps = [1000, 400, 400, 400, 400]
for i in range(1, 5):
    register(
        id=f"PointMaze{i}-v0",
        entry_point=f"custom_envs.envs.pointmaze:PointMazeEnv{i}",
        max_episode_steps=pointmaze_steps[i],
    )

register(
    id=f"UMaze-v0",
    entry_point=f"custom_envs.envs.pointmaze:UMazeEnv",
    max_episode_steps=400,
)

register(
    id=f"Medium-v0",
    entry_point=f"custom_envs.envs.pointmaze:MediumEnv",
    max_episode_steps=400,
)

register(
    id=f"Large-v0",
    entry_point=f"custom_envs.envs.pointmaze:LargeEnv",
    max_episode_steps=400,
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

POINTMAZE_DRO_VARIANTS: OrderedDict[str, str] = OrderedDict(
    [
        ("UMaze-v0", "UMaze-v0"),
        ("Medium-v0", "Medium-v0"),
        ("Large-v0", "Large-v0"),
    ]
)
POINTMAZE_DRO_NUM_ENVS = 3


def _encode_pointmaze_task(env_name: str, worker_id: int) -> Task:
    """Encode a PointMaze task payload so MultiTaskDROWrapper can track it."""
    payload = dict(env_name=env_name, worker_id=worker_id)
    return Task(env_name=env_name, data=pickle.dumps(payload))


class PointMazeTaskEnv(gym.Env):
    """Proxy around Gymnasium PointMaze variants with Meta-World style hooks."""

    metadata = {}

    def __init__(self, env_name: str, render_mode: str | None = None):
        super().__init__()
        self._env_name = env_name
        self._render_mode = render_mode
        self._env = gym.make(env_name)
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space
        if (
            isinstance(self.observation_space, gym.spaces.Dict)
            and "desired_goal" in self.observation_space.spaces
        ):
            self.goal_space = self.observation_space.spaces["desired_goal"]
        else:
            self.goal_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(1,),
                dtype=np.float64,
            )
        self._current_task: Task | None = None

    def set_task(self, task: Task) -> None:
        assert (
            task.env_name == self._env_name
        ), f"Cannot set task {task.env_name} on {self._env_name}"
        self._current_task = task

    def step(self, action):
        return self._env.step(action)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        return self._env.reset(seed=seed, options=options)

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()


def _init_pointmaze_task_env(
    env_name: str,
    *,
    seed: int | None = None,
    terminate_on_success: bool = False,
    use_one_hot: bool = False,
    env_id: int | None = None,
    num_tasks: int | None = None,
    recurrent_info_in_obs: bool = False,
    normalize_reward_in_recurrent_info: bool = True,
    reward_normalization_method: Literal["gymnasium", "exponential"] | None = None,
    normalize_observations: bool = False,
    reward_alpha: float = 0.001,
    flatten_observations: bool = False,
    render_mode: Literal["human", "rgb_array", "depth_array"] | None = None,
):
    """Create a single PointMaze env wrapped like Meta-World tasks."""
    env = PointMazeTaskEnv(env_name, render_mode=render_mode)
    # if seed is not None:
    #     env.seed(seed)  # type: ignore
    env = gym.wrappers.TimeLimit(env, getattr(env, "max_episode_steps", 200))  # type: ignore
    env = AutoTerminateOnSuccessWrapper(env)
    env.toggle_terminate_on_success(terminate_on_success)
    if use_one_hot:
        assert env_id is not None, "Need to pass env_id through constructor"
        assert num_tasks is not None, "Need to pass num_tasks through constructor"
        env = OneHotWrapper(env, env_id, num_tasks)
    if recurrent_info_in_obs:
        env = RNNBasedMetaRLWrapper(
            env, normalize_reward=normalize_reward_in_recurrent_info
        )
    if reward_normalization_method == "gymnasium":
        env = gym.wrappers.NormalizeReward(env)
    elif reward_normalization_method == "exponential":
        env = NormalizeRewardsExponential(reward_alpha=reward_alpha, env=env)
    if normalize_observations:
        env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if seed is not None:
        env.action_space.seed(seed)
    return env


def make_dro_pointmaze_envs(
    seed: int | None = None,
    vector_strategy: Literal["sync", "async"] = "async",
    autoreset_mode: gym.vector.AutoresetMode | str = gym.vector.AutoresetMode.SAME_STEP,
    terminate_on_success: bool = False,
    use_one_hot: bool = False,
    recurrent_info_in_obs: bool = False,
    normalize_reward_in_recurrent_info: bool = True,
    reward_normalization_method: Literal["gymnasium", "exponential"] | None = None,
    normalize_observations: bool = False,
    reward_alpha: float = 0.001,
    render_mode: Literal["human", "rgb_array", "depth_array"] | None = None,
    flatten_observations: bool = True,
) -> gym.vector.VectorEnv:
    """
    Create a DRO-ready PointMaze vector env with MultiTaskDROWrapper.

    Each async worker wraps the PointMaze Open/Medium/Large tasks and can
    rebalance sampling probability via MultiTaskDROWrapper.
    """
    vectorizer: type[gym.vector.VectorEnv] = getattr(
        gym.vector, f"{vector_strategy.capitalize()}VectorEnv"
    )
    env_tasks = [
        [
            _encode_pointmaze_task(env_name, env_idx)
            for env_name in POINTMAZE_DRO_VARIANTS.keys()
        ]
        for env_idx in range(POINTMAZE_DRO_NUM_ENVS)
    ]

    def _build_pointmaze_env(
        env_cls: str,
        tasks: list[Task],
        env_id: int | None = None,
        num_tasks: int | None = None,
    ) -> gym.Env:
        del tasks  # Tasks do not customize the PointMaze variants yet.
        worker_seed = (
            seed + env_id if seed is not None and env_id is not None else seed
        )
        return _init_pointmaze_task_env(
            env_name=env_cls,
            seed=worker_seed,
            terminate_on_success=terminate_on_success,
            use_one_hot=use_one_hot,
            env_id=env_id,
            num_tasks=num_tasks,
            recurrent_info_in_obs=recurrent_info_in_obs,
            normalize_reward_in_recurrent_info=normalize_reward_in_recurrent_info,
            reward_normalization_method=reward_normalization_method,
            normalize_observations=normalize_observations,
            reward_alpha=reward_alpha,
            flatten_observations=flatten_observations,
            render_mode=render_mode,
        )

    def _make_env_internal(i: int) -> gym.Env:
        first_env_name = next(iter(POINTMAZE_DRO_VARIANTS.keys()))
        first_env_cls = POINTMAZE_DRO_VARIANTS[first_env_name]
        first_tasks = [task for task in env_tasks[i] if task.env_name == first_env_name]
        env = _build_pointmaze_env(
            env_cls=first_env_cls,
            tasks=first_tasks,
            env_id=i,
            num_tasks=len(POINTMAZE_DRO_VARIANTS),
        )
        env = MultiTaskDROWrapper(
            env,
            env_tasks[i],
            POINTMAZE_DRO_VARIANTS,
            include_task_one_hot=use_one_hot,
            env_factory=_build_pointmaze_env,
            env_id=i,
        )
        env = CheckpointWrapper(env, f"DRO_PointMaze_{i}")
        return env

    return vectorizer(
        [partial(_make_env_internal, i=i) for i in range(POINTMAZE_DRO_NUM_ENVS)],
        autoreset_mode=autoreset_mode,
    )


def _dro_pointmaze_vector_entry_point(
    vector_strategy: str = "async",
    autoreset_mode: gym.vector.AutoresetMode | str = gym.vector.AutoresetMode.SAME_STEP,
    seed: int | None = None,
    use_one_hot: bool = False,
    num_envs: int | None = None,
    **kwargs,
) -> gym.vector.VectorEnv:
    del num_envs  # Gymnasium passes this when vectorizing; fixed worker count here.
    return make_dro_pointmaze_envs(
        seed=seed,
        vector_strategy=vector_strategy,
        autoreset_mode=autoreset_mode,
        use_one_hot=use_one_hot,
        **kwargs,
    )


register(
    id="Pointmaze/dro_pointmaze",
    vector_entry_point=lambda vector_strategy="async",
                              autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
                              seed=None,
                              use_one_hot=False,
                              num_envs=None,
                              **kwargs: _dro_pointmaze_vector_entry_point(
        vector_strategy=vector_strategy,
        autoreset_mode=autoreset_mode,
        seed=seed,
        use_one_hot=use_one_hot,
        num_envs=num_envs,
        **kwargs,
    ),
    kwargs={},
)
