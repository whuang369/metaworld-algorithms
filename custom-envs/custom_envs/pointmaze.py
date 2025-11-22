# pyright: reportAttributeAccessIssue=false, reportIncompatibleMethodOverride=false, reportOptionalMemberAccess=false
from dataclasses import dataclass
from functools import cached_property
from typing import override

import gymnasium as gym
import numpy as np

from metaworld_algorithms.types import Agent, GymVectorEnv

from metaworld_algorithms.config.envs import EnvConfig

from custom_envs.evaluation import evaluation


@dataclass(frozen=True)
class PointMazeConfig(EnvConfig):
    use_one_hot: bool = False
    vector_strategy: str = "async"
    autoreset_mode: gym.vector.AutoresetMode | str = gym.vector.AutoresetMode.SAME_STEP
    recurrent_info_in_obs: bool = False
    normalize_reward_in_recurrent_info: bool = True
    reward_normalization_method: str | None = None
    normalize_observations: bool = False
    reward_alpha: float = 0.001
    render_mode: str | None = None
    num_tasks: int = 4

    @cached_property
    @override
    def action_space(self) -> gym.Space:
        return gym.spaces.Box(-1.0, 1.0, (2,), np.float32)

    @cached_property
    @override
    def observation_space(self) -> gym.Space:
        task_low = [0.0 for _ in range(self.num_tasks)]
        task_high = [1.0 for _ in range(self.num_tasks)]
        env_obs_space = gym.spaces.Box(
            np.array([-np.inf for _ in range(8)] + task_low, dtype=np.float64),
            np.array([np.inf for _ in range(8)] + task_high, dtype=np.float64),
        )

        if self.use_one_hot:
            one_hot_lb = np.zeros(self.num_tasks)
            one_hot_ub = np.ones(self.num_tasks)
            env_obs_space = gym.spaces.Box(
                np.concatenate([env_obs_space.low, one_hot_lb]),
                np.concatenate([env_obs_space.high, one_hot_ub]),
                dtype=np.float64,
            )

        if self.recurrent_info_in_obs:
            assert isinstance(self.action_space, gym.spaces.Box)
            env_obs_space = gym.spaces.Box(
                np.concatenate(
                    [env_obs_space.low, self.action_space.low, [-np.inf], [0.0]]
                ),
                np.concatenate(
                    [env_obs_space.high, self.action_space.high, [np.inf], [1.0]]
                ),
                dtype=np.float64,
            )

        return env_obs_space

    @override
    def evaluate(
        self, envs: GymVectorEnv, agent: Agent
    ) -> tuple[float, float, dict[str, float]]:
        return evaluation(agent, envs, num_episodes=self.evaluation_num_episodes)[:3]

    @override
    def spawn(self, seed: int = 1) -> GymVectorEnv:
        env_id = (
            self.env_id
            if "/" in self.env_id
            else f"Pointmaze/{self.env_id}"
        )
        return gym.make_vec(  # pyright: ignore[reportReturnType]
            env_id,
            seed=seed,
            vector_strategy=self.vector_strategy,
            autoreset_mode=self.autoreset_mode,
            use_one_hot=self.use_one_hot,
            terminate_on_success=self.terminate_on_success,
            recurrent_info_in_obs=self.recurrent_info_in_obs,
            normalize_reward_in_recurrent_info=self.normalize_reward_in_recurrent_info,
            reward_normalization_method=self.reward_normalization_method,
            normalize_observations=self.normalize_observations,
            reward_alpha=self.reward_alpha,
            render_mode=self.render_mode,
        )
