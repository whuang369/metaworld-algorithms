import abc
import time
from collections import deque
from typing import Deque, Generic, Self, TypeVar, override

import gymnasium as gym
import numpy as np
import numpy.typing as npt
import orbax.checkpoint as ocp
from flax import struct
from flax.training.train_state import TrainState
from jaxtyping import Float

from metaworld_algorithms.checkpoint import get_checkpoint_save_args
from metaworld_algorithms.config.envs import EnvConfig, MetaLearningEnvConfig
from metaworld_algorithms.config.rl import (
    AlgorithmConfig,
    GradientBasedMetaLearningTrainingConfig,
    MetaLearningTrainingConfig,
    OffPolicyTrainingConfig,
    OnPolicyTrainingConfig,
    RNNBasedMetaLearningTrainingConfig,
    TrainingConfig,
)
from metaworld_algorithms.monitoring.utils import log
from metaworld_algorithms.rl.buffers import (
    AbstractReplayBuffer,
    MultiTaskRolloutBuffer,
)
from metaworld_algorithms.types import (
    Action,
    AuxPolicyOutputs,
    CheckpointMetadata,
    GymVectorEnv,
    LogDict,
    MetaLearningAgent,
    Observation,
    ReplayBufferCheckpoint,
    ReplayBufferSamples,
    RNNState,
    Rollout,
)
from metaworld.env_dict import ALL_V3_ENVIRONMENTS, MT10_V3, MT25_V3, MT50_V3

AlgorithmConfigType = TypeVar("AlgorithmConfigType", bound=AlgorithmConfig)
TrainingConfigType = TypeVar("TrainingConfigType", bound=TrainingConfig)
EnvConfigType = TypeVar("EnvConfigType", bound=EnvConfig)
MetaLearningTrainingConfigType = TypeVar(
    "MetaLearningTrainingConfigType", bound=MetaLearningTrainingConfig
)
DataType = TypeVar("DataType", ReplayBufferSamples, Rollout, list[Rollout])


from dataclasses import dataclass, field
import numpy as np
class SMTScheduler:
    def __init__(self,
                 num_tasks,
                 K=3,
                 threshold_low=np.array([2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000]),
                 threshold_high=np.array([4000, 3000, 3000, 4000, 4000, 4000, 3000, 3500, 4000, 4000]),
                 kappa=0.8,
                 total_budget=100_000_000,
                 stage1_budget=85_000_000,
                 min_prob=1/10/4):

        if num_tasks == 50:
            self.K = 8
            self.threshold_low = np.array([
                2000 for _ in range(50)
            ])
            self.threshold_high = np.array([
                3000,  # assembly-v3
                2000,  # basketball-v3
                3000,  # bin-picking-v3
                3000,  # box-close-v3
                3000,  # button-press-topdown-v3
                3000,  # button-press-topdown-wall-v3
                3500,  # button-press-v3
                3500,  # button-press-wall-v3
                3000,  # coffee-button-v3
                2000,  # coffee-pull-v3
                2000,  # coffee-push-v3
                3500,  # dial-turn-v3
                3000,  # disassemble-v3
                4000,  # door-close-v3
                3000,  # door-lock-v3
                4000,  # door-open-v3
                3000,  # door-unlock-v3
                3500,  # hand-insert-v3
                4000,  # drawer-close-v3
                4000,  # drawer-open-v3
                4000,  # faucet-open-v3
                4000,  # faucet-close-v3
                3000,  # hammer-v3
                4000,  # handle-press-side-v3
                4000,  # handle-press-v3
                3000,  # handle-pull-side-v3
                3500,  # handle-pull-v3
                2500,  # lever-pull-v3
                3000,  # pick-place-wall-v3
                3000,  # pick-out-of-hole-v3
                3000,  # pick-place-v3
                4000,  # plate-slide-v3
                3500,  # plate-slide-side-v3
                4000,  # plate-slide-back-v3
                4000,  # plate-slide-back-side-v3
                3500,  # peg-insert-side-v3
                3500,  # peg-unplug-side-v3
                2000,  # soccer-v3
                2500,  # stick-push-v3
                2000,  # stick-pull-v3
                3000,  # push-v3
                3000,  # push-wall-v3
                3000,  # push-back-v3
                4000,  # reach-v3
                4000,  # reach-wall-v3
                3000,  # shelf-place-v3
                3500,  # sweep-into-v3
                3000,  # sweep-v3
                4000,  # window-open-v3
                4000,  # window-close-v3
            ])
            self.kappa = 0.8
            self.total_budget = 500_000_000
            self.stage1_budget = 425_000_000
            self.min_prob=1/10/4
        else:
            self.K = K
            self.num_tasks = num_tasks
            self.threshold_low = threshold_low
            self.threshold_high = threshold_high
            self.kappa = kappa
            self.stage1_budget = stage1_budget
            self.min_prob = min_prob

        self.steps = 0
        self.eval_metrics = np.full(num_tasks, -np.inf)
        self.budgets = np.zeros(num_tasks, dtype=int)
        self.steps_since_scheduled = np.zeros(num_tasks, dtype=int)

        all_tasks = list(range(num_tasks))
        np.random.shuffle(all_tasks)
        self.pool_main       = all_tasks[K:]
        self.pool_active     = all_tasks[:K]
        self.pool_solved     = []
        self.pool_unsolvable = []

        self.budgets[self.pool_active] = int(kappa * total_budget / K)

    def _make_distribution(self):
        dist = np.full(self.num_tasks, self.min_prob)
        if self.pool_active:
            active_prob = (1.0 - self.min_prob * self.num_tasks) / len(self.pool_active)
            dist[self.pool_active] += active_prob

        dist /= np.sum(dist) # normalize so that when all tasks are solved, we go back to uniform
        print(f'{dist=}')

        return dist

    def _evict(self, task_id, dest):
        dest.append(task_id)
        self.pool_active.remove(task_id)
        self.steps_since_scheduled[task_id] = 0

    def update(self, steps_used, eval_metrics):
        """
        Args:
            steps_used:   np.ndarray of shape (num_tasks,), steps used per task since last update
            eval_metrics: np.ndarray of shape (num_tasks,), lower = harder
        Returns:
            np.ndarray of shape (num_tasks,) — sampling distribution over tasks
        """
        self.steps += steps_used.sum()
        self.steps_since_scheduled += steps_used
        self.eval_metrics = eval_metrics

        if self.steps >= self.stage1_budget:
            self.pool_active = self.pool_unsolvable
            self.pool_unsolvable = []
            return self._make_distribution()

        for task_id in list(self.pool_active):
            metric = self.eval_metrics[task_id]
            if metric > self.threshold_high[task_id]:
                self._evict(task_id, self.pool_solved)
            elif self.steps_since_scheduled[task_id] >= self.budgets[task_id]:
                dest = self.pool_unsolvable if metric < self.threshold_low[task_id] else self.pool_main
                self._evict(task_id, dest)

        remaining_budget = self.stage1_budget - self.steps
        while len(self.pool_active) < self.K and self.pool_main:
            new_task = min(self.pool_main, key=lambda t: self.eval_metrics[t])
            self.pool_main.remove(new_task)
            self.pool_active.append(new_task)
            self.budgets[new_task] = int(self.kappa * remaining_budget / self.K)
            self.steps_since_scheduled[new_task] = 0

        print(f'{self.budgets=}')

        return self._make_distribution()

class Algorithm(
    abc.ABC,
    Generic[AlgorithmConfigType, TrainingConfigType, EnvConfigType, DataType],
    struct.PyTreeNode,
):
    """Based on https://github.com/kevinzakka/nanorl/blob/main/nanorl/agent.py"""

    num_tasks: int = struct.field(pytree_node=False)
    gamma: float = struct.field(pytree_node=False)

    @staticmethod
    @abc.abstractmethod
    def initialize(
        config: AlgorithmConfigType, env_config: EnvConfigType, seed: int = 1
    ) -> "Algorithm": ...

    @abc.abstractmethod
    def get_num_params(self) -> dict[str, int]: ...

    @abc.abstractmethod
    def train(
        self,
        config: TrainingConfigType,
        envs: GymVectorEnv,
        env_config: EnvConfigType,
        run_timestamp: str | None = None,
        seed: int = 1,
        track: bool = True,
        checkpoint_manager: ocp.CheckpointManager | None = None,
        checkpoint_metadata: CheckpointMetadata | None = None,
        buffer_checkpoint: ReplayBufferCheckpoint | None = None,
        eval_env: GymVectorEnv = None,
    ) -> Self: ...


class MetaLearningAlgorithm(
    Algorithm[
        AlgorithmConfigType,
        MetaLearningTrainingConfigType,
        MetaLearningEnvConfig,
        DataType,
    ],
    Generic[AlgorithmConfigType, MetaLearningTrainingConfigType, DataType],
):
    @staticmethod
    @abc.abstractmethod
    def initialize(
        config: AlgorithmConfigType, env_config: MetaLearningEnvConfig, seed: int = 1
    ) -> "MetaLearningAlgorithm": ...

    @abc.abstractmethod
    def update(self, data: DataType) -> tuple[Self, LogDict]: ...

    @abc.abstractmethod
    def wrap(self) -> MetaLearningAgent: ...

    @abc.abstractmethod
    def train(
        self,
        config: MetaLearningTrainingConfigType,
        envs: GymVectorEnv,
        env_config: MetaLearningEnvConfig,
        run_timestamp: str | None = None,
        seed: int = 1,
        track: bool = True,
        checkpoint_manager: ocp.CheckpointManager | None = None,
        checkpoint_metadata: CheckpointMetadata | None = None,
        buffer_checkpoint: ReplayBufferCheckpoint | None = None,
        eval_env: GymVectorEnv = None,
    ) -> Self: ...


class GradientBasedMetaLearningAlgorithm(
    MetaLearningAlgorithm[
        AlgorithmConfigType, GradientBasedMetaLearningTrainingConfig, list[Rollout]
    ],
    Generic[AlgorithmConfigType],
):
    @abc.abstractmethod
    def sample_action_and_aux(
        self, observation: Observation
    ) -> tuple[Self, Action, AuxPolicyOutputs]: ...

    def spawn_rollout_buffer(
        self,
        env_config: EnvConfig,
        training_config: GradientBasedMetaLearningTrainingConfig,
        seed: int | None = None,
    ) -> MultiTaskRolloutBuffer:
        return MultiTaskRolloutBuffer(
            num_tasks=training_config.meta_batch_size,
            num_rollout_steps=training_config.rollouts_per_task
            * env_config.max_episode_steps,
            env_obs_space=env_config.observation_space,
            env_action_space=env_config.action_space,
            seed=seed,
        )

    @abc.abstractmethod
    def adapt(self, rollouts: Rollout) -> Self: ...

    @abc.abstractmethod
    def init_ensemble_networks(self) -> Self: ...

    @override
    def train(
        self,
        config: GradientBasedMetaLearningTrainingConfig,
        envs: GymVectorEnv,
        env_config: MetaLearningEnvConfig,
        run_timestamp: str | None = None,
        seed: int = 1,
        track: bool = True,
        checkpoint_manager: ocp.CheckpointManager | None = None,
        checkpoint_metadata: CheckpointMetadata | None = None,
        buffer_checkpoint: ReplayBufferCheckpoint | None = None,
        eval_env: GymVectorEnv = None,
    ) -> Self:
        global_episodic_return: Deque[float] = deque([], maxlen=20 * self.num_tasks)
        global_episodic_length: Deque[int] = deque([], maxlen=20 * self.num_tasks)
        start_step, episodes_ended = 0, 0

        if checkpoint_metadata is not None:
            start_step = checkpoint_metadata["step"]
            episodes_ended = checkpoint_metadata["episodes_ended"]

        rollout_buffer = self.spawn_rollout_buffer(env_config, config, seed)

        # NOTE: We assume that eval evns are deterministically initialised and there's no state
        # that needs to be carried over when they're used.
        eval_envs = env_config.spawn_test(seed)

        start_time = time.time()

        steps_per_iter = (
            config.meta_batch_size
            * config.rollouts_per_task
            * env_config.max_episode_steps
            * (config.num_inner_gradient_steps + 1)
        )

        for _iter in range(
            start_step, config.total_steps // steps_per_iter
        ):  # Outer step
            global_step = _iter * steps_per_iter
            print(f"Iteration {_iter}, Global num of steps {global_step}")

            envs.call("sample_tasks")
            self = self.init_ensemble_networks()
            all_rollouts: list[Rollout] = []

            # Sampling step
            # Collect num_inner_gradient_steps D datasets + collect 1 D' dataset
            for _step in range(config.num_inner_gradient_steps + 1):
                print(f"- Collecting inner step {_step}")
                obs, _ = envs.reset()
                rollout_buffer.reset()
                episode_started = np.ones((envs.num_envs,))

                while not rollout_buffer.ready:
                    self, actions, aux_policy_outs = self.sample_action_and_aux(obs)

                    next_obs, rewards, terminations, truncations, infos = envs.step(
                        actions
                    )

                    rollout_buffer.add(
                        obs,
                        actions,
                        rewards,
                        episode_started,
                        value=aux_policy_outs.get("value"),
                        log_prob=aux_policy_outs.get("log_prob"),
                        mean=aux_policy_outs.get("mean"),
                        std=aux_policy_outs.get("std"),
                    )

                    episode_started = np.logical_or(terminations, truncations)
                    obs = next_obs

                    for i, env_ended in enumerate(episode_started):
                        if env_ended:
                            global_episodic_return.append(
                                infos["final_info"]["episode"]["r"][i]
                            )
                            global_episodic_length.append(
                                infos["final_info"]["episode"]["l"][i]
                            )

                rollouts = rollout_buffer.get()
                all_rollouts.append(rollouts)

                # Inner policy update for the sake of sampling close to adapted policy during the
                # computation of the objective.
                if _step < config.num_inner_gradient_steps:
                    print(f"- Adaptation step {_step}")
                    self = self.adapt(rollouts)

            mean_episodic_return = np.mean(list(global_episodic_return))
            print("- Mean episodic return: ", mean_episodic_return)
            if track:
                log(
                    {"charts/mean_episodic_returns": mean_episodic_return},
                    step=global_step,
                )

            # Outer policy update
            print("- Computing outer step")
            self, logs = self.update(all_rollouts)

            # Evaluation
            if global_step % config.evaluation_frequency == 0 and global_step > 0:
                print("- Evaluating on the test set...")
                mean_success_rate, mean_returns, mean_success_per_task = (
                    env_config.evaluate_metalearning(eval_envs, self.wrap())
                )

                eval_metrics = {
                    "charts/mean_success_rate": float(mean_success_rate),
                    "charts/mean_evaluation_return": float(mean_returns),
                } | {
                    f"charts/{task_name}_success_rate": float(success_rate)
                    for task_name, success_rate in mean_success_per_task.items()
                }

                if config.evaluate_on_train:
                    print("- Evaluating on the train set...")
                    _, _, eval_success_rate_per_train_task = (
                        env_config.evaluate_metalearning_on_train(
                            envs=envs,
                            agent=self.wrap(),
                        )
                    )
                    for (
                        task_name,
                        success_rate,
                    ) in eval_success_rate_per_train_task.items():
                        eval_metrics[f"charts/{task_name}_train_success_rate"] = float(
                            success_rate
                        )

                print(
                    f"Mean evaluation success rate: {mean_success_rate:.4f}"
                    + f" return: {mean_returns:.4f}"
                )

                if track:
                    log(eval_metrics, step=global_step)

                if checkpoint_manager is not None:
                    checkpoint_manager.save(
                        global_step,
                        args=get_checkpoint_save_args(
                            self,
                            envs,
                            global_step,
                            episodes_ended,
                            run_timestamp,
                        ),
                        metrics={
                            k.removeprefix("charts/"): v
                            for k, v in eval_metrics.items()
                        },
                    )
                    print("- Saved Model")

            # Logging
            print(logs)
            sps = global_step / (time.time() - start_time)
            print("- SPS: ", sps)
            if track:
                log({"charts/SPS": sps} | logs, step=global_step)

        eval_envs.close()
        del eval_envs

        return self


class RNNBasedMetaLearningAlgorithm(
    MetaLearningAlgorithm[
        AlgorithmConfigType, RNNBasedMetaLearningTrainingConfig, Rollout
    ],
    Generic[AlgorithmConfigType],
):
    @abc.abstractmethod
    def sample_action_and_aux(
        self, state: RNNState, observation: Observation
    ) -> tuple[Self, RNNState, Action, AuxPolicyOutputs]: ...

    def spawn_rollout_buffer(
        self,
        env_config: EnvConfig,
        training_config: RNNBasedMetaLearningTrainingConfig,
        example_state: RNNState,
        seed: int | None = None,
    ) -> MultiTaskRolloutBuffer:
        return MultiTaskRolloutBuffer(
            num_tasks=training_config.meta_batch_size,
            num_rollout_steps=training_config.rollouts_per_task
            * env_config.max_episode_steps,
            env_obs_space=env_config.observation_space,
            env_action_space=env_config.action_space,
            rnn_state_dim=example_state.shape[-1],
            seed=seed,
        )

    @abc.abstractmethod
    def init_recurrent_state(self, batch_size: int) -> tuple[Self, RNNState]: ...

    @abc.abstractmethod
    def reset_recurrent_state(
        self, current_state: RNNState, reset_mask: npt.NDArray[np.bool_]
    ) -> tuple[Self, RNNState]: ...

    @override
    def train(
        self,
        config: RNNBasedMetaLearningTrainingConfig,
        envs: GymVectorEnv,
        env_config: MetaLearningEnvConfig,
        run_timestamp: str | None = None,
        seed: int = 1,
        track: bool = True,
        checkpoint_manager: ocp.CheckpointManager | None = None,
        checkpoint_metadata: CheckpointMetadata | None = None,
        buffer_checkpoint: ReplayBufferCheckpoint | None = None,
        eval_env: GymVectorEnv = None,
    ) -> Self:
        global_episodic_return: Deque[float] = deque([], maxlen=20 * self.num_tasks)
        global_episodic_length: Deque[int] = deque([], maxlen=20 * self.num_tasks)
        start_step, episodes_ended = 0, 0

        if checkpoint_metadata is not None:
            start_step = checkpoint_metadata["step"]
            episodes_ended = checkpoint_metadata["episodes_ended"]

        _, example_state = self.init_recurrent_state(config.meta_batch_size)
        rollout_buffer = self.spawn_rollout_buffer(
            env_config, config, example_state, seed
        )

        # NOTE: We assume that eval evns are deterministically initialised and there's no state
        # that needs to be carried over when they're used.
        eval_envs = env_config.spawn_test(seed)

        start_time = time.time()

        steps_per_iter = (
            config.meta_batch_size
            * config.rollouts_per_task
            * env_config.max_episode_steps
        )

        for _iter in range(
            start_step, config.total_steps // steps_per_iter
        ):  # Outer step
            global_step = _iter * steps_per_iter
            print(f"Iteration {_iter}, Global num of steps {global_step}")

            envs.call("sample_tasks")
            self, states = self.init_recurrent_state(config.meta_batch_size)
            obs, _ = envs.reset()
            rollout_buffer.reset()
            episode_started = np.ones((envs.num_envs,))

            while not rollout_buffer.ready:
                self, next_states, actions, aux_policy_outs = (
                    self.sample_action_and_aux(states, obs)
                )

                next_obs, rewards, terminations, truncations, infos = envs.step(actions)

                rollout_buffer.add(
                    obs,
                    actions,
                    rewards,
                    episode_started,
                    value=aux_policy_outs.get("value"),
                    log_prob=aux_policy_outs.get("log_prob"),
                    mean=aux_policy_outs.get("mean"),
                    std=aux_policy_outs.get("std"),
                    rnn_state=states,
                )

                episode_started = np.logical_or(terminations, truncations)
                obs = next_obs
                states = next_states

                for i, env_ended in enumerate(episode_started):
                    if env_ended:
                        global_episodic_return.append(
                            infos["final_info"]["episode"]["r"][i]
                        )
                        global_episodic_length.append(
                            infos["final_info"]["episode"]["l"][i]
                        )

            rollouts = rollout_buffer.get()

            mean_episodic_return = np.mean(list(global_episodic_return))
            print("- Mean episodic return: ", mean_episodic_return)
            if track:
                log(
                    {"charts/mean_episodic_returns": mean_episodic_return},
                    step=global_step,
                )

            # Outer policy update
            print("- Computing update")
            self, logs = self.update(rollouts)

            # Evaluation
            if global_step % config.evaluation_frequency == 0 and global_step > 0:
                print("- Evaluating on the test set...")
                mean_success_rate, mean_returns, mean_success_per_task = (
                    env_config.evaluate_metalearning(eval_envs, self.wrap())
                )

                eval_metrics = {
                    "charts/mean_success_rate": float(mean_success_rate),
                    "charts/mean_evaluation_return": float(mean_returns),
                } | {
                    f"charts/{task_name}_success_rate": float(success_rate)
                    for task_name, success_rate in mean_success_per_task.items()
                }

                if config.evaluate_on_train:
                    print("- Evaluating on the train set...")
                    _, _, eval_success_rate_per_train_task = (
                        env_config.evaluate_metalearning_on_train(
                            envs=envs,
                            agent=self.wrap(),
                        )
                    )
                    for (
                        task_name,
                        success_rate,
                    ) in eval_success_rate_per_train_task.items():
                        eval_metrics[f"charts/{task_name}_train_success_rate"] = float(
                            success_rate
                        )

                print(
                    f"Mean evaluation success rate: {mean_success_rate:.4f}"
                    + f" return: {mean_returns:.4f}"
                )

                if track:
                    log(eval_metrics, step=global_step)

                if checkpoint_manager is not None:
                    checkpoint_manager.save(
                        global_step,
                        args=get_checkpoint_save_args(
                            self,
                            envs,
                            global_step,
                            episodes_ended,
                            run_timestamp,
                        ),
                        metrics={
                            k.removeprefix("charts/"): v
                            for k, v in eval_metrics.items()
                        },
                    )
                    print("- Saved Model")

            # Logging
            print(
                {
                    k: v
                    for k, v in logs.items()
                    if not (k.startswith("nn") or k.startswith("data"))
                }
            )
            sps = global_step / (time.time() - start_time)
            print("- SPS: ", sps)
            if track:
                log({"charts/SPS": sps} | logs, step=global_step)

        eval_envs.close()
        del eval_envs

        return self


class OffPolicyAlgorithm(
    Algorithm[
        AlgorithmConfigType, OffPolicyTrainingConfig, EnvConfig, ReplayBufferSamples
    ],
    Generic[AlgorithmConfigType],
):
    @abc.abstractmethod
    def spawn_replay_buffer(
        self, env_config: EnvConfig, config: OffPolicyTrainingConfig, seed: int = 1
    ) -> AbstractReplayBuffer: ...

    @abc.abstractmethod
    def update(self, data: ReplayBufferSamples) -> tuple[Self, LogDict]: ...

    @abc.abstractmethod
    def sample_action(self, observation: Observation) -> tuple[Self, Action]: ...

    @abc.abstractmethod
    def eval_action(self, observations: Observation) -> Action: ...

    def reset(self, env_mask: npt.NDArray[np.bool_]) -> None:
        del env_mask
        pass  # For evaluation interface compatibility

    @override
    def train(
        self,
        config: OffPolicyTrainingConfig,
        envs: GymVectorEnv,
        env_config: EnvConfig,
        run_timestamp: str | None = None,
        seed: int = 1,
        track: bool = True,
        checkpoint_manager: ocp.CheckpointManager | None = None,
        checkpoint_metadata: CheckpointMetadata | None = None,
        buffer_checkpoint: ReplayBufferCheckpoint | None = None,
        eval_env: GymVectorEnv = None,
    ) -> Self:
        global_episodic_return: Deque[float] = deque([], maxlen=20 * self.num_tasks)
        global_episodic_length: Deque[int] = deque([], maxlen=20 * self.num_tasks)

        obs, _ = envs.reset()

        done = np.full((envs.num_envs,), False)
        start_step, episodes_ended = 0, 0

        if checkpoint_metadata is not None:
            start_step = checkpoint_metadata["step"]
            episodes_ended = checkpoint_metadata["episodes_ended"]

        replay_buffer = self.spawn_replay_buffer(env_config, config, seed)
        if buffer_checkpoint is not None:
            replay_buffer.load_checkpoint(buffer_checkpoint)

        start_time = time.time()

        for global_step in range(start_step, config.total_steps // envs.num_envs):
            total_steps = global_step * envs.num_envs

            if global_step < config.warmstart_steps:
                actions = envs.action_space.sample()
            else:
                self, actions = self.sample_action(obs)

            next_obs, rewards, terminations, truncations, infos = envs.step(actions)
            done = np.logical_or(terminations, truncations)

            buffer_obs = next_obs
            if "final_obs" in infos:
                buffer_obs = np.where(
                    done[:, None], np.stack(infos["final_obs"]), next_obs
                )
            replay_buffer.add(obs, buffer_obs, actions, rewards, done)

            obs = next_obs

            for i, env_ended in enumerate(done):
                if env_ended:
                    global_episodic_return.append(
                        infos["final_info"]["episode"]["r"][i]
                    )
                    global_episodic_length.append(
                        infos["final_info"]["episode"]["l"][i]
                    )
                    episodes_ended += 1

            if global_step % 500 == 0 and global_episodic_return:
                print(
                    f"global_step={total_steps}, mean_episodic_return={np.mean(list(global_episodic_return))}"
                )
                if track:
                    log(
                        {
                            "charts/mean_episodic_return": np.mean(
                                list(global_episodic_return)
                            ),
                            "charts/mean_episodic_length": np.mean(
                                list(global_episodic_length)
                            ),
                        },
                        step=total_steps,
                    )

            if global_step > config.warmstart_steps:
                # Update the agent with data
                data = replay_buffer.sample(config.batch_size)
                self, logs = self.update(data)

                # Logging
                if global_step % 100 == 0:
                    sps_steps = (global_step - start_step) * envs.num_envs
                    sps = int(sps_steps / (time.time() - start_time))
                    print("SPS:", sps)

                    if track:
                        log({"charts/SPS": sps} | logs, step=total_steps)

                # Evaluation
                if (
                    config.evaluation_frequency > 0
                    and episodes_ended % config.evaluation_frequency == 0
                    and done.any()
                    and global_step > 0
                ):
                    mean_success_rate, mean_returns, mean_success_per_task = (
                        env_config.evaluate(envs, self)
                    )
                    eval_metrics = {
                        "charts/mean_success_rate": float(mean_success_rate),
                        "charts/mean_evaluation_return": float(mean_returns),
                    } | {
                        f"charts/{task_name}_success_rate": float(success_rate)
                        for task_name, success_rate in mean_success_per_task.items()
                    }
                    print(
                        f"total_steps={total_steps}, mean evaluation success rate: {mean_success_rate:.4f}"
                        + f" return: {mean_returns:.4f}"
                    )

                    if track:
                        log(eval_metrics, step=total_steps)

                    # Checkpointing
                    if checkpoint_manager is not None:
                        if not done.all():
                            raise NotImplementedError(
                                "Checkpointing currently doesn't work for the case where evaluation is run before all envs have finished their episodes / are about to be reset."
                            )

                        checkpoint_manager.save(
                            total_steps,
                            args=get_checkpoint_save_args(
                                self,
                                envs,
                                global_step,
                                episodes_ended,
                                run_timestamp,
                                buffer=replay_buffer,
                            ),
                            metrics={
                                k.removeprefix("charts/"): v
                                for k, v in eval_metrics.items()
                            },
                        )

                    # Reset envs again to exit eval mode
                    obs, _ = envs.reset()

        return self


class OnPolicyAlgorithm(
    Algorithm[AlgorithmConfigType, OnPolicyTrainingConfig, EnvConfig, Rollout],
    Generic[AlgorithmConfigType],
):
    @abc.abstractmethod
    def sample_action_and_aux(
        self, observation: Observation
    ) -> tuple[Self, Action, AuxPolicyOutputs]: ...

    @abc.abstractmethod
    def sample_action(self, observation: Observation) -> tuple[Self, Action]: ...

    @abc.abstractmethod
    def eval_action(self, observations: Observation) -> Action: ...

    def reset(self, env_mask: npt.NDArray[np.bool_]) -> None:
        del env_mask
        pass  # For evaluation interface compatibility

    @abc.abstractmethod
    def update(
        self,
        data: Rollout,
        dones: Float[npt.NDArray, "task 1"],
        next_obs: Float[Observation, " task"] | None = None,
    ) -> tuple[Self, LogDict]: ...

    def spawn_rollout_buffer(
        self,
        env_config: EnvConfig,
        training_config: OnPolicyTrainingConfig,
        seed: int | None = None,
    ) -> MultiTaskRolloutBuffer:
        return MultiTaskRolloutBuffer(
            training_config.rollout_steps,
            self.num_tasks,
            env_config.observation_space,
            env_config.action_space,
            seed,
        )

    # @staticmethod
    # def update_task_weights(gaps, eta, base=None):
    #     """
    #     Compute KL-regularized DRO task weights using q_i ∝ base_i * exp(eta * gap_i).
    #
    #     Parameters
    #     ----------
    #     gaps : np.ndarray
    #         1D array of per-task gaps (e.g., reference - success).
    #     eta : float
    #         KL-DRO sharpness parameter (η = 0 gives uniform; η → ∞ gives argmax).
    #     base : np.ndarray or None
    #         Base distribution p0. If None, use uniform over tasks.
    #
    #     Returns
    #     -------
    #     np.ndarray
    #         Normalized task weights q (same shape as gaps).
    #     """
    #     gaps = np.asarray(gaps)
    #     if base is None:
    #         base = np.ones_like(gaps) / len(gaps)
    #     else:
    #         base = np.asarray(base)
    #         base = base / base.sum()  # ensure normalized
    #
    #     # Compute unnormalized exponentiated weights: p0_i * exp(η*g_i)
    #     logits = np.log(base + 1e-12) + eta * gaps
    #     weights = np.exp(logits - np.max(logits))  # stable softmax
    #
    #     return weights / weights.sum()

    # @staticmethod
    # def exponentiated_gradient_ascent_step(w, returns, returns_ref, learning_rate=0.1,
    #                                        eps=0.05, min_prob=0.03):
    #     # Use s_t - s_{t-1} instead of s_ref - s_t
    #     diff = np.clip(returns_ref - returns, 0, np.inf)
    #
    #     w_new = w * np.exp(learning_rate * diff)
    #
    #     # Normalize to ensure weights sum to 1
    #     w_new = w_new / w_new.sum()
    #
    #     # Smoothing to prevent weights form getting too close to 0
    #     w_uniform = 1 / len(w_new) * np.ones(len(w_new))
    #     w_new = (1 - eps) * w_new + eps * w_uniform
    #
    #     # w_focus = np.zeros(len(w_new))
    #     # w_focus[env_id] = 1.0
    #     # w_new = 0.5 * w_new + 0.5 * w_focus
    #     def clip_and_normalize(p, c=0.01):
    #         p = np.maximum(p, c)  # clip
    #         p = p / p.sum()  # renormalize
    #         return p
    #
    #     for i in range(10):
    #         w_new = clip_and_normalize(w_new, c=min_prob)
    #
    #     return w_new

    ##########################################################################
    # KL PROJECTION onto { q in simplex : q_i >= dro_eps }
    ##########################################################################
    @staticmethod
    def kl_project_with_floor(z, dro_eps):
        """
        KL projection of distribution z onto the convex set:
            { q : q_i >= dro_eps,  sum_i q_i = 1 }

        Solves:
            minimize_q KL(q || z)
            subject to q_i >= dro_eps.

        Returns a valid probability vector q.
        """
        z = np.asarray(z, dtype=float)
        z = z / z.sum()  # ensure distribution
        k = len(z)

        # Start with all coordinates "free"
        free = np.ones(k, dtype=bool)
        q = np.zeros_like(z)

        while True:
            num_clipped = (~free).sum()
            mass_free = 1.0 - dro_eps * num_clipped

            if mass_free < 0:
                # dro_eps too large to be feasible; fallback
                return np.ones(k) / k

            z_free_sum = z[free].sum()

            if z_free_sum == 0:
                # degenerate free mass
                q[free] = mass_free / free.sum()
            else:
                scale = mass_free / z_free_sum
                q[free] = scale * z[free]

            # Clipped ones set to dro_eps
            q[~free] = dro_eps

            # If any free entries fell below dro_eps, move them to clipped set
            violated = free & (q < dro_eps - 1e-12)
            if not violated.any():
                break

            free[violated] = False

        q /= q.sum()  # final normalization
        return q

    ##########################################################################
    # MAIN DRO UPDATE WITH OPTIONAL KL PROJECTION (dro_eps)
    ##########################################################################
    def update_task_weights(self, q, gap, eta, step_size, p0=None, dro_eps=None):
        """
        Perform ONE exponentiated-gradient (mirror-ascent) step on the
        KL-regularized DRO objective:

            maximize_q   gap^T q  -  (1/eta) * KL(q || p0)

        Includes an optional KL projection enforcing q_i >= dro_eps.

        Parameters
        ----------
        q : np.ndarray, shape (k,)
            Current task weights.
        gap : np.ndarray, shape (k,)
        eta : float
            DRO regularization strength.
        step_size : float
            Mirror ascent step size. Must satisfy 0 < step_size <= eta.
        p0 : np.ndarray or None
            Base distribution for KL regularization.
        dro_eps : float or None
            If provided, enforce q_i >= dro_eps by KL projection.

        Returns
        -------
        q_new : np.ndarray, shape (k,)
        """

        k = len(q)

        # Default p0: uniform
        if p0 is None:
            p0 = np.ones(k) / k

        # geometric Polyak averaging rate
        alpha = step_size / eta  # in [0,1]

        # Mirror-ascent step:
        #   log q_new ∝ (1-alpha) log q + alpha log p0 + step_size * gap
        log_q_new = (1 - alpha) * np.log(q) + alpha * np.log(p0) + step_size * gap

        # log-sum-exp normalize
        log_q_new -= np.max(log_q_new)
        q_new = np.exp(log_q_new)
        q_new /= q_new.sum()

        # KL PROJECT if dro_eps provided
        if dro_eps is not None and dro_eps > 0:
            q_new = self.kl_project_with_floor(q_new, dro_eps)

        return q_new

    @staticmethod
    def set_task_distributions(
        envs: GymVectorEnv,
        distribution: npt.NDArray[np.float64],
    ) -> None:
        """Set the task probability distribution for all environments in the vector environment.

        Args:
            envs: The vector environment containing multiple wrapped environments
            distributions: List of probability distributions, one per environment in the vector env.
                          Each distribution is an array of probabilities for tasks in that environment.
        """
        from metaworld.wrappers import MultiTaskDROWrapper

        if isinstance(envs, gym.vector.SyncVectorEnv):
            # SyncVectorEnv has envs attribute
            for env_idx in range(len(envs.envs)):
                env = envs.envs[env_idx]
                # Navigate through wrappers to find or MultiTaskDROWrapper
                current = env
                while current is not None:
                    if isinstance(current, (MultiTaskDROWrapper)):
                        current.set_task_distribution(distribution)
                        break
                    # Move to next wrapper
                    if hasattr(current, 'env'):
                        current = current.env
                    else:
                        break
        else:
            # AsyncVectorEnv - use call method to set task distribution
            # Since AsyncVectorEnv doesn't expose envs attribute, we use call() method
            # The call() method will find set_task_distribution on MultiTaskDROWrapper in the wrapper chain
            envs.call('set_task_distribution', distribution)

    def reset_policy_optimizer(self):
        if hasattr(self, "policy") and isinstance(self.policy, TrainState):
            new_opt_state = self.policy.tx.init(self.policy.params)
            new_policy = self.policy.replace(opt_state=new_opt_state, step=0)
            return self.replace(policy=new_policy)
        return self

    def reset_value_optimizer(self):
        if hasattr(self, "value_function") and isinstance(self.value_function, TrainState):
            new_opt_state = self.value_function.tx.init(self.value_function.params)
            new_vf = self.value_function.replace(opt_state=new_opt_state, step=0)
            return self.replace(value_function=new_vf)
        return self

    @override
    def train(
        self,
        config: OnPolicyTrainingConfig,
        envs: GymVectorEnv,
        env_config: EnvConfig,
        run_timestamp: str | None = None,
        seed: int = 1,
        track: bool = True,
        checkpoint_manager: ocp.CheckpointManager | None = None,
        checkpoint_metadata: CheckpointMetadata | None = None,
        buffer_checkpoint: ReplayBufferCheckpoint | None = None,
        eval_env: GymVectorEnv = None,
        reset_optimizer_steps: int = 100_000,
    ) -> Self:
        global_episodic_return: Deque[float] = deque([], maxlen=20 * self.num_tasks)
        global_episodic_length: Deque[int] = deque([], maxlen=20 * self.num_tasks)
        # global_episodic_return: Deque[float] = deque([], maxlen=20 * self.num_tasks)

        success_history = [deque([], maxlen=50) for _ in range(self.num_tasks)]
        return_max = np.zeros(self.num_tasks)
        return_ref = np.ones(self.num_tasks) * 5000
        is_task_past_threshold = np.zeros(self.num_tasks)

        task_return_sum = np.zeros(self.num_tasks)
        task_success_any_step = np.zeros(self.num_tasks)
        task_attempts = np.zeros(self.num_tasks)

        dro_task_return_sum = np.zeros(self.num_tasks)
        dro_task_success_any_step = np.zeros(self.num_tasks)
        dro_task_attempts = np.zeros(self.num_tasks)
        dro_task_counts = np.zeros(self.num_tasks)
        dro_task_steps = np.zeros(self.num_tasks, dtype=int)

        dro_mean_return_per_task_prev = None

        task_success_already_found = np.zeros(self.num_tasks)

        task_abs_adv = None

        obs, _ = envs.reset()

        # metaworld_cls_to_task_name = {v.__name__: k for k, v in MT10_V3.items()}
        # task_names = [metaworld_cls_to_task_name[task_name] for task_name in envs.get_attr("task_name")]

        # env_list = envs.get_attr("env")
        # for env in env_list:
        #     print(env.tasks)
        # exit()

        if config.dro:
            if self.num_tasks == 10:
                task_names = list(MT10_V3.keys())
            elif self.num_tasks == 25:
                task_names = list(MT25_V3.keys())
            elif self.num_tasks == 50:
                task_names = list(MT50_V3.keys())
            elif self.num_tasks == 4:
                task_names = list(['PointMaze1', 'PointMaze2', 'PointMaze3', 'PointMaze4'])
            elif self.num_tasks == 3:
                task_names = list(['PointMaze/UMaze', 'PointMaze/Medium', 'PointMaze/Large'])
            else:
                raise NotImplementedError

            success_ref = np.ones(self.num_tasks)
            dist = np.ones(self.num_tasks)/self.num_tasks
            # dist[:] = 0
            # dist[12] = 1
            self.set_task_distributions(envs, dist)

            rng = np.random.default_rng(seed)

            smt_scheduler = SMTScheduler(
                num_tasks=self.num_tasks,
            )
            smt_steps_used = np.zeros(self.num_tasks, dtype=int)

        episode_started = np.ones((envs.num_envs,))
        start_step, episodes_ended = 0, 0

        if checkpoint_metadata is not None:
            start_step = checkpoint_metadata["step"]
            episodes_ended = checkpoint_metadata["episodes_ended"]

        rollout_buffer = self.spawn_rollout_buffer(env_config, config, seed)

        start_time = time.time()

        # Track update count for DRO updates
        update_count = 0
        eval_count = 0

        for global_step in range(start_step, config.total_steps // envs.num_envs):
            total_steps = global_step * envs.num_envs

            self, actions, aux_policy_outs = self.sample_action_and_aux(obs)
            next_obs, rewards, terminations, truncations, infos = envs.step(actions)

            rollout_buffer.add(
                obs,
                actions,
                rewards,
                episode_started,
                value=aux_policy_outs.get("value"),
                log_prob=aux_policy_outs.get("log_prob"),
                mean=aux_policy_outs.get("mean"),
                std=aux_policy_outs.get("std"),
            )

            episode_started = np.logical_or(terminations, truncations)
            obs = next_obs

            # Meta-World does not terminate on success during training, 
            # so we need to check *every* transition for success, not just the last one.
            # Be careful not to confuse index i with task_id.
            # task_success_already_found is indexed by i, not task_id. We reset it every episode.
            for i in range(self.num_tasks):
                if task_success_already_found[i] == 1:
                    continue

                if episode_started[i]:
                    # obs_ = infos["final_obs"] # episode ended, so fetch the actual final_obs
                    task_id = np.argmax(infos["final_obs"][i][-self.num_tasks:])
                    is_success = infos["final_info"]["success"][i]
                else:
                    # obs_ = obs # episode not ended, so we can just use obs
                    task_id = np.argmax(obs[i][-self.num_tasks:])
                    is_success = infos["success"][i]

                task_success_any_step[task_id] += is_success
                dro_task_success_any_step[task_id] += is_success

                if is_success:
                    task_success_already_found[i] = 1


            for i, env_ended in enumerate(episode_started):
                if env_ended:
                    global_episodic_return.append(
                        infos["final_info"]["episode"]["r"][i]
                    )
                    global_episodic_length.append(
                        infos["final_info"]["episode"]["l"][i]
                    )
                    episodes_ended += 1

                    task_id = np.argmax(infos["final_obs"][i][-self.num_tasks:])
                    task_attempts[task_id] += 1
                    dro_task_attempts[task_id] += 1
                    task_return_sum[task_id] += infos["final_info"]["episode"]["r"][i]
                    dro_task_return_sum[task_id] += infos["final_info"]["episode"]["r"][i]

                    # if task_success_already_found[i]:
                    #     return_ref[task_id] = max(
                    #         return_ref[task_id],
                    #         infos["final_info"]["episode"]["r"][i]
                    #     )

                    success_history[task_id].append(task_success_already_found[i])

                    if is_task_past_threshold[task_id] == 0 and len(success_history[task_id]) == 50 and np.mean(success_history[task_id]) > 0.5:
                        is_task_past_threshold[task_id] = 1

                    return_max[task_id] = max(
                        return_max[task_id],
                        infos["final_info"]["episode"]["r"][i]
                    )
                    if is_task_past_threshold[task_id] == 1:
                        return_ref[task_id] = return_max[task_id]

                    # end of episode, so reset our search for a success in the next trajectory
                    task_success_already_found[i] = 0

                    dro_task_steps[task_id] += int(infos["final_info"]["episode"]["l"][i])

            # if global_step % 500 == 0 and global_episodic_return:
            #     print(
            #         f"global_step={total_steps}, mean_episodic_return={np.mean(list(global_episodic_return))}"
            #     )
            #
            #     if track:
            #         log(
            #             {
            #                 "charts/mean_episodic_return": np.mean(
            #                     list(global_episodic_return)
            #                 ),
            #                 "charts/mean_episodic_length": np.mean(
            #                     list(global_episodic_length)
            #                 ),
            #             },
            #             step=total_steps,
            #         )

            if (global_step+1) % config.dro_rollout_steps == 0:
                # success rate should be zero for tasks we did not sample.
                dro_task_attempts[dro_task_attempts == 0] = 1
                dro_mean_success_per_task = dro_task_success_any_step / dro_task_attempts
                dro_mean_return_per_task = dro_task_return_sum / dro_task_attempts
                if dro_mean_return_per_task_prev is None:
                    dro_mean_return_per_task_prev = np.zeros(self.num_tasks)

                if config.task_sampling_algo == 'uniform':
                    gaps = np.zeros(self.num_tasks)

                elif config.task_sampling_algo == 'dro':
                    # gaps = success_ref - dro_mean_success_per_task
                    gaps = (return_ref - dro_mean_return_per_task) / return_ref
                elif config.task_sampling_algo == 'learning_progress':
                    slopes = np.abs(dro_mean_return_per_task - dro_mean_return_per_task_prev)
                    slopes /= (slopes.max() + 0.01)
                    gaps = slopes
                elif config.task_sampling_algo == 'learning_potential':
                    if task_abs_adv is None:
                        gaps = np.zeros(self.num_tasks)
                    else:
                        gaps = task_abs_adv #/ np.max(task_abs_adv)

                    print(gaps)

                if config.task_sampling_algo == 'smt':
                    dist = smt_scheduler.update(
                        steps_used=dro_task_steps,
                        eval_metrics=dro_mean_return_per_task,
                    )
                    self.set_task_distributions(envs, dist)
                else:
                    dist = self.update_task_weights(q=dist, gap=gaps, eta=config.dro_eta, step_size=config.dro_learning_rate, dro_eps=config.dro_eps)

                    if config.task_sampling_algo == 'learning_progress':
                        for task_id in range(self.num_tasks):
                            if len(success_history[task_id]) == 50 and np.mean(success_history[task_id]) > 0.9:
                                dist[task_id] = 0

                        if np.all(dist == 0):
                            dist = np.ones(self.num_tasks)/self.num_tasks
                        else:
                            dist = self.kl_project_with_floor(dist, config.dro_eps)

                # dist[:] = 0
                # dist[12] = 1
                self.set_task_distributions(envs, dist)

                if track:
                    dro_metrics = {}
                    for i, task_name in enumerate(task_names):
                        dro_metrics[f"dro_weight/{task_name}"] = dist[i]
                        dro_metrics[f"dro_ref/{task_name}"] = return_ref[i]

                        # if dro_task_attempts[i] > 0:
                        #     dro_metrics[f'dro/{task_name}_success_rate'] = dro_mean_success_per_task[i]
                        #     dro_metrics[f"dro/{task_name}_return"] = dro_mean_return_per_task[i]
                        # else:
                        #     dro_metrics[f'dro/{task_name}_success_rate'] = float("nan")
                        #     dro_metrics[f"dro/{task_name}_return"] = float("nan")
                        # dro_metrics[f"dro/{task_name}_frac"] = dro_task_frac[i]
                        # dro_metrics[f"dro/{task_name}_ref"] = success_ref[i]

                    log(dro_metrics, step=total_steps)

                dro_task_return_sum[:] = 0
                dro_task_success_any_step[:] = 0
                dro_task_attempts[:] = 0
                dro_task_counts[:] = 0
                dro_task_steps[:] = 0
                dro_mean_return_per_task_prev[:] = dro_mean_return_per_task

            # if reset_optimizer_steps > 0 and global_step % reset_optimizer_steps == 0 and global_step > 0:
            #     print('BEFORE RESET')
            #     print("First moment (mu):", self.policy.opt_state)
            #     self.reset_policy_optimizer()
            #     self.reset_value_optimizer()
            #     print('AFTER RESET')
            #     print("Second moment (nu):", self.policy.opt_state[0])

            # Logging
            if global_step % 10_000 == 0:
                sps_steps = (global_step - start_step) * envs.num_envs
                sps = int(sps_steps / (time.time() - start_time))
                print("SPS:", sps)

                if track:
                    log({"charts/SPS": sps}, step=total_steps)

            if rollout_buffer.ready:
                rollouts = rollout_buffer.get()
                self, logs = self.update(
                    rollouts,
                    dones=terminations,
                    next_obs=np.where(
                        episode_started[:, None], np.stack(infos["final_obs"]), next_obs
                    ),
                )
                rollout_buffer.reset()
                update_count += 1

                task_abs_adv = logs.pop("metrics/task_abs_adv", np.zeros(self.num_tasks))

                if track:
                    log(logs, step=total_steps)

                # success rate should be zero for tasks we did not sample.
                task_attempts[task_attempts == 0] = 1
                mean_success_per_task = task_success_any_step / task_attempts
                mean_return_per_task = task_return_sum / task_attempts

                print(f'{task_success_any_step=}')
                print(f'{mean_success_per_task=}')
                print(f'{task_attempts=}')

                # for i in range(self.num_tasks):
                #     print(success_history[i])
                #     print(f"{task_names[i]}: {len(success_history[i])} steps, {np.mean(success_history[i])} success rate")

                if track:
                    train_metrics = {}
                    for i, task_name in enumerate(task_names):
                        if task_attempts[i] > 0:
                            train_metrics[f'train_success_rate/{task_name}'] = mean_success_per_task[i]
                            train_metrics[f'train_return/{task_name}'] = mean_return_per_task[i]
                        else:
                            train_metrics[f'train_success_rate/{task_name}'] = float("nan")
                            train_metrics[f'train_return/{task_name}'] = float("nan")

                    log(train_metrics, step=total_steps)

                print(f"{'Task Name':<30} {'Success Rate':>12} {'Task Weight':>12}")
                print("-" * 60)
                for i in range(self.num_tasks):
                    print(f"{task_names[i]:<30} {mean_success_per_task[i]:>10.3f} {dist[i]:>10.3f}")
                print("=" * 60)

                task_return_sum[:] = 0
                task_success_any_step[:] = 0
                task_attempts[:] = 0

                # Evaluation
                if (
                    config.evaluation_frequency > 0
                    and episodes_ended % config.evaluation_frequency == 0
                    and episode_started.any()
                    and global_step > 0
                ):
                    eval_count += 1
                    if eval_env is None or config.dro==False:
                        mean_success_rate, mean_returns, mean_success_per_task = (
                            env_config.evaluate(envs, self)
                        )
                        eval_metrics = {
                                           "charts/mean_success_rate": float(mean_success_rate),
                                           "charts/mean_evaluation_return": float(mean_returns),
                                       } | {
                                           f"charts/{task_name}_success_rate": float(success_rate)
                                           for task_name, success_rate in mean_success_per_task.items()
                                       }
                        print(
                            f"total_steps={total_steps}, mean evaluation success rate: {mean_success_rate:.4f}"
                            + f" return: {mean_returns:.4f}"
                        )

                        if track:
                            log(eval_metrics, step=total_steps)

                        # Checkpointing
                        if checkpoint_manager is not None:
                            if not episode_started.all():
                                raise NotImplementedError(
                                    "Checkpointing currently doesn't work for the case where evaluation is run before all envs have finished their episodes / are about to be reset."
                                )

                            checkpoint_manager.save(
                                total_steps,
                                args=get_checkpoint_save_args(
                                    self,
                                    envs,
                                    global_step,
                                    episodes_ended,
                                    run_timestamp,
                                ),
                                metrics={
                                    k.removeprefix("charts/"): v
                                    for k, v in eval_metrics.items()
                                },
                            )

                        # Reset envs again to exit eval mode
                        _, _ = envs.reset()
                        episode_started = np.ones((envs.num_envs,))
                    else:
                        mean_success_rate, mean_returns, mean_success_per_task, mean_return_per_task = (
                            env_config.evaluate(eval_env, self)
                        )
                        eval_metrics = {
                            "charts/mean_success_rate": float(mean_success_rate),
                            "charts/mean_return": float(mean_returns),
                        } | {
                            f"charts/{task_name}_success_rate": float(success_rate)
                            for task_name, success_rate in mean_success_per_task.items()
                        } | {
                            f"charts/{task_name}_return": float(mean_return)
                            for task_name, mean_return in mean_return_per_task.items()
                        }
                        print(
                            f"total_steps={total_steps}, mean evaluation success rate: {mean_success_rate:.4f}"
                            + f" return: {mean_returns:.4f}"
                        )
                        # print(mean_success_per_task)
                        # print(mean_return_per_task)

                        if track:
                            log(eval_metrics, step=total_steps)

                        # Checkpointing
                        if checkpoint_manager is not None:
                            if not episode_started.all():
                                raise NotImplementedError(
                                    "Checkpointing currently doesn't work for the case where evaluation is run before all envs have finished their episodes / are about to be reset."
                                )

                            checkpoint_manager.save(
                                total_steps,
                                args=get_checkpoint_save_args(
                                    self,
                                    eval_env,
                                    global_step,
                                    episodes_ended,
                                    run_timestamp,
                                ),
                                metrics={
                                    k.removeprefix("charts/"): v
                                    for k, v in eval_metrics.items()
                                },
                            )

                        # Reset envs again to exit eval mode
                        # obs, _ = envs.reset()
                        eval_env.reset()
                        episode_started = np.ones((envs.num_envs,))

        return self
