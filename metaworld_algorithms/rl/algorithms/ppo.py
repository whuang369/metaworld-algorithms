from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from typing import Literal, Self, override

import chex
import distrax
import gymnasium as gym
import jax
import jax.numpy as jnp
import jax.flatten_util
import numpy as np
import numpy.typing as npt
from flax import struct
from flax.core import FrozenDict
from flax.training.train_state import TrainState
from jaxtyping import Array, Float, PRNGKeyArray, PyTree

from metaworld_algorithms.config.envs import EnvConfig
from metaworld_algorithms.config.networks import (
    ContinuousActionPolicyConfig,
    ValueFunctionConfig,
)
from metaworld_algorithms.config.rl import AlgorithmConfig
from metaworld_algorithms.monitoring.utils import (
    Histogram,
    get_logs,
    prefix_dict,
    pytree_histogram,
)
from metaworld_algorithms.rl.algorithms.utils import to_minibatch_iterator
from metaworld_algorithms.rl.networks import ContinuousActionPolicy, ValueFunction, Ensemble
from metaworld_algorithms.types import (
    Action,
    AuxPolicyOutputs,
    LogDict,
    LogProb,
    Observation,
    Rollout,
    Value,
)

from .utils import LinearFeatureBaseline, compute_gae, explained_variance
from .base import OnPolicyAlgorithm


@jax.jit
def _sample_action(
    policy: TrainState, observation: Observation, key: PRNGKeyArray
) -> tuple[Float[Array, "... action_dim"], PRNGKeyArray]:
    key, action_key = jax.random.split(key)
    dist: distrax.Distribution
    dist = policy.apply_fn(policy.params, observation)
    action = dist.sample(seed=action_key)
    return action, key


@jax.jit
def _get_value(value_function: TrainState, observation: Observation) -> Value:
    return value_function.apply_fn(value_function.params, observation)


@jax.jit
def _eval_action(
    policy: TrainState, observation: Observation
) -> Float[Array, "... action_dim"]:
    dist: distrax.Distribution
    dist = policy.apply_fn(policy.params, observation)
    return dist.mode()


@jax.jit
def _sample_action_dist_and_value(
    policy: TrainState,
    value_function: TrainState,
    observation: Observation,
    key: PRNGKeyArray,
) -> tuple[
    Action,
    LogProb,
    Action,
    Action,
    Value,
    PRNGKeyArray,
]:
    dist: distrax.Distribution
    key, action_key = jax.random.split(key)
    dist = policy.apply_fn(policy.params, observation)
    action, action_log_prob = dist.sample_and_log_prob(seed=action_key)
    value = value_function.apply_fn(value_function.params, observation)
    return action, action_log_prob, dist.mode(), dist.stddev(), value, key  # pyright: ignore[reportReturnType]


@jax.jit
def _sample_action_dist(
    policy: TrainState,
    observation: Observation,
    key: PRNGKeyArray,
) -> tuple[
    Action,
    LogProb,
    Action,
    Action,
    PRNGKeyArray,
]:
    dist: distrax.Distribution
    key, action_key = jax.random.split(key)
    dist = policy.apply_fn(policy.params, observation)
    action, action_log_prob = dist.sample_and_log_prob(seed=action_key)
    return action, action_log_prob, dist.mode(), dist.stddev(), key  # pyright: ignore[reportReturnType]


@dataclass(frozen=True)
class PPOConfig(AlgorithmConfig):
    policy_config: ContinuousActionPolicyConfig = ContinuousActionPolicyConfig()
    vf_config: ValueFunctionConfig | None = ValueFunctionConfig()
    clip_eps: float = 0.2
    baseline_type: Literal["linear", "mlp"] = "mlp"
    clip_vf_loss: bool = True
    entropy_coefficient: float = 5e-3
    vf_coefficient: float = 0.001
    normalize_advantages: bool = True
    gae_lambda: float = 0.97
    num_gradient_steps: int = 32
    num_epochs: int = 16
    target_kl: float | None = None

class PPO(OnPolicyAlgorithm[PPOConfig]):
    policy: TrainState
    value_function: TrainState | None
    value_function_ensemble: TrainState | None
    key: PRNGKeyArray
    gamma: float = struct.field(pytree_node=False)
    clip_eps: float = struct.field(pytree_node=False)
    baseline_type: Literal["linear", "mlp"] = struct.field(pytree_node=False)
    clip_vf_loss: bool = struct.field(pytree_node=False)
    entropy_coefficient: float = struct.field(pytree_node=False)
    vf_coefficient: float = struct.field(pytree_node=False)
    normalize_advantages: bool = struct.field(pytree_node=False)

    gae_lambda: float = struct.field(pytree_node=False)
    num_gradient_steps: int = struct.field(pytree_node=False)
    num_epochs: int = struct.field(pytree_node=False)
    target_kl: float | None = struct.field(pytree_node=False)
    # task_abs_adv: npt.NDArray | None = struct.field(pytree_node=False, default=None)

    @override
    @staticmethod
    def initialize(config: PPOConfig, env_config: EnvConfig, seed: int = 1) -> "PPO":
        assert isinstance(env_config.action_space, gym.spaces.Box), (
            "Non-box spaces currently not supported."
        )
        assert isinstance(env_config.observation_space, gym.spaces.Box), (
            "Non-box spaces currently not supported."
        )

        master_key = jax.random.PRNGKey(seed)
        algorithm_key, actor_init_key, vf_init_key, value_function_ensemble_init_key = jax.random.split(master_key, 4)
        dummy_obs = jnp.array(
            [env_config.observation_space.sample() for _ in range(config.num_tasks)]
        )

        policy_net = ContinuousActionPolicy(
            int(np.prod(env_config.action_space.shape)), config=config.policy_config
        )
        policy = TrainState.create(
            apply_fn=policy_net.apply,
            params=policy_net.init(actor_init_key, dummy_obs),
            tx=config.policy_config.network_config.optimizer.spawn(),
        )

        value_function = None
        if config.vf_config is not None:
            assert config.baseline_type == "mlp", (
                "MLP baseline must be specified if vf_config is provided"
            )
            vf_net = ValueFunction(config.vf_config)
            value_function = TrainState.create(
                apply_fn=vf_net.apply,
                params=vf_net.init(vf_init_key, dummy_obs),
                tx=config.vf_config.network_config.optimizer.spawn(),
            )
            vds = False
            value_function_ensemble = None
            if vds:
                # value_function_ensemble_init_key = jax.random.split(master_key, 2)

                value_function_cls = partial(ValueFunction, config=config.vf_config)
                # just 2 vf because we can use the main one as well, so 3 total.
                value_function_ensemble_net = Ensemble(value_function_cls, num=2)
                value_function_ensemble = TrainState.create(
                    apply_fn=value_function_ensemble_net.apply,
                    params=value_function_ensemble_net.init(value_function_ensemble_init_key, dummy_obs),
                    tx=config.vf_config.network_config.optimizer.spawn(),
                )

        return PPO(
            num_tasks=config.num_tasks,
            policy=policy,
            value_function=value_function,
            value_function_ensemble=value_function_ensemble,
            key=algorithm_key,
            gamma=config.gamma,
            clip_eps=config.clip_eps,
            baseline_type=config.baseline_type,
            clip_vf_loss=config.clip_vf_loss,
            entropy_coefficient=config.entropy_coefficient,
            vf_coefficient=config.vf_coefficient,
            normalize_advantages=config.normalize_advantages,
            gae_lambda=config.gae_lambda,
            num_gradient_steps=config.num_gradient_steps,
            num_epochs=config.num_epochs,
            target_kl=config.target_kl,
        )

    @override
    def get_num_params(self) -> dict[str, int]:
        ret = {
            "policy_num_params": sum(
                x.size for x in jax.tree.leaves(self.policy.params)
            ),
        }
        if self.baseline_type == "mlp":
            assert self.value_function is not None
            ret["vf_num_params"] = sum(
                x.size for x in jax.tree.leaves(self.value_function.params)
            )
        return ret

    @override
    def sample_action(self, observation: Observation) -> tuple[Self, Action]:
        action, key = _sample_action(self.policy, observation, self.key)
        return self.replace(key=key), jax.device_get(action)

    @override
    def sample_action_and_aux(
        self, observation: Observation
    ) -> tuple[Self, Action, AuxPolicyOutputs]:
        if self.baseline_type == "mlp":
            action, log_prob, mean, std, value, key = _sample_action_dist_and_value(
                self.policy, self.value_function, observation, self.key
            )
            action, log_prob, mean, std, value = jax.device_get(
                (action, log_prob, mean, std, value)
            )
            aux_outputs = {
                "log_prob": log_prob,
                "mean": mean,
                "std": std,
                "value": value,
            }
        else:
            action, log_prob, mean, std, key = _sample_action_dist(
                self.policy, observation, self.key
            )
            action, log_prob, mean, std = jax.device_get((action, log_prob, mean, std))
            aux_outputs = {
                "log_prob": log_prob,
                "mean": mean,
                "std": std,
            }
        return (
            self.replace(key=key),
            action,
            aux_outputs,
        )

    @override
    def eval_action(self, observations: Observation) -> Action:
        return jax.device_get(_eval_action(self.policy, observations))

    def segment_mean(self, x, idx, num_segments):
        # x: (N,), idx: (N,)
        sums = jax.ops.segment_sum(x, idx, num_segments)
        counts = jax.ops.segment_sum(jnp.ones_like(x), idx, num_segments)
        return sums / (counts + 1e-3)

    def segment_std(self, x, idx, num_segments):
        means = self.segment_mean(x, idx, num_segments)
        sq_means = self.segment_mean(x ** 2, idx, num_segments)
        var = sq_means - means ** 2
        return jnp.sqrt(jnp.maximum(var, 0.0)) + 1e-3

    def update_policy(self, data: Rollout) -> tuple[Self, LogDict]:
        assert data.advantages is not None

        # if self.normalize_advantages:
        #     advantages = (
        #         data.advantages - data.advantages.mean(axis=0, keepdims=True)
        #     ) / (data.advantages.std(axis=0, keepdims=True) + 1e-8)
        # else:
        #     advantages = data.advantages

        # PER-TASK ADVANTAGE NORMALIZATION
        # If returns are large for some tasks (easy tasks) but not others (hard tasks),
        # then global normalization will make all hard task have negative advantage,
        # which will dramatically slow learning on these tasks.

        if self.normalize_advantages:
            # data.advantages: (N,1)
            # data.observations: (N, obs_dim + num_tasks)
            adv = data.advantages[:, 0]  # (N,)
            obs = data.observations
            num_tasks = self.num_tasks

            # Extract per-sample task IDs (from appended one-hot)
            task_ids = jnp.argmax(obs[:, -num_tasks:], axis=-1)  # (N,)

            # Compute per-task means and stds
            means = self.segment_mean(adv, task_ids, num_tasks)  # (num_tasks,)
            stds = self.segment_std(adv, task_ids, num_tasks)  # (num_tasks,)

            # Broadcast task-wise stats to per-sample vectors
            adv_mean = means[task_ids]  # (N,)
            adv_std = stds[task_ids]  # (N,)

            # Normalize
            advantages = ((adv - adv_mean) / adv_std).reshape(-1, 1)

        else:
            advantages = data.advantages

        def policy_loss(params: FrozenDict) -> tuple[Float[Array, ""], LogDict]:
            action_dist: distrax.Distribution
            new_log_probs: Float[Array, " *batch"]
            assert data.log_probs is not None

            action_dist = self.policy.apply_fn(params, data.observations)
            new_log_probs = action_dist.log_prob(data.actions)  # pyright: ignore[reportAssignmentType]
            log_ratio = new_log_probs.reshape(data.log_probs.shape) - data.log_probs
            ratio = jnp.exp(log_ratio)

            # For logs
            approx_kl = jax.lax.stop_gradient(((ratio - 1) - log_ratio).mean())
            clip_fracs = jax.lax.stop_gradient(
                (jnp.abs(ratio - 1.0) > self.clip_eps).mean()
            )

            pg_loss1 = -advantages * ratio  # pyright: ignore[reportOptionalOperand]
            pg_loss2 = -advantages * jnp.clip(  # pyright: ignore[reportOptionalOperand]
                ratio, 1 - self.clip_eps, 1 + self.clip_eps
            )
            pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

            entropy_loss = action_dist.entropy().mean()

            return pg_loss - self.entropy_coefficient * entropy_loss, {
                "losses/entropy_loss": entropy_loss,
                "losses/policy_loss": pg_loss,
                "losses/approx_kl": approx_kl,
                "losses/clip_fracs": clip_fracs,
            }

        (_, logs), policy_grads = jax.value_and_grad(policy_loss, has_aux=True)(
            self.policy.params
        )
        # policy_grads_flat, _ = jax.flatten_util.ravel_pytree(policy_grads)
        # grads_hist_dict = prefix_dict(
        #     "nn/policy_grads", pytree_histogram(policy_grads["params"])
        # )

        policy = self.policy.apply_gradients(grads=policy_grads)
        # policy_params_flat, _ = jax.flatten_util.ravel_pytree(policy.params["params"])
        # param_hist_dict = prefix_dict(
        #     "nn/policy_params", pytree_histogram(policy.params["params"])
        # )

        return self.replace(policy=policy), logs

        # return self.replace(policy=policy), logs | {
        #     "nn/policy_grad_norm": jnp.linalg.norm(policy_grads_flat),
        #     "nn/policy_param_norm": jnp.linalg.norm(policy_params_flat),
        #     **grads_hist_dict,
        #     **param_hist_dict,
        # }

    def update_value_function(self, data: Rollout) -> tuple[Self, LogDict]:
        assert self.value_function is not None

        def value_function_loss(params: FrozenDict) -> tuple[Float[Array, ""], LogDict]:
            assert self.value_function is not None
            new_values: Float[Array, "*batch 1"]
            new_values = self.value_function.apply_fn(params, data.observations)
            chex.assert_equal_shape((new_values, data.returns))
            assert data.values is not None and data.returns is not None

            if self.clip_vf_loss:
                vf_loss_unclipped = (new_values - data.returns) ** 2
                v_clipped = data.values + jnp.clip(
                    new_values - data.values, -self.clip_eps, self.clip_eps
                )
                vf_loss_clipped = (v_clipped - data.returns) ** 2
                vf_loss = 0.5 * jnp.maximum(vf_loss_unclipped, vf_loss_clipped).mean()
            else:
                vf_loss = 0.5 * ((new_values - data.returns) ** 2).mean()

            return self.vf_coefficient * vf_loss, {
                "losses/value_function": vf_loss,
                "losses/values": new_values.mean(),
            }

        (_, logs), vf_grads = jax.value_and_grad(value_function_loss, has_aux=True)(
            self.value_function.params
        )
        # vf_grads_flat, _ = jax.flatten_util.ravel_pytree(vf_grads)
        # grads_hist_dict = prefix_dict(
        #     "nn/vf_grads", pytree_histogram(vf_grads["params"])
        # )

        value_function = self.value_function.apply_gradients(grads=vf_grads)
        # vf_params_flat, _ = jax.flatten_util.ravel_pytree(value_function.params)
        # param_hist_dict = prefix_dict(
        #     "nn/vf_params", pytree_histogram(value_function.params["params"])
        # )

        return self.replace(value_function=value_function), logs

        # return self.replace(value_function=value_function), logs | {
        #     "nn/vf_grad_norm": jnp.linalg.norm(vf_grads_flat),
        #     "nn/vf_param_norm": jnp.linalg.norm(vf_params_flat),
        #     **grads_hist_dict,
        #     **param_hist_dict,
        # }

    def update_value_function_ensemble(self, data: Rollout) -> tuple[Self, LogDict]:
        assert self.value_function_ensemble is not None
        assert data.returns is not None and data.values is not None

        def value_function_ensemble_loss(
            params: FrozenDict,
        ) -> tuple[Float[Array, ""], LogDict]:
            # new_values_ens: (K, B, 1) if K = ensemble_size
            new_values_ens: Float[Array, "... 1"]
            new_values_ens = self.value_function_ensemble.apply_fn(
                params, data.observations
            )

            # Broadcast returns / old values to ensemble shape.
            # data.returns: (B, 1) → (K, B, 1)
            returns = data.returns
            values_old = data.values
            # Broadcasting happens automatically in JAX ops below.

            if self.clip_vf_loss:
                vf_loss_unclipped = (new_values_ens - returns) ** 2
                v_clipped = values_old + jnp.clip(
                    new_values_ens - values_old, -self.clip_eps, self.clip_eps
                )
                vf_loss_clipped = (v_clipped - returns) ** 2
                vf_loss = 0.5 * jnp.maximum(
                    vf_loss_unclipped, vf_loss_clipped
                ).mean()
            else:
                vf_loss = 0.5 * ((new_values_ens - returns) ** 2).mean()

            return self.vf_coefficient * vf_loss, {
                "losses/value_function_ensemble": vf_loss,
                "losses/values_ensemble": new_values_ens.mean(),
            }

        (_, logs), vf_grads = jax.value_and_grad(
            value_function_ensemble_loss, has_aux=True
        )(self.value_function_ensemble.params)

        # vf_grads_flat, _ = jax.flatten_util.ravel_pytree(vf_grads)
        # grads_hist_dict = prefix_dict(
        #     "nn/vf_ens_grads", pytree_histogram(vf_grads["params"])
        # )

        value_function_ensemble = self.value_function_ensemble.apply_gradients(
            grads=vf_grads
        )
        # vf_params_flat, _ = jax.flatten_util.ravel_pytree(
        #     value_function_ensemble.params
        # )
        # param_hist_dict = prefix_dict(
        #     "nn/vf_ens_params",
        #     pytree_histogram(value_function_ensemble.params["params"]),
        # )

        return self.replace(value_function_ensemble=value_function_ensemble), logs

        # return self.replace(value_function_ensemble=value_function_ensemble), logs | {
        #     "nn/vf_ens_grad_norm": jnp.linalg.norm(vf_grads_flat),
        #     "nn/vf_ens_param_norm": jnp.linalg.norm(vf_params_flat),
        #     **grads_hist_dict,
        #     **param_hist_dict,
        # }

    @jax.jit
    def _get_activations(
        self, data: Rollout
    ) -> tuple[PyTree[Array], PyTree[Array] | None]:
        _, policy_state = self.policy.apply_fn(
            self.policy.params, data.observations, mutable=["intermediates"]
        )
        policy_acts = policy_state["intermediates"]

        vf_acts = None
        if self.baseline_type == "mlp":
            assert self.value_function is not None
            _, vf_state = self.value_function.apply_fn(
                self.value_function.params, data.observations, mutable=["intermediates"]
            )
            vf_acts = vf_state["intermediates"]

        return policy_acts, vf_acts

    @jax.jit
    def _update_inner(self, data: Rollout) -> tuple[Self, LogDict]:
        #
        # task_ids = data.observations[..., -self.num_tasks :]
        #
        # if self.use_task_weights:
        #     task_weights = extract_task_weights(self.alpha.params, task_ids)
        # else:
        #     task_weights = None
        #
        # actor_data = critic_data = data
        # actor_alpha_vals = critic_alpha_vals = alpha_vals
        # actor_task_weights = critic_task_weights = task_weights
        # alpha_val_indices = None
        #
        # if self.split_critic_losses or self.split_actor_losses:
        #     split_data, _ = self.split_data_by_tasks(data, task_ids)
        #     split_alpha_vals, alpha_val_indices = self.split_data_by_tasks(
        #         alpha_vals, task_ids
        #     )
        #     split_task_weights, _ = (
        #         self.split_data_by_tasks(task_weights, task_ids)
        #         if task_weights is not None
        #         else (None, None)
        #     )
        #
        #     if self.split_critic_losses:
        #         critic_data = split_data
        #         critic_alpha_vals = split_alpha_vals
        #         critic_task_weights = split_task_weights
        #
        #     if self.split_actor_losses:
        #         actor_data = split_data
        #         actor_alpha_vals = split_alpha_vals
        #         actor_task_weights = split_task_weights
        #
        # self, critic_logs = self.update_critic(
        #     critic_data, critic_alpha_vals, critic_task_weights
        # )
        # self, log_probs, actor_logs = self.update_actor(
        #     actor_data, actor_alpha_vals, actor_task_weights
        # )


        self, policy_logs = self.update_policy(data)

        vf_logs = {}
        if self.baseline_type == "mlp":
            self, vf_logs = self.update_value_function(data)

            # if self.value_function_ensemble is not None:
            #     self, vf_ens_logs = self.update_value_function_ensemble(data)
            #     vf_logs = vf_logs | vf_ens_logs

        return self, policy_logs | vf_logs

    @override
    def update(
        self,
        data: Rollout,
        dones: Float[npt.NDArray, "task 1"],
        next_obs: Float[Observation, " task"] | None = None,
    ) -> tuple[Self, LogDict]:
        last_values = None
        if next_obs is not None and self.baseline_type == "mlp":
            last_values = _get_value(self.value_function, next_obs)

        if self.baseline_type == "linear":
            values, returns = LinearFeatureBaseline.get_baseline_values_and_returns(
                data, self.gamma
            )
            data = data._replace(values=values, returns=returns)
            dones = np.ones(data.rewards.shape[1:], dtype=data.rewards.dtype)
        data = compute_gae(data, self.gamma, self.gae_lambda, last_values, dones)

        # Compute per-task mean absolute normalized advantage on full batch
        task_ids = np.argmax(data.observations.reshape(-1, data.observations.shape[-1])[:, -self.num_tasks:], axis=-1)
        adv = data.advantages.reshape(-1)
        means = np.bincount(task_ids, weights=adv, minlength=self.num_tasks) / np.bincount(task_ids,
                                                                                           minlength=self.num_tasks).clip(
            min=1)
        stds = np.sqrt(
            np.bincount(task_ids, weights=(adv - means[task_ids]) ** 2, minlength=self.num_tasks) / np.bincount(
                task_ids, minlength=self.num_tasks).clip(min=1))
        adv_normalized = (adv - means[task_ids]) / (stds[task_ids] + 1e-8)
        task_abs_adv = np.bincount(task_ids, weights=np.abs(adv_normalized), minlength=self.num_tasks) / np.bincount(
            task_ids, minlength=self.num_tasks).clip(min=1)

        assert data.advantages is not None and data.returns is not None
        assert data.values is not None and data.stds is not None
        assert data.means is not None and data.log_probs is not None
        # diagnostic_logs = prefix_dict(
        #     "data",
        #     {
        #         **get_logs("advantages", data.advantages),
        #         **get_logs("returns", data.returns),
        #         **get_logs("values", data.values),
        #         **get_logs("rewards", data.rewards),
        #         **get_logs(
        #             "num_episodes", data.dones.sum(axis=1), hist=False, std=False
        #         ),
        #         "action_std": Histogram(data.stds.reshape(-1)),
        #         "action_mean": Histogram(data.means.reshape(-1)),
        #         "approx_entropy": np.mean(-data.log_probs),
        #     },
        # )

        key, minibatch_iterator_key = jax.random.split(self.key)
        self = self.replace(key=key)
        seed = jax.random.randint(
            minibatch_iterator_key, (), minval=0, maxval=jnp.iinfo(jnp.int32).max
        ).item()
        minibatch_iterator = to_minibatch_iterator(
            data, self.num_gradient_steps, int(seed)
        )

        update_logs = defaultdict(list)
        keep_training = True
        for epoch in range(self.num_epochs):
            for step in range(self.num_gradient_steps):
                minibatch_rollout = next(minibatch_iterator)
                self, logs = self._update_inner(minibatch_rollout)
                for k, v in logs.items():
                    update_logs[k].append(float(jax.device_get(v)))

                if epoch == 0 and step == 0:  # Initial KL and Loss
                    update_logs["metrics/kl_before"] = [logs["losses/approx_kl"]]
                    update_logs["metrics/policy_loss_before"] = [
                        logs["losses/policy_loss"]
                    ]

                    if "losses/value_function" in logs:
                        update_logs["metrics/vf_loss_before"] = [
                            logs["losses/value_function"]
                        ]

                if self.target_kl and logs["losses/approx_kl"] > 1.5 * self.target_kl:
                    print(
                        f"Stopped early at KL {logs['losses/approx_kl']}, (epoch: {epoch}, steps: {step})"
                    )
                    keep_training = False
                    break

            if not keep_training:
                break

        # Finalize logs
        final_logs: dict = {
            "metrics/explained_variance": explained_variance(
                data.values.reshape(-1), data.returns.reshape(-1)
            )
        }
        for k, v in update_logs.items():
            if not isinstance(v[0], Histogram):
                final_logs[k] = np.mean(v)
            else:
                # TODO: should probably not be just the last histogram
                final_logs[k] = v[-1]

        # log activations
        # policy_acts, vf_acts = self._get_activations(next(minibatch_iterator))
        # final_logs.update(prefix_dict("nn/activations", pytree_histogram(policy_acts)))
        # if vf_acts is not None:
        #     final_logs.update(pytree_histogram(vf_acts))

        # return self, diagnostic_logs | final_logs
        return self, final_logs | {"metrics/task_abs_adv": task_abs_adv}
