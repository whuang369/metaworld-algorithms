from __future__ import annotations

from collections import defaultdict

import gymnasium as gym
import numpy as np
import numpy.typing as npt

from metaworld_algorithms.types import Agent

def _get_pointmaze_task_names(
    eval_envs: gym.vector.SyncVectorEnv | gym.vector.AsyncVectorEnv,
) -> list[str]:
    """Derive stable task identifiers for the PointMaze benchmark."""
    try:
        task_names = eval_envs.get_attr("task_name")
        if task_names is not None:
            return task_names
    except Exception:
        pass

    try:
        task_ids = eval_envs.get_attr("task_id")
    except Exception:
        task_ids = [None] * eval_envs.num_envs

    names: list[str] = []
    for env_idx, task_id in enumerate(task_ids):
        if task_id is None:
            names.append(f"PointMaze{env_idx + 1}")
            continue
        task_array = np.asarray(task_id)
        if task_array.size == 0:
            names.append(f"PointMaze{env_idx + 1}")
            continue
        task_index = int(np.argmax(task_array))
        names.append(f"PointMaze{task_index + 1}")
    return names


def evaluation(
    agent: Agent,
    eval_envs: gym.vector.SyncVectorEnv | gym.vector.AsyncVectorEnv,
    num_episodes: int = 50,
) -> tuple[float, float, dict[str, float], dict[str, list[float]]]:
    terminate_on_success = bool(
        np.all(eval_envs.get_attr("terminate_on_success"))
    )
    eval_envs.call("toggle_terminate_on_success", True)

    obs: npt.NDArray[np.float64]
    obs, _ = eval_envs.reset()
    agent.reset(np.ones(eval_envs.num_envs, dtype=np.bool_))

    task_names = _get_pointmaze_task_names(eval_envs)
    unique_task_names = list(dict.fromkeys(task_names))
    successes = {task_name: 0 for task_name in unique_task_names}
    episodic_returns: dict[str, list[float]] = {
        task_name: [] for task_name in unique_task_names
    }

    def eval_done() -> bool:
        return all(len(r) >= num_episodes for r in episodic_returns.values())

    try:
        while not eval_done():
            actions = agent.eval_action(obs)
            obs, _, terminations, truncations, infos = eval_envs.step(actions)

            dones = np.logical_or(terminations, truncations)
            agent.reset(dones)

            final_info = infos.get("final_info")
            if final_info is None:
                continue

            for env_idx, env_ended in enumerate(dones):
                if not env_ended:
                    continue
                episodic_returns[task_names[env_idx]].append(
                    float(final_info["episode"]["r"][env_idx])
                )
                if len(episodic_returns[task_names[env_idx]]) <= num_episodes:
                    successes[task_names[env_idx]] += int(
                        final_info["success"][env_idx]
                    )
    finally:
        eval_envs.call("toggle_terminate_on_success", terminate_on_success)

    episodic_returns = {
        task_name: returns[:num_episodes]
        for task_name, returns in episodic_returns.items()
    }
    success_rate_per_task = {
        task_name: successes[task_name] / num_episodes
        for task_name in unique_task_names
    }
    mean_success_rate = (
        float(np.mean(list(success_rate_per_task.values())))
        if success_rate_per_task
        else 0.0
    )
    mean_returns = (
        float(np.mean(list(episodic_returns.values()))) if episodic_returns else 0.0
    )

    return (
        mean_success_rate,
        mean_returns,
        success_rate_per_task,
        episodic_returns,
    )
