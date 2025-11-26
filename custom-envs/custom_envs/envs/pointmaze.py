from typing import Optional, Tuple

import gymnasium as gym
import numpy as np

def get_obs_info(obs, infos, task_id):
    next_obs = [obs['observation'][i] for i in range(4)] + [obs['achieved_goal'][i] for i in range(2)] + [obs['desired_goal'][i] for i in range(2)] + task_id
    next_obs = np.array(next_obs, dtype=np.float64)
    info_dict = {
        'is_success': infos['success'],
        'success': infos['success'],
        # 'final_obs': next_obs.copy(),
    }

    return next_obs, info_dict

def get_obs(obs, task_id):
    next_obs = [obs['observation'][i] for i in range(4)] + [obs['achieved_goal'][i] for i in range(2)] + [obs['desired_goal'][i] for i in range(2)] + task_id

    next_obs = np.array(next_obs)

    return next_obs

maps = [
    [[1, 1, 1, 1, 1],
    [1, "r", 0, "g", 1],
    [1, 1, 1, 1, 1]],

    [[1, 1, 1, 1, 1, 1, 1],
    [1, "r", 0, 0, 0, "g", 1],
    [1, 1, 1, 1, 1, 1, 1]],

    [[1, 1, 1, 1, 1],
    [1, "r", 0, 0, 1],
    [1, 1, 1, 0, 1],
    [1, "g", 0, 0, 1],
    [1, 1, 1, 1, 1]],

    [[1, 1, 1, 1, 1],
    [1, "r", "r", 0, 1],
    [1, 1, 1, 0, 1],
    [1, "g", 0, 0, 1],
    [1, 1, 1, 1, 1]],
]

n = 4

class PointMazeEnv1(gym.Env):
    def __init__(self):
        example_map = maps[0]

        self.env = gym.make('PointMaze_UMaze-v3', maze_map=example_map)
        self.action_space = self.env.action_space
        self.observation_space = gym.spaces.Box(np.array([-np.inf for _ in range(8)] + [0 for _ in range(n)]),
                                                np.array([np.inf for _ in range(8)] + [1 for _ in range(n)]),
                                                (n+8,),
                                                np.float64)
        self.task_id = [0 for _ in range(n)]
        self.task_id[0] = 1

        super().__init__()

    def step(self, a):

        obs, reward, terminations, truncations, infos = self.env.step(a)
        next_obs, infos = get_obs_info(obs, infos, self.task_id)

        return next_obs, reward, terminations, truncations, infos

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):

        obs, _ = self.env.reset()
        next_obs = get_obs(obs, self.task_id)

        return next_obs, {}


class PointMazeEnv2(gym.Env):
    def __init__(self):
        example_map = maps[1]

        self.env = gym.make('PointMaze_UMaze-v3', maze_map=example_map)
        self.action_space = self.env.action_space
        self.observation_space = gym.spaces.Box(np.array([-np.inf for _ in range(8)] + [0 for _ in range(n)]),
                                                np.array([np.inf for _ in range(8)] + [1 for _ in range(n)]),
                                                (n+8,),
                                                np.float64)
        self.task_id = [0 for _ in range(n)]
        self.task_id[1] = 1

        super().__init__()

    def step(self, a):

        obs, reward, terminations, truncations, infos = self.env.step(a)
        next_obs, infos = get_obs_info(obs, infos, self.task_id)

        return next_obs, reward, terminations, truncations, infos

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):

        obs, _ = self.env.reset()
        next_obs = get_obs(obs, self.task_id)

        return next_obs, {}


class PointMazeEnv3(gym.Env):
    def __init__(self): # batch_size=256, 372, 1024
        example_map = maps[2]

        self.env = gym.make('PointMaze_UMaze-v3', maze_map=example_map)
        self.action_space = self.env.action_space
        self.observation_space = gym.spaces.Box(np.array([-np.inf for _ in range(8)] + [0 for _ in range(n)]),
                                                np.array([np.inf for _ in range(8)] + [1 for _ in range(n)]),
                                                (n+8,),
                                                np.float64)
        self.task_id = [0 for _ in range(n)]
        self.task_id[2] = 1

        super().__init__()

    def step(self, a):

        obs, reward, terminations, truncations, infos = self.env.step(a)
        next_obs, infos = get_obs_info(obs, infos, self.task_id)

        return next_obs, reward, terminations, truncations, infos

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):

        obs, _ = self.env.reset()
        next_obs = get_obs(obs, self.task_id)

        return next_obs, {}


class PointMazeEnv4(gym.Env):
    def __init__(self):
        example_map = maps[3]

        self.env = gym.make('PointMaze_UMaze-v3', maze_map=example_map)
        self.action_space = self.env.action_space
        self.observation_space = gym.spaces.Box(np.array([-np.inf for _ in range(8)] + [0 for _ in range(n)]),
                                                np.array([np.inf for _ in range(8)] + [1 for _ in range(n)]),
                                                (n+8,),
                                                np.float64)
        self.task_id = [0 for _ in range(n)]
        self.task_id[3] = 1

        super().__init__()

    def step(self, a):

        obs, reward, terminations, truncations, infos = self.env.step(a)
        next_obs, infos = get_obs_info(obs, infos, self.task_id)

        return next_obs, reward, terminations, truncations, infos

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):

        obs, _ = self.env.reset()
        next_obs = get_obs(obs, self.task_id)

        return next_obs, {}

m = 3

class UMazeEnv(gym.Env):
    def __init__(self, reward_type: int = 0):
        if reward_type == 0:
            self.env = gym.make('PointMaze_UMazeDense-v3')
            print("Dense is created")
        else:
            self.env = gym.make('PointMaze_UMaze-v3')
            print("Sparse is created")
        self.action_space = self.env.action_space
        self.observation_space = gym.spaces.Box(np.array([-np.inf for _ in range(8)] + [0 for _ in range(m)]),
                                                np.array([np.inf for _ in range(8)] + [1 for _ in range(m)]),
                                                (m+8,),
                                                np.float64)
        self.task_id = [0 for _ in range(m)]
        self.task_id[0] = 1

        super().__init__()

    def step(self, a):

        obs, reward, terminations, truncations, infos = self.env.step(a)
        next_obs, infos = get_obs_info(obs, infos, self.task_id)

        return next_obs, reward, terminations, truncations, infos

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):

        obs, _ = self.env.reset()
        next_obs = get_obs(obs, self.task_id)

        return next_obs, {}

class MediumEnv(gym.Env):
    def __init__(self, reward_type: int = 0):
        if reward_type == 0:
            self.env = gym.make('PointMaze_MediumDense-v3')
        else:
            self.env = gym.make('PointMaze_Medium-v3')
        self.action_space = self.env.action_space
        self.observation_space = gym.spaces.Box(np.array([-np.inf for _ in range(8)] + [0 for _ in range(m)]),
                                                np.array([np.inf for _ in range(8)] + [1 for _ in range(m)]),
                                                (m+8,),
                                                np.float64)
        self.task_id = [0 for _ in range(m)]
        self.task_id[1] = 1

        super().__init__()

    def step(self, a):

        obs, reward, terminations, truncations, infos = self.env.step(a)
        next_obs, infos = get_obs_info(obs, infos, self.task_id)

        return next_obs, reward, terminations, truncations, infos

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):

        obs, _ = self.env.reset()
        next_obs = get_obs(obs, self.task_id)

        return next_obs, {}

class LargeEnv(gym.Env):
    def __init__(self, reward_type: int = 0):
        if reward_type == 0:
            self.env = gym.make('PointMaze_LargeDense-v3')
        else:
            self.env = gym.make('PointMaze_Large-v3')
        self.action_space = self.env.action_space
        self.observation_space = gym.spaces.Box(np.array([-np.inf for _ in range(8)] + [0 for _ in range(m)]),
                                                np.array([np.inf for _ in range(8)] + [1 for _ in range(m)]),
                                                (m+8,),
                                                np.float64)
        self.task_id = [0 for _ in range(m)]
        self.task_id[2] = 1

        super().__init__()

    def step(self, a):

        obs, reward, terminations, truncations, infos = self.env.step(a)
        next_obs, infos = get_obs_info(obs, infos, self.task_id)

        return next_obs, reward, terminations, truncations, infos

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):

        obs, _ = self.env.reset()
        next_obs = get_obs(obs, self.task_id)

        return next_obs, {}