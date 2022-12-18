from collections import deque
import time

import numpy as np
from dmcgym import DMCGYM
from dm_control import composer
from dm_control.composer.environment import ObservationPadding

import dm_env
import gym


class EnvironmentWrapper(composer.Environment):
    def __init__(self, task, time_limit=float('inf'), random_state=None,
                 n_sub_steps=None,
                 raise_exception_on_physics_error=True,
                 strip_singleton_obs_buffer_dim=False,
                 max_reset_attempts=1,
                 delayed_observation_padding=ObservationPadding.ZERO):
        super().__init__(task=task,
                         time_limit=time_limit,
                         random_state=random_state,
                         n_sub_steps=n_sub_steps,
                         raise_exception_on_physics_error=raise_exception_on_physics_error,
                         strip_singleton_obs_buffer_dim=strip_singleton_obs_buffer_dim,
                         max_reset_attempts=max_reset_attempts,
                         delayed_observation_padding=delayed_observation_padding)
    @property
    def observation_updater(self):
        return self._observation_updater


class DMCGYMWrapper(DMCGYM):
    def __int__(self, env: dm_env.Environment):
        super().__init__(env)

    @property
    def task_robot(self):
        return self._env.task.robot

    @property
    def observation_updater(self):
        return self._env.observation_updater


class ClipActionWrapper(gym.wrappers.ClipAction):
    def __init__(self, env):
        super().__init__(env)

    @property
    def action_space(self):
        return gym.spaces.Box(-1.0, 1.0, dtype=np.float32)


class RecordEpisodeStatisticsWrapper(gym.wrappers.RecordEpisodeStatistics):
    def __init__(self, env, deque_size):
        super().__init__(env, deque_size)
        self.episode_en_returns = None
        self.episode_dist_returns = None
        self.distance_return_queue = deque(maxlen=deque_size)
        self.energy_return_queue = deque(maxlen=deque_size)

    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        self.episode_dist_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_en_returns = np.zeros(self.num_envs, dtype=np.float32)
        return observations

    def step(self, action):
        observations, rewards, dones, infos = self.env.step(action)
        if rewards == 0:
            self.episode_returns += rewards
            self.episode_dist_returns += rewards
            self.episode_en_returns += rewards
        else:
            try:
                self.episode_returns += rewards['total_reward']
                self.episode_dist_returns += rewards['distance_reward']
                self.episode_en_returns += rewards['energy_reward']
            except:
                import ipdb; ipdb.set_trace()
        self.episode_lengths += 1
        if not self.is_vector_env:
            infos = [infos]
            dones = [dones]
        else:
            infos = list(infos)  # Convert infos to mutable type
        for i in range(len(dones)):
            if dones[i]:
                infos[i] = infos[i].copy()
                episode_return = self.episode_returns[i]
                episode_dist_return = self.episode_dist_returns[i]
                episode_en_return = self.episode_en_returns[i]
                episode_length = self.episode_lengths[i]
                episode_info = {
                    "r": episode_return,
                    "dr": episode_dist_return,
                    "er": episode_en_return,
                    "l": episode_length,
                    "t": round(time.perf_counter() - self.t0, 6),
                }
                infos[i]["episode"] = episode_info
                self.return_queue.append(episode_return)
                self.distance_return_queue.append(episode_dist_return)
                self.energy_return_queue.append(episode_en_return)
                self.length_queue.append(episode_length)
                self.episode_count += 1
                self.episode_returns[i] = 0
                self.episode_dist_returns[i] = 0
                self.episode_en_returns[i] = 0
                self.episode_lengths[i] = 0
        if self.is_vector_env:
            infos = tuple(infos)
        return (
            observations,
            rewards if rewards == 0 else rewards['total_reward'],
            dones if self.is_vector_env else dones[0],
            infos if self.is_vector_env else infos[0],
        )

