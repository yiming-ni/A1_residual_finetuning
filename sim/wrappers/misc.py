import numpy as np
from dmcgym import DMCGYM
from dm_control import composer
from dm_control.composer.environment import ObservationPadding

import dm_env
import gym
from gym import spaces


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

