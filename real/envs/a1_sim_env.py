import time

import numpy as np

import gym
import gym.spaces
import isaacgym
from absl import logging
# from dm_control.utils import rewards
from real import resetters
# from real.envs import env_builder
# from real.robots import a1, a1_robot, robot_config
# from real.utilities import pose3d
import os, sys
import yaml

# file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append('/home/yiming-ni/A1_dribbling/A1_AMP/isaacgymenvs')

from learning.amp_players import AMPPlayerContinuous
from tasks.a1_dribbling import A1Dribbling
from rl_games.common import vecenv
import torch

POLICY_DIR = '/home/yiming-ni/A1_dribbling/A1_AMP/isaacgymenvs/runs/drib_noise02_goal02_hlimit05_delay_init_vpen'

class A1IG(gym.Env):
    def __init__(
            self,
            zero_action: np.ndarray = np.asarray([0.05, 0.9, -1.8] * 4),
            action_offset: np.ndarray = np.asarray([0.2, 0.4, 0.4] * 4),
    ):
        logging.info(
            "WARNING: this code executes low-level control on the robot.")
        input("Press enter to continue...")

        # pass yaml config file
        with open(os.path.join(POLICY_DIR, 'config.yaml'), 'r') as f:
            cfg = yaml.load(f, Loader=yaml.SafeLoader)

        # modify num_envs, device, etc in cfg
        cfg['task']['env']['numEnvs'] = 1
        cfg['task']['env']['stateInit'] = 'Default'
        cfg['task']['sim']['physx']['use_gpu'] = False

        self.player = AMPPlayerContinuous(cfg)

        self.resetter = resetters.GetupResetter(self.env,
                                                True,
                                                standing_pose=zero_action)
        self.original_kps = self.env.robot._motor_kps.copy()
        self.original_kds = self.env.robot._motor_kds.copy()

        min_actions = zero_action - action_offset
        max_actions = zero_action + action_offset

        self.action_space = gym.spaces.Box(min_actions, max_actions)
        self._estimated_velocity = np.zeros(3)
        self._reset_var()

        self.obs_dict = self.player.env.obs_dict
        self.obs_buf = self.player.env.obs_buf

        self.observation_space = gym.spaces.Box(float("-inf"),
                                                float("inf"),
                                                shape=self.obs_buf.shape,
                                                dtype=np.float32)

    def _reset_var(self):
        self.prev_action = np.zeros_like(self.action_space.low)
        self.prev_qpos = None
        self._last_timestamp = time.time()
        self._prev_pose = None

    def reset(self):
        return self.player.env_reset(self.player.env)

    def _get_imu(self):
        rpy = self.env._robot.GetBaseRollPitchYaw()
        drpy = self.env._robot.GetBaseRollPitchYawRate()

        assert len(rpy) >= 3, rpy
        assert len(drpy) >= 3, drpy

        channels = ["R", "P", "dR", "dP", "dY"]
        observations = np.zeros(len(channels))
        for i, channel in enumerate(channels):
            if channel == "R":
                observations[i] = rpy[0]
            if channel == "Rcos":
                observations[i] = np.cos(rpy[0])
            if channel == "Rsin":
                observations[i] = np.sin(rpy[0])
            if channel == "P":
                observations[i] = rpy[1]
            if channel == "Pcos":
                observations[i] = np.cos(rpy[1])
            if channel == "Psin":
                observations[i] = np.sin(rpy[1])
            if channel == "Y":
                observations[i] = rpy[2]
            if channel == "Ycos":
                observations[i] = np.cos(rpy[2])
            if channel == "Ysin":
                observations[i] = np.sin(rpy[2])
            if channel == "dR":
                observations[i] = drpy[0]
            if channel == "dP":
                observations[i] = drpy[1]
            if channel == "dY":
                observations[i] = drpy[2]
        return observations

    def _compute_delta_time(self, current_time):
        delta_time_s = current_time - self._last_timestamp
        self._last_timestamp = current_time
        return delta_time_s

    def _update_vel(self, delta_time_s):
        self._estimated_velocity = self.env._robot.GetBaseVelocity()

    def observation(self, original_obs=None):
        if not original_obs:
            original_obs = self.player.env.obs_buf.detach().numpy()
        original_acs = self.player.env.actions.detach().numpy()

        return np.concatenate(
            [original_obs, original_acs]).astype(np.float32)

    def step(self, action):
        obs_dict = self.player.env.obs_dict
        ppo_acs = self.player.get_action(obs_dict)
        ppo_acs = ppo_acs.detach().numpy()
        real_acs = ppo_acs + action

        obs_dict, rew, done, info_dict = self.player.env.step(real_acs)
        original_obs = obs_dict['obs'].detach().numpy()
        rew = rew.detach().numpy()
        done = done.detach().numpy()

        self.prev_action[:] = action
        obs = self.observation(original_obs)

        return obs, rew, done, info_dict
