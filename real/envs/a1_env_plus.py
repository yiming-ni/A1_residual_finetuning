import numpy as np
from collections import deque

import torch

from real.interface.a1_interface import A1Interface
from robot_utils_real import compute_local_pos, compute_local_root_quat


ENERGY_SCALE = 0.01


def get_dribble_reward(ball_xy, goal_xy, torque, dof_vel, energy_w):
    dist = (goal_xy[0] - ball_xy[0]) ** 2 + (goal_xy[1] - ball_xy[1]) ** 2
    dist_rew = np.exp(-0.5 * dist)
    energy_sum = np.sum(np.square(torque * dof_vel))
    energy_reward = np.exp(- ENERGY_SCALE * energy_sum)
    total_reward = (1 - energy_w) * dist_rew + energy_w * energy_reward
    return total_reward


class A1RealPlus():
    def __init__(self, energy_weight: float,
                 receive_ip,
                 receive_port,
                 send_ip,
                 send_port,
                 zero_action: np.ndarray = np.asarray([0.05, 0.9, -1.8] * 4),
                 action_offset: np.ndarray = np.asarray([0.2, 0.4, 0.4] * 4)):
        self._robot = A1Interface(recv_IP=receive_ip, recv_port=receive_port, send_IP=send_ip, send_port=send_port)
        self.energy_weight = energy_weight
        self._prev_action = deque(maxlen=15)
        self._prev_obs = deque(maxlen=15)
        self._default_pos = zero_action
        self.timestep = 0.0
        self.est_timestep = 0.0
        self.time_in_sec = 0.0
        self.ppo_obs = None
        self._max_action = zero_action + action_offset
        self._min_action = zero_action - action_offset

    def _process_recv_obs(self, raw_obs):
        q = raw_obs[0:4]
        base_quat = np.copy(np.array(q[1], q[2], q[3], q[0]))
        motor_pos = np.copy(raw_obs[7:19])
        return base_quat, motor_pos

    def reset(self):
        # todo reset & init action filter
        self.timestep = 0.0
        self.est_timestep = 0
        self.time_in_sec = 0.0
        self.ppo_obs = self.get_observation(np.zeros(12))
        return np.concatenate([self.ppo_obs, np.zeros(12)])

    def step(self, action):
        ppo_action = self.get_action(torch.from_numpy(self.ppo_obs.reshape(1, -1)).to(torch.float32))

        res_action = self._rescale_res(action)
        actual_action_normalized = np.clip(res_action + ppo_action, -1, 1)
        actual_action_unfiltered = self.unnormalize_actions(actual_action_normalized)
        actual_action = self._action_filter.filter(actual_action_unfiltered)
        actual_action = np.clip(actual_action, self._min_action, self._max_action)

        for _ in range(30):
            self._robot.send_command(actual_action)
        self.ppo_obs = self.get_observation(action)
        reward = self.get_reward()
        done = False
        info = {}
        res_obs = np.concatenate([self.ppo_obs, ppo_action])
        return res_obs, reward, done, info

    def get_observation(self, stepped_action):
        raw_obs = self._robot.receive_observation()
        base_quat, motor_pos = self._process_recv_obs(raw_obs)
        body_pos = None
        ball_loc = None
        goal_loc = None
        base_quat = torch.from_numpy(base_quat).unsqueeze(0)
        body_pos = torch.from_numpy(body_pos).unsqueeze(0)
        ball_loc = torch.from_numpy(ball_loc).unsqueeze(0)
        goal_loc = torch.from_numpy(goal_loc).unsqueeze(0)
        temp_rot = compute_local_root_quat(base_quat)
        temp_rot = temp_rot.numpy().flatten()

        local_ball_pos = compute_local_pos(body_pos, ball_loc, base_quat)
        local_goal_pos = compute_local_pos(body_pos, goal_loc, base_quat)
        ob_curr = np.concatenate([temp_rot,
                                  local_ball_pos,
                                  motor_pos])
        if self.timestep == 0:
            [self._prev_obs.append(ob_curr) for _ in range(15)]
            [self._prev_action.append(np.zeros(12)) for _ in range(15)]
        self._prev_action.append(stepped_action)
        cur_obs = np.concatenate(
            [np.array(self._prev_obs).flatten(), ob_curr, local_goal_pos, np.array(self._prev_action).flatten()])
        self._prev_obs.append(cur_obs)
        return cur_obs

    def get_reward(self, ball_pos, goal_pos, torque, qvel):
        return get_dribble_reward(ball_xy=ball_pos[:2],
                                  goal_xy=goal_pos[:2],
                                  torque=torque,
                                  dof_vel=qvel,
                                  energy_w=self.energy_weight)