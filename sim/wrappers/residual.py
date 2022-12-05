from collections import deque

import gym
import numpy as np
import torch
# from multiprocessing import Process
from rl_games.algos_torch import network_builder
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.algos_torch.models import ModelA2CContinuousLogStd
import copy

from dmcgym.env import dmc_obs2gym_obs

from ..wrappers.action_filter import ActionFilterButter
from sim.robots.robot_utils import compute_local_root_quat, compute_local_pos
import gym.spaces as spaces

# NUM_CURR_OBS = 16
NUM_CURR_OBS = 18 + 3
local_root_obs = True
# num_obs = 436
num_obs = 518
NUM_MOTORS = 12

PARAMS = {
    'name': 'amp',
    'separate': True,
    'space': {'continuous': {'mu_activation': 'None', 'sigma_activation': 'None',
                             'mu_init': {'name': 'random_uniform_initializer', 'a': -0.01, 'b': 0.01},
                             'sigma_init': {'name': 'const_initializer', 'val': -3},
                             'fixed_sigma': True, 'learn_sigma': False}},
    'mlp': {'units': [1024, 512], 'activation': 'relu', 'd2rl': False, 'initializer': {'name': 'default'},
            'regularizer': {'name': 'None'}},
    'disc': {'units': [1024, 512], 'activation': 'relu', 'initializer': {'name': 'default'}}
}

ARGS = {'actions_num': 12, 'input_shape': (num_obs,), 'num_seqs': 4096, 'value_size': 1, 'amp_input_shape': (245,)}

# MODEL_PATH = '/home/yiming-ni/A1_dribbling/A1_AMP/isaacgymenvs/runs/gp7std3lr5e5fr033randprob9_10000.pth'
MODEL_PATH = '/home/yiming-ni/A1_dribbling/A1_AMP/isaacgymenvs/runs/threshold05_65000.pth'


class ResidualWrapper(gym.Wrapper):
    # class AddPreviousActions(gym.ObservationWrapper):
    def __init__(self, env, action_history: int = 1, *args, **kwargs):
        # self.actions = deque(maxlen=action_history)
        super().__init__(env, *args, **kwargs)
        # import ipdb; ipdb.set_trace()


        self._pd_action_offset = np.array(
            [0.0000, 0.9000, -1.8000, 0.0000, 0.9000, -1.8000, 0.0000, 0.9000, -1.8000, 0.0000, 0.9000, -1.8000])
        self._pd_action_scale = np.array(
            [1.1239, 3.1416, 1.2526, 1.1239, 3.1416, 1.2526, 1.1239, 3.1416, 1.2526, 1.1239, 3.1416, 1.2526])
        self.default_target_positions = [0.0, 0.9, -1.80, 0.0, 0.9, -1.80,
                                         0.0, 0.90, -1.80, 0.0, 0.9, -1.80]
        low = self.default_target_positions - self._pd_action_scale
        high = self.default_target_positions + self._pd_action_scale
        self.action_space = gym.spaces.Box(low, high)

        self.policy_freq = 30
        self.action_filter_order = 2
        self._action_filter = ActionFilterButter(lowcut=None, highcut=[4], sampling_rate=self.policy_freq,
                                                order=self.action_filter_order, num_joints=NUM_MOTORS)
        self.init_model(MODEL_PATH)
        self.prev_observations = deque(maxlen=15)
        self.observation_space = gym.spaces.Box(-1., 1., shape=(num_obs,), dtype=np.float32)

    def _build_net(self, config):
        net = network_builder.A2CBuilder.Network(PARAMS, **config)
        self.model = ModelA2CContinuousLogStd.Network(net)
        self.model.to('cpu')
        self.model.eval()
        self.is_rnn = self.model.is_rnn()

        obs_shape = torch_ext.shape_whc_to_cwh((num_obs,))
        self.running_mean_std = RunningMeanStd(obs_shape).to('cpu')
        self.running_mean_std.eval()
        return

    def restore_checkpoint(self, fn):
        checkpoint = torch_ext.load_checkpoint(fn)
        statedict = {}
        for k in checkpoint['model'].keys():
            if '_disc' not in k:
                statedict[k] = checkpoint['model'][k]
        self.model.load_state_dict(statedict)

        self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])

    def get_action_from_numpy(self, obs):
        obs = torch.from_numpy(obs).unsqueeze(0)
        return self.get_action(obs)

    def get_action(self, obs):
        obs = self._preproc_obs(obs)
        input_dict = {
            'is_train': False,
            'prev_actions': None,
            'obs': obs,
            'rnn_states': None
        }
        with torch.no_grad():
            res_dict = self.model(input_dict)
        mu = res_dict['mus']
        # action = res_dict['actions']
        self.states = res_dict['rnn_states']
        current_action = mu

        return self.rescale_actions(torch.clamp(current_action, -1.0, 1.0))[0].detach().numpy()

    def _preproc_obs(self, obs):
        obs = self.running_mean_std(obs)
        return obs

    def rescale_actions(self, action):
        d = torch.ones((12,), dtype=torch.float, device='cpu')
        m = torch.zeros((12,), dtype=torch.float, device='cpu')
        scaled_action = action * d + m
        return scaled_action

    def init_model(self, model_path):
        self._build_net(ARGS)
        self.restore_checkpoint(model_path)

    def reset(self, *args, **kwargs):
        self._action_filter.reset()
        self._action_filter.init_history(self._pd_action_offset)
        obs = super().reset(*args, **kwargs)
        # import ipdb; ipdb.set_trace()
        # curr_obs = self.get_curr_obs_from_raw(obs)
        # for _ in range(15):
        #     self.env.task_robot.update_observations(curr_obs)
        # obs = self.env.env.observation_updater.get_observation()
        # obs = np.concatenate([obs, np.zeros(shape)])
        # return self.reset_walk_obs()
        return self.reset_dribble_obs()

    def make_walk_obs(self):
        body_rotation = self.env.env.observation_updater.get_observation()['a1_description/body_rotation']
        joint_pos = self.env.env.observation_updater.get_observation()['a1_description/joints_pos']
        prev_actions = self.env.env.observation_updater.get_observation()['a1_description/prev_action']
        # import ipdb; ipdb.set_trace()
        curr_obs = np.concatenate([body_rotation, joint_pos])
        return np.concatenate([np.array(self.prev_observations).flatten(), body_rotation, joint_pos, prev_actions]), curr_obs

    def make_dribble_obs(self):
        updater_obs = self.env.env.observation_updater.get_observation()
        body_rotation = updater_obs['a1_description/body_rotation']
        body_pos = updater_obs['a1_description/body_position']
        joint_pos = updater_obs['a1_description/joints_pos']
        prev_actions = updater_obs['a1_description/prev_action']
        ball_loc = updater_obs['ball_loc']
        goal_loc = updater_obs['goal_loc']
        body_rotation = torch.from_numpy(body_rotation).unsqueeze(0)
        body_pos = torch.from_numpy(body_pos).unsqueeze(0)
        joint_pos = torch.from_numpy(joint_pos).unsqueeze(0)
        prev_actions = torch.from_numpy(prev_actions).unsqueeze(0)
        ball_loc = torch.from_numpy(ball_loc).unsqueeze(0)
        goal_loc = torch.from_numpy(goal_loc).unsqueeze(0)
        local_ball_obs = compute_local_pos(body_pos, ball_loc, body_rotation)
        local_goal_obs = compute_local_pos(body_pos, goal_loc, body_rotation)[:, :2]

        # print('body pos: ', body_pos)
        print('ball pos: ', ball_loc)
        # print('ball obs: ', local_ball_obs)
        body_rotation = compute_local_root_quat(body_rotation)
        curr_obs = torch.cat((body_rotation, joint_pos, local_ball_obs), dim=-1)
        return torch.cat(list(self.prev_observations) + [curr_obs, local_goal_obs, prev_actions], dim=-1).cpu().numpy(), curr_obs

    def reset_walk_obs(self):
        body_rotation = self.env.env.observation_updater.get_observation()['a1_description/body_rotation']
        joint_pos = self.env.env.observation_updater.get_observation()['a1_description/joints_pos']
        prev_actions = self.env.env.observation_updater.get_observation()['a1_description/prev_action']
        curr_obs = np.concatenate([body_rotation, joint_pos])
        for _ in range(self.prev_observations.maxlen):
            self.prev_observations.append(curr_obs)
        return np.concatenate([np.array(self.prev_observations).flatten(), body_rotation, joint_pos, prev_actions])

    def reset_dribble_obs(self):
        body_rotation = self.env.env.observation_updater.get_observation()['a1_description/body_rotation']
        body_pos = self.env.env.observation_updater.get_observation()['a1_description/body_position']
        joint_pos = self.env.env.observation_updater.get_observation()['a1_description/joints_pos']
        prev_actions = self.env.env.observation_updater.get_observation()['a1_description/prev_action']
        ball_loc = self.env.env.observation_updater.get_observation()['ball_loc']
        goal_loc = self.env.env.observation_updater.get_observation()['goal_loc']
        # import ipdb; ipdb.set_trace()
        body_rotation = torch.from_numpy(body_rotation).unsqueeze(0)
        body_pos = torch.from_numpy(body_pos).unsqueeze(0)
        joint_pos = torch.from_numpy(joint_pos).unsqueeze(0)
        prev_actions = torch.from_numpy(prev_actions).unsqueeze(0)
        ball_loc = torch.from_numpy(ball_loc).unsqueeze(0)
        goal_loc = torch.from_numpy(goal_loc).unsqueeze(0)
        local_ball_obs = compute_local_pos(body_pos, ball_loc, body_rotation)
        local_goal_obs = compute_local_pos(body_pos, goal_loc, body_rotation)[:, :2]
        body_rotation = compute_local_root_quat(body_rotation)
        curr_obs = torch.cat((body_rotation, joint_pos, local_ball_obs), dim=-1)
        for _ in range(self.prev_observations.maxlen):
            self.prev_observations.append(curr_obs)
        prev_obs = torch.cat(list(self.prev_observations), dim=-1)
        # import ipdb; ipdb.set_trace()
        return torch.cat([prev_obs, curr_obs, local_goal_obs, prev_actions], dim=-1).cpu().numpy()


    def unnormalize_actions(self, actions):
        return self._pd_action_offset + self._pd_action_scale * actions

    def normalize_actions(self, actions):
        return actions / self._pd_action_scale - self._pd_action_offset

    def step(self, action):
        # get PPO action
        obs = self.observation()
        # before_step_obs, curr_obs = self.make_walk_obs()
        before_step_obs, curr_obs = self.make_dribble_obs()
        # ppo_action = self.get_action(torch.from_numpy(before_step_obs.reshape(1, -1)).to(torch.float32))
        ppo_action = action
        self.env.task_robot.update_actions(ppo_action)  # TODO how to update action after adding residual
        base_action = self.unnormalize_actions(ppo_action)
        base_action_filtered = self._action_filter.filter(base_action)
        #
        # res_action = self._rescalre(action)
        # actual_action = np.clip(res_action + base_action_filtered, joint limits)
        # actual_action = np.clip(base_action_filtered, self.action_space.low, self.action_space.high)
        actual_action = base_action_filtered
        # ls hack
        actual_action = actual_action.astype(np.float32)
        # print('before step: ', actual_action)

        # self.env.task_robot.update_observations(before_step_obs[15 * NUM_CURR_OBS:16 * NUM_CURR_OBS])
        self.prev_observations.append(curr_obs)
        obs, reward, done, info = self.env.step(actual_action)
        # import ipdb; ipdb.set_trace()
        # obs, reward, done, info = self.env.step(actual_action)
        # obs = np.concatenate([obs, ppo_action])
        # done=False
        # obs, _ = self.make_walk_obs()
        obs, _ = self.make_dribble_obs()
        return obs, reward, done, info

    def observation(self):
        """get observation from the task. This is the observation for base policy."""
        observation = self.env.env.observation_updater.get_observation()
        # import ipdb; ipdb.set_trace()
        observation = dmc_obs2gym_obs(observation)
        observation = self.env.observation(observation)
        # print('observation: ', observation)
        return observation

    def ppo_observation(self, raw_obs):
        raw_ball_obs = raw_obs[16*15:16*15+3]
        raw_goal_obs = raw_obs[16*16+3:16*16+6]
        prev_obs = raw_obs[:16*15]
        curr_obs = raw_obs[16*15+3:16*16+3]
        prev_acs = raw_obs[-12*15:]
        robot_quat = curr_obs[:4]
        robot_quat = torch.from_numpy(robot_quat).unsqueeze(0)
        raw_ball_obs = torch.from_numpy(raw_ball_obs).unsqueeze(0)
        raw_goal_obs = torch.from_numpy(raw_goal_obs).unsqueeze(0)
        local_ball_obs = compute_local_pos(robot_quat, raw_ball_obs)
        local_goal_obs = compute_local_pos(robot_quat, raw_goal_obs)
        robot_quat = compute_local_root_quat(robot_quat)
        refined_curr_obs = np.concatenate([robot_quat, curr_obs[4:], local_ball_obs])
        return np.concatenate([prev_obs, refined_curr_obs, local_goal_obs, prev_acs])

    def get_curr_obs_from_raw(self, obs):
        raw_ball_obs = obs[16 * 15:16 * 15 + 3]
        curr_obs = obs[16 * 15 + 3:16 * 16 + 3]
        robot_quat = curr_obs[:4]
        local_ball_obs = compute_local_pos(robot_quat, raw_ball_obs)
        robot_quat = compute_local_root_quat(robot_quat)
        refined_curr_obs = np.concatenate([robot_quat, curr_obs[4:], local_ball_obs])
        return refined_curr_obs



