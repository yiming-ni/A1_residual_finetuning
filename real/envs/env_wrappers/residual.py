from collections import deque

import gym
# import ipdb
import numpy as np
import torch
import pickle
# from multiprocessing import Process
from rl_games.algos_torch import network_builder
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.algos_torch.models import ModelA2CContinuousLogStd

from real.envs.env_wrappers.action_filter import ActionFilterButter
from robot_utils_real import compute_local_pos, compute_local_root_quat


NUM_CURR_OBS = 18 + 3
local_root_obs = True
num_obs = 518
# MODEL_PATH = '/home/yiming-ni/A1_dribbling/A1_AMP/isaacgymenvs/runs/threshold05_65000.pth'
MODEL_PATH = '/home/yiming-ni/A1_Dribbling/threshold05_65000.pth'

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


class ResidualWrapper(gym.Wrapper):
    def __init__(self, env, residual_scale, real_robot: bool = False, action_history: int = 15, *args, **kwargs):
        super().__init__(env, *args, **kwargs)

        self.curr_obs = None
        self.obs = None
        self.residual_scale = residual_scale
        self._pd_action_scale = np.array(
            [1.1239, 3.1416, 1.2526, 1.1239, 3.1416, 1.2526, 1.1239, 3.1416, 1.2526, 1.1239, 3.1416, 1.2526])
        self.default_target_positions = np.array([0.0, 0.9, -1.80, 0.0, 0.9, -1.80,
                                         0.0, 0.90, -1.80, 0.0, 0.9, -1.80])
        low = self.default_target_positions - self._pd_action_scale
        high = self.default_target_positions + self._pd_action_scale
        self.action_space = gym.spaces.Box(low, high)

        self.policy_freq = 33
        self.action_filter_order = 2
        self._action_filter = ActionFilterButter(lowcut=None, highcut=[4], sampling_rate=self.policy_freq,
                                                 order=self.action_filter_order, num_joints=NUM_MOTORS)
        self.init_model(MODEL_PATH)
        self.prev_observations = deque(maxlen=action_history)
        self.prev_actions = deque(maxlen=action_history)
        self.observation_space = gym.spaces.Box(-1., 1., shape=(num_obs + NUM_MOTORS,), dtype=np.float32)
        self.real_robot = real_robot

        # hack
        # self.m = 0
        # f = open('/home/yiming-ni/A1_Dribbling/A1-RL-Exp-MuJoCo/A1Env/igobs.pt', 'rb')
        # self.acs_gt = pickle.load(f)

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
        self._action_filter.init_history(self.default_target_positions)
        self.env.reset(use_pid=True)
        self.obs, self.curr_obs = self.reset_obs()
        # add ppo output (zero) to ppo obs
        return np.concatenate([self.obs, np.zeros(12)])

    def _get_real_obs(self):
        body_rotation = self.env.obs_a1_state.rot_quat
        body_pos = self.env.obs_a1_state.base_pos
        joint_pos = self.env.obs_a1_state.motor_pos
        ball_loc = self.env.obs_a1_state.ball_loc  # TODO not implemented
        # goal_loc = self.env.obs_a1_state.goal_loc  # TODO not implemented

        # hack
        ball_loc = np.array([0.6, 0, 0.1])
        goal_loc = np.array([1.2, 0, 0])

        body_rotation = torch.from_numpy(body_rotation).unsqueeze(0)
        body_pos = torch.from_numpy(body_pos).unsqueeze(0)
        joint_pos = torch.from_numpy(joint_pos).unsqueeze(0)
        ball_loc = torch.from_numpy(ball_loc).unsqueeze(0)
        goal_loc = torch.from_numpy(goal_loc).unsqueeze(0)
        local_ball_obs = compute_local_pos(body_pos, ball_loc, body_rotation)
        local_goal_obs = compute_local_pos(body_pos, goal_loc, body_rotation)[:, :2]
        body_rotation = compute_local_root_quat(body_rotation)
        curr_obs = torch.cat((body_rotation, joint_pos, local_ball_obs), dim=-1)
        return curr_obs, local_goal_obs

    def make_obs(self):
        curr_obs, local_goal_obs = self._get_real_obs()
        # import ipdb; ipdb.set_trace()
        return torch.cat(list(self.prev_observations) + [curr_obs, local_goal_obs] + list(self.prev_actions),
                         dim=-1).cpu().numpy().flatten(), curr_obs

    def reset_obs(self):
        curr_obs, local_goal_obs = self._get_real_obs()
        default_pos = torch.from_numpy(np.zeros(12)).unsqueeze(0)
        for _ in range(self.prev_actions.maxlen):
            self.prev_actions.append(default_pos)
        prev_actions = torch.cat(list(self.prev_actions), dim=-1)

        for _ in range(self.prev_observations.maxlen):
            self.prev_observations.append(curr_obs)
        prev_obs = torch.cat(list(self.prev_observations), dim=-1)
        # import ipdb; ipdb.set_trace()
        return torch.cat([prev_obs, curr_obs, local_goal_obs, prev_actions], dim=-1).cpu().numpy().flatten(), curr_obs

    def unnormalize_actions(self, actions):
        return self.default_target_positions + self._pd_action_scale * actions

    def normalize_actions(self, actions):
        return actions / self._pd_action_scale - self.default_target_positions

    def step(self, action):
        # get PPO action
        ppo_action = self.get_action(torch.from_numpy(self.obs.reshape(1, -1)).to(torch.float32))

        res_action = self._rescale_res(action)
        actual_action_normalized = np.clip(res_action + ppo_action, -1, 1)
        actual_action_unfiltered = self.unnormalize_actions(actual_action_normalized)
        actual_action = self._action_filter.filter(actual_action_unfiltered)
        # print('actual_action: ', actual_action)

        ######## try ########
        # ppo_action_unnormalized = self.unnormalize_actions(ppo_action)
        # actual_action_unnormalized = np.clip(res_action + ppo_action_unnormalized, self.action_space.low, self.action_space.high)
        # actual_action_normalized = self.normalize_actions(actual_action_unnormalized)
        # actual_action = self._action_filter.filter(actual_action_unnormalized)
        ######################

        actual_action = np.clip(actual_action, self.action_space.low, self.action_space.high)
        actual_action = actual_action.astype(np.float32)
        self.prev_actions.append(torch.from_numpy(actual_action_normalized).unsqueeze(0))
        self.prev_observations.append(self.curr_obs)

        reward, done, info = self.env.step(actual_action)

        self.obs, self.curr_obs = self.make_obs()
        obs = np.concatenate([self.obs.flatten(), ppo_action])
        return obs, reward, done, info

    def _rescale_res(self, action):
        return action * self.residual_scale
