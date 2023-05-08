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

from ..wrappers.action_filter import ActionFilterButter
from sim.robots.robot_utils import compute_local_root_quat, compute_local_pos

is_dribble = True

if is_dribble:
    NUM_CURR_OBS = 18 + 3
    local_root_obs = True
    num_obs = 518
    # MODEL_PATH = '/home/zli/Documents/yni/A1_AMP/isaacgymenvs/runs/randurdf_randgoal_randr_cont/nn/randurdf_randgoal_randr_cont_70000.pth'
    # MODEL_PATH = '/home/zli/Documents/yni/A1_AMP/isaacgymenvs/runs/randurdf_fixgoal_randr_fov/nn/randurdf_fixgoal_randr_fov_50000.pth'
    MODEL_PATH = '/home/zli/Documents/yni/A1_AMP/isaacgymenvs/runs/randr_fov_small/nn/randr_fov_small_50000.pth'
    # MODEL_PATH = '/home/zli/Documents/yni/A1_AMP/isaacgymenvs/runs/threshold05/nn/threshold05_65000.pth'

else:
    NUM_CURR_OBS = 16
    local_root_obs = False
    num_obs = 436
    MODEL_PATH = '/home/yiming-ni/A1_dribbling/A1_AMP/isaacgymenvs/runs/gp7std3lr5e5fr033randprob9_10000.pth'

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
    # class AddPreviousActions(gym.ObservationWrapper):
    def __init__(self, env, residual_scale, fov, real_robot: bool = False, action_history: int = 15, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        # import ipdb; ipdb.set_trace()

        self.residual_scale = residual_scale
        self._pd_action_offset = np.array(
            [0.0000, 0.9000, -1.8000, 0.0000, 0.9000, -1.8000, 0.0000, 0.9000, -1.8000, 0.0000, 0.9000, -1.8000])
        self._pd_action_scale = np.array(
            [1.1239, 3.1416, 1.2526, 1.1239, 3.1416, 1.2526, 1.1239, 3.1416, 1.2526, 1.1239, 3.1416, 1.2526])
        self.default_target_positions = [0.0, 0.9, -1.80, 0.0, 0.9, -1.80,
                                         0.0, 0.90, -1.80, 0.0, 0.9, -1.80]
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

         # init fov
        self.is_fov = fov
        if self.is_fov:
            self.blind = 0.
            self.fov = np.zeros(4)
            self.fov[0] = 0.25
            self.fov[1] = 5.0
            self.fov[2] = 1.732
            self.fov[-1] = 0.3

            # init local_ball_pos_prev
            self.local_ball_pos_prev = torch.zeros((1, 3), dtype=torch.float)

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
        # self._action_filter.init_history(self.env.task_robot._INIT_QPOS)
        super().reset(*args, **kwargs)
        if self.is_fov:
            self.local_ball_pos_prev = torch.zeros((1, 3), dtype=torch.float)
        obs = self.reset_obs()
        # add ppo output (zero) to ppo obs
        return np.concatenate([obs, np.zeros(12)])

    def reset_obs(self):
        if is_dribble:
            return self.reset_dribble_obs()
        else:
            return self.reset_walk_obs()

    def make_obs(self):
        if is_dribble:
            return self.make_dribble_obs()
        else:
            return self.make_walk_obs()

    def _get_sim_obs_for_walk(self):
        sim_obs = self.env.env.observation_updater.get_observation()
        body_rotation = sim_obs['a1_description/body_rotation']
        joint_pos = sim_obs['a1_description/joints_pos']
        prev_actions = sim_obs['a1_description/prev_action']
        return body_rotation, joint_pos, prev_actions

    def _get_real_obs_for_walk(self):
        pass

    def make_walk_obs(self):
        if self.real_robot:
            body_rotation, joint_pos, prev_actions = self._get_real_obs_for_walk()
        else:
            body_rotation, joint_pos, prev_actions = self._get_sim_obs_for_walk()
        # import ipdb; ipdb.set_trace()
        curr_obs = np.concatenate([body_rotation, joint_pos])
        return np.concatenate(
            [np.array(self.prev_observations).flatten(), body_rotation, joint_pos, prev_actions]), curr_obs

    def _get_sim_obs_for_dribble(self):
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

        # ball_loc = np.array([2.0, 1, 0.1])
        # goal_loc = np.array([3, 0, 0])

        ball_loc = torch.from_numpy(ball_loc).unsqueeze(0)
        goal_loc = torch.from_numpy(goal_loc).unsqueeze(0)
        local_ball_obs = compute_local_pos(body_pos, ball_loc, body_rotation)
        # print('local_ball_pos: ', local_ball_obs)
        if self.is_fov:
            local_ball_obs = self._check_fov(local_ball_obs)
        else:
            local_ball_obs[:, 2] = ball_loc[:, 2]
        # local_ball_obs[:, 0] = 3
        # local_ball_obs[:, 1] = 0
        # local_ball_obs[:, 2] = 0.1

        local_goal_obs = compute_local_pos(body_pos, goal_loc, body_rotation)[:, :2]
        # print("robot pos: {}, ball pos: {}, goal pos: {}".format(body_pos, local_ball_obs, local_goal_obs))
        body_rotation = compute_local_root_quat(body_rotation)
        curr_obs = torch.cat((body_rotation, joint_pos, local_ball_obs), dim=-1)
        return curr_obs, local_goal_obs, prev_actions

    def _check_fov(self, local_ball_pos):
        outside = ((local_ball_pos[:, 0] < self.fov[0]) |
                    (local_ball_pos[:, 0] > self.fov[1]) |
                    (local_ball_pos[:, 1] > self.fov[2] * local_ball_pos[:, 0]) |
                    (local_ball_pos[:, 1] < - self.fov[2] * local_ball_pos[:, 0]) |
                    (local_ball_pos[:, 2] > self.fov[-1]))

        # import ipdb; ipdb.set_trace(cond=torch.any(outside))
        if local_ball_pos[:, 0] > self.fov[1]:
            print('problem')
        if outside[0] == 1.0:
            self.blind = 1.0
            local_ball_pos[:] = self.local_ball_pos_prev[:]
        else:
            self.blind = 0.0
            self.local_ball_pos_prev[:] = local_ball_pos[:]

        return local_ball_pos

    def _get_real_obs_for_dribble(self):
        # TODO: use method _robot.get... to copy over
        body_rotation = self.env.env._robot.GetBaseOrientation()
        body_pos = self.env.env._robot.GetBasePosition()
        joint_pos = self.env.env._robot.GetMotorAngles()
        # ball_loc = self.env.env._robot.GetBallPosition()  # TODO not implemented
        # goal_loc = self.env.env._robot.GetGoalPosition()  # TODO not implemented

        # hack 
        ball_loc = self.env.ball_loc
        goal_loc = self.env.goal_loc

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

    def make_dribble_obs(self):
        if self.real_robot:
            curr_obs, local_goal_obs = self._get_real_obs_for_dribble()
            # import ipdb; ipdb.set_trace()
            return torch.cat(list(self.prev_observations) + [curr_obs, local_goal_obs] + list(self.prev_actions),
                             dim=-1).cpu().numpy().flatten(), curr_obs
        else:
            curr_obs, local_goal_obs, prev_actions = self._get_sim_obs_for_dribble()
            return torch.cat(list(self.prev_observations) + [curr_obs, local_goal_obs, prev_actions],
                             dim=-1).cpu().numpy().flatten(), curr_obs

    def reset_walk_obs(self):
        if self.real_robot:
            body_rotation, joint_pos, prev_actions = self._get_real_obs_for_walk()
        else:
            body_rotation, joint_pos, prev_actions = self._get_sim_obs_for_walk()
        curr_obs = np.concatenate([body_rotation, joint_pos])
        for _ in range(self.prev_observations.maxlen):
            self.prev_observations.append(curr_obs)
        return np.concatenate(
            [np.array(self.prev_observations).flatten(), body_rotation, joint_pos, prev_actions]).flatten()

    def reset_dribble_obs(self):
        if self.real_robot:
            curr_obs, local_goal_obs = self._get_real_obs_for_dribble()
            default_pos = torch.from_numpy(self._pd_action_offset).unsqueeze(0)
            for _ in range(self.prev_actions.maxlen):
                self.prev_actions.append(default_pos)
            prev_actions = torch.cat(list(self.prev_actions), dim=-1)
        else:
            curr_obs, local_goal_obs, prev_actions = self._get_sim_obs_for_dribble()
        for _ in range(self.prev_observations.maxlen):
            self.prev_observations.append(curr_obs)
        prev_obs = torch.cat(list(self.prev_observations), dim=-1)
        # import ipdb; ipdb.set_trace()
        return torch.cat([prev_obs, curr_obs, local_goal_obs, prev_actions], dim=-1).cpu().numpy().flatten()

    def unnormalize_actions(self, actions):
        return self._pd_action_offset + self._pd_action_scale * actions

    def normalize_actions(self, actions):
        return actions / self._pd_action_scale - self._pd_action_offset

    def step(self, action):
        # get PPO action
        before_step_obs, curr_obs = self.make_obs()
        ppo_action = self.get_action(torch.from_numpy(before_step_obs.reshape(1, -1)).to(torch.float32))

        res_action = self._rescale_res(action)
        actual_action_normalized = np.clip(res_action + ppo_action, -1, 1)
        actual_action_unfiltered = self.unnormalize_actions(actual_action_normalized)
        actual_action = self._action_filter.filter(actual_action_unfiltered)

        ######## try ########
        # ppo_action_unnormalized = self.unnormalize_actions(ppo_action)
        # actual_action_unnormalized = np.clip(res_action + ppo_action_unnormalized, self.action_space.low, self.action_space.high)
        # actual_action_normalized = self.normalize_actions(actual_action_unnormalized)
        # actual_action = self._action_filter.filter(actual_action_unnormalized)
        ######################

        actual_action = np.clip(actual_action, self.action_space.low, self.action_space.high)
        actual_action = actual_action.astype(np.float32)
        if self.real_robot:
            self.prev_actions.append(torch.from_numpy(actual_action_normalized).unsqueeze(0))
        else:
            self.env.task_robot.update_actions(actual_action_normalized)
        self.prev_observations.append(curr_obs)

        obs, reward, done, info = self.env.step(actual_action)

        obs, _ = self.make_obs()
        obs = np.concatenate([obs.flatten(), ppo_action])
        if self.is_fov and self.blind == 1.0:
            reward = 0
        return obs, reward, done, info

    def _rescale_res(self, action):
        return action * self.residual_scale
