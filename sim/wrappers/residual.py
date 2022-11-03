import gym
import numpy as np
import torch
# from multiprocessing import Process
from rl_games.algos_torch import network_builder
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.algos_torch.models import ModelA2CContinuousLogStd
import copy

from action_filter import ActionFilterButter

NUM_CURR_OBS = 18
local_root_obs = True
num_obs = 518
NUM_MOTORS = 12

PARAMS = {
    'name': 'amp',
    'separate': True,
    'space': {'continuous': {'mu_activation': 'None', 'sigma_activation': 'None',
                             'mu_init': {'name': 'random_uniform_initializer', 'a': -0.01, 'b': 0.01},
                             'sigma_init': {'name': 'const_initializer', 'val': -3},
                             'fixed_sigma': True, 'learn_sigma': False}
              },
    'mlp': {'units': [1024, 512], 'activation': 'relu', 'd2rl': False, 'initializer': {'name': 'default'},
            'regularizer': {'name': 'None'}},
    'disc': {'units': [1024, 512], 'activation': 'relu', 'initializer': {'name': 'default'}}
}

ARGS = {'actions_num': 12, 'input_shape': (num_obs,), 'num_seqs': 4096, 'value_size': 1, 'amp_input_shape': (245,)}

MODEL_PATH = '/home/yiming-ni/A1_dribbling/A1_AMP/isaacgymenvs/runs/drib_noise_goal_perturb_hlimit_delay_init/nn/drib_noise_goal_perturb_hlimit_delay_init.pth'


class ResidualWrapper(gym.Wrapper):
    # class AddPreviousActions(gym.ObservationWrapper):
    def __init__(self, env, action_history: int = 1, *args, **kwargs):
        # self.actions = deque(maxlen=action_history)
        super().__init__(env, *args, **kwargs)
        # assert isinstance(env.observation_space, gym.spaces.Dict)
        # assert 'actions' not in env.observation_space.spaces
        # new_obs = copy.copy(env.observation_space.spaces)
        # low = np.repeat(env.action_space.low, repeats=action_history, axis=0)
        # high = np.repeat(env.action_space.high, repeats=action_history, axis=0)
        # action_space = gym.spaces.Box(low, high)
        # new_obs['actions'] = action_space
        # self.observation_space = gym.spaces.Dict(new_obs)

        self._pd_action_offset = np.array(
            [0.0000, 0.9000, -1.8000, 0.0000, 0.9000, -1.8000, 0.0000, 0.9000, -1.8000, 0.0000, 0.9000, -1.8000])
        self._pd_action_scale = np.array(
            [1.1239, 3.1416, 1.2526, 1.1239, 3.1416, 1.2526, 1.1239, 3.1416, 1.2526, 1.1239, 3.1416, 1.2526])
        self.default_target_positions = [0.0, 0.9, -1.80, 0.0, 0.9, -1.80,
                                         0.0, 0.90, -1.80, 0.0, 0.9, -1.80]
        self.action_filter = ActionFilterButter(lowcut=None, highcut=[4], sampling_rate=self.policy_freq,
                                                order=self.action_filter_order, num_joints=NUM_MOTORS)
        self.init_model(MODEL_PATH)

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
        # action = res_dict['actions']
        self.states = res_dict['rnn_states']
        current_action = mu

        return self.rescale_actions(torch.clamp(current_action, -1.0, 1.0))

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
        self.action_filter.reset()
        # for _ in range(self.actions.maxlen):
        #     self.actions.append(np.zeros_like(self.action_space.low))
        return super().reset(*args, **kwargs)

    def step(self, action):
        # get PPO action
        ppo_action = self.get_action(self.env.observation()).detach().numpy()
        base_action = self._pd_action_offset + self._pd_action_scale * ppo_action
        base_action_filtered = self._action_filter.filter(base_action)
        #
        # res_action = self._rescalre(action)
        # actual_action = np.clip(res_action + base_action_filtered, joint limits)
        obs, reward, done, info = super().step(base_action_filtered)
        # self.actions.append(copy.deepcopy(action))
        return obs, reward, done, info

    # def observation(self, observation):
    #     observation = copy.copy(observation)
    #     observation['actions'] = np.concatenate(self.actions)
    #     return observation
    #
