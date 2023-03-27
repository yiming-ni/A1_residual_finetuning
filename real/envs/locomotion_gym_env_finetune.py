"""This file implements the locomotion gym env."""
import collections
import time
from multiprocessing import Process

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from collections import deque
from dataclasses import dataclass
from real.envs.env_wrappers.action_filter import ActionFilterButter
from real.interface.a1_interface import A1Interface
import roslibpy

import time

_NUM_SIMULATION_ITERATION_STEPS = 300
_LOG_BUFFER_LENGTH = 5000


@dataclass
class A1State:
    rot_quat: np.ndarray
    motor_pos: np.ndarray
    ball_loc: np.ndarray
    goal_loc: np.ndarray
    base_pos: np.ndarray


def get_dribble_reward(ball_xy, goal_xy):
    dist = (goal_xy[0] - ball_xy[0]) ** 2 + (goal_xy[1] - ball_xy[1]) ** 2
    dist_rew = np.exp(-0.5 * dist)
    return dist_rew


class LocomotionGymEnv(gym.Env):
    """The gym environment for the locomotion tasks."""

    def __init__(self,
                 episode_length,
                 real_ball,
                 real_goal,
                 recv_IP='127.0.0.1',
                 recv_port=8001,
                 send_IP='127.0.0.1',
                 send_port=8000,
                #  recv_IP="192.168.123.132",
                #  recv_port=32770,
                #  send_IP="192.168.123.12",
                #  send_port=32769
                 ):

        self.seed()
        self.ep_len = episode_length
        self.real_ball = real_ball
        self.real_goal = real_goal

        self.pGain = [80.0] * 12
        self.dGain = 12 * [1.0]
        self.pGain_noisy = np.copy(self.pGain)
        self.dGain_noisy = np.copy(self.dGain)

        self.a1 = A1Interface(recv_IP=recv_IP,
                              recv_port=recv_port,
                              send_IP=send_IP,
                              send_port=send_port)

        self.default_target_positions = [0.0, 0.90, -1.80, 0.0, 0.90, -1.80,
                                         0.0, 0.90, -1.80, 0.0, 0.90, -1.80]

        self.goal_pos = np.array([2.5, 0.5, 0])

        self.sim_freq = 1000
        self.exp_env_freq = 30
        self.h_freq = 1
        self.num_sims_per_planner_step = (self.sim_freq // self.h_freq) // self.exp_env_freq
        self.num_sims_per_env_step = self.sim_freq // self.exp_env_freq  # 2000 / 30 = 67
        self.num_sims_per_high_level = self.sim_freq // self.h_freq
        self.secs_per_env_step = self.num_sims_per_env_step / self.sim_freq  # 67 / 2000 = 0.033s
        self.real_env_freq = int(1 / self.secs_per_env_step)  # 33

        # observation
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(530,))  # <= 400 # 317

        # the low & high does not actually limit the actions output from MLP network, manually clip instead
        self.action_space = spaces.Box(low=-1, high=1, shape=(12,))
        self.history_len = 15
        self.previous_obs = deque(maxlen=self.history_len)
        self.previous_acs = deque(maxlen=self.history_len)

        # action filter
        self.laction_filter_order = 2
        self._init_action_filter()

        self._init_obs_a1_state()
        self.reset(use_pid=False)
        self.iters_so_far = 0
        self.update_iter(self.iters_so_far)

        if self.real_ball or self.real_goal:
            self.client = roslibpy.Ros(host='localhost', port=9090)
            self.client.run()

        if self.real_ball:
            self.ball_listener = roslibpy.Topic(self.client, '/global_ball_pos', 'std_msgs/Float32MultiArray')
            self.ball_listener.subscribe(self.ball_msg_callback)
        
        if self.real_goal:
            self.goal_listener = roslibpy.Topic(self.client, '/global_robot_pos', 'std_msgs/Float32MultiArray')
            self.goal_listener.subscribe(self.robot_pos_msg_callback)

    def close(self):
        if hasattr(self, '_robot') and self._robot:
            self._robot.Terminate()

    def seed(self, seed=None):
        self.np_random, self.np_random_seed = seeding.np_random(seed)
        return [self.np_random_seed]

    def ball_msg_callback(self, msg):
        self.obs_a1_state.ball_loc = np.array(msg['data'])

    def robot_pos_msg_callback(self, msg):
        self.obs_a1_state.base_pos = np.array(msg['data'][:3])
        # print("received msg: ", msg['data'])  # size=7

    ##########################################
    #            Init and Reset              #
    ##########################################

    def _init_obs_a1_state(self):
        rot_quat = np.zeros((4,))
        motor_pos = np.zeros((12,))
        ball_loc = np.array([3., 0., 0.1])
        goal_loc = np.array([4., 0., 0.])
        base_pos = np.array([0., 0., 0.27])
        self.obs_a1_state = A1State(ball_loc=ball_loc, goal_loc=goal_loc, rot_quat=rot_quat, base_pos=base_pos,
                                    motor_pos=motor_pos)
        return

    def _init_action_filter(self):
        self.laction_filter = ActionFilterButter(lowcut=None, highcut=[4], sampling_rate=self.real_env_freq,
                                                 order=self.laction_filter_order, num_joints=12)

    def reset(self, use_pid=False):
        self.time_in_sec = 0
        self.timestep = 0
        self.est_timestep = 0
        self.cycle_timestep = 0

        self.reward = None
        self.done = None
        self.info = {}
        self.delta_acs_err = 0.0

        if use_pid:
            self._reset_robot_pose()
            self._init_obs_a1_state()
            obs = self.a1.receive_observation()
            self.process_recv_package(obs)
        return

    def pid_ctrl(self):
        policy_count = 0
        # previous_time = time.time()
        while True:
            obs = self.a1.receive_observation()
            self.process_recv_package(obs)

            self.a1.send_command(np.concatenate(
                (self.default_target_positions, np.zeros((15,)), np.zeros((15,)), np.zeros((1,)))).squeeze())
            policy_count += 1
            # current_time = time.time()
            # print("Frequency: ", 1/(current_time - previous_time))
            # previous_time = current_time
            time.sleep(0.00005)

    def _reset_robot_pose(self):
        print("Start PID Control")
        proc = Process(target=self.pid_ctrl)
        proc.start()
        input("Press Enter to start training...")
        proc.terminate()
        return

    def step(self, action):
        previous_time = time.time()

        for li in range(self.num_sims_per_env_step):  # 30 Hz, num_sims_per_env_step = 67
            self.a1.send_command(action)
            obs = self.a1.receive_observation()
            self.process_recv_package(obs)
            time.sleep(0.00005)
            current_time = time.time()
            # print("Frequency: ", 1/(current_time - previous_time))
            previous_time = current_time

        self._update_env(step=True)
        reward = self.get_reward(self.obs_a1_state.ball_loc, self.obs_a1_state.goal_loc)
        done = self._termination()
        self.info['terminated'] = done
        self.info['max_torque'] = 0.0
        return reward, done, self.info

    def process_recv_package(self, obs):
        self.obs_a1_state.motor_pos = np.array(obs[7:19])
        self.obs_a1_state.rot_quat = np.array([obs[1], obs[2], obs[3], obs[0]])
        return

    def _update_env(self, step=True):
        if step:
            self.timestep += 1
            self.time_in_sec = (self.timestep * self.num_sims_per_env_step) / self.sim_freq  # self.sim.time
            self.cycle_timestep += 1

    def _termination(self):
        if self.timestep > self.ep_len:
            return True
        # TODO: add other conditions
        return False

    ##########################################
    #                Reward                  #
    ##########################################
    def get_reward(self, ball_pos, goal_pos):
        return get_dribble_reward(ball_xy=ball_pos[:2],
                                  goal_xy=goal_pos[:2])

    def update_iter(self, curr_iter):
        self.iters_so_far = curr_iter
