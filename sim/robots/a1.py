import os
from collections import deque
from functools import cached_property
from typing import Optional

import numpy as np
from dm_control import composer, mjcf
from dm_control.composer.observation import observable
from dm_control.locomotion.walkers import base
from dm_control.utils.transformations import quat_to_euler
from dm_env import specs
from sim.wrappers.residual import NUM_CURR_OBS

import torch

from ..wrappers.residual import ResidualWrapper
from ..robots.robot_utils import compute_local_root_quat, compute_local_pos

ASSETS_DIR = os.path.join(os.path.dirname(__file__), 'a1')
_A1_XML_PATH = os.path.join(ASSETS_DIR, 'xml', 'a1.xml')
# _BALL_XML_PATH = os.path.join(ASSETS_DIR, 'xml', 'ball.xml')


class A1Observables(base.WalkerObservables):

    @composer.observable
    def joints_vel(self):
        return observable.MJCFFeature('qvel', self._entity.observable_joints)


    @composer.observable
    def a_prev_observation(self):
        return observable.Generic(lambda _: self._entity.prev_observation)

    @property
    def proprioception(self):
        return ([self.joints_pos, self.joints_vel] +
                self._collect_from_attachments('proprioception'))

    # @composer.observable
    # def b_robot_observation(self):
    #     return observable.Generic(
    #         lambda physics: self._entity.get_robot_observation(physics)
    #     )

    @composer.observable
    def curr_observation(self):
        return observable.Generic(
            lambda physics: self._entity.get_curr_observation(physics)
        )

    @composer.observable
    def goal_obs(self):
        return observable.Generic(lambda _: self._entity.goal_obs)

    @composer.observable
    def prev_action(self):
        return observable.Generic(lambda _: self._entity.prev_action)

    @composer.observable
    def sensors_velocimeter(self):
        return observable.Generic(
            lambda physics: self._entity.get_velocity(physics))

    @property
    def kinematic_sensors(self):
        return ([
            self.sensors_gyro, self.sensors_velocimeter, self.sensors_framequat
        ] + self._collect_from_attachments('kinematic_sensors'))

    @composer.observable
    def body_position(self):
        return observable.MJCFFeature('xpos', self._entity.root_body)

    @composer.observable
    def body_rotation(self):
        return observable.MJCFFeature('xquat', self._entity.root_body)[[1, 2, 3, 0]]


class A1(base.Walker):
    _INIT_QPOS = np.asarray([0.0, 0.9, -1.8] * 4)
    _QPOS_OFFSET = np.asarray([1.1239, 3.1416, 1.2526] * 4)
    _INIT_QUAT = np.asarray([0.0, 0.0, 0.0, 1.0])
    # _INIT_QUAT = compute_local_root_quat(torch.from_numpy(_INIT_QUAT.reshape(1, -1))).numpy().flatten()
    _INIT_OBS = np.concatenate([_INIT_QUAT, _INIT_QPOS])

    """A composer entity representing a Jaco arm."""

    def _build(self,
               name: Optional[str] = None,
               action_history: int = 1,
               learn_kd: bool = False):
        """Initializes the JacoArm.

    Args:
      name: String, the name of this robot. Used as a prefix in the MJCF name
        name attributes.
    """
        self._mjcf_root = mjcf.from_path(_A1_XML_PATH)
        if name:
            self._mjcf_root.model = name
        self._root_body = self._mjcf_root.find('body', 'trunk')
        self._root_body.pos[-1] = 0.125

        self._joints = self._mjcf_root.find_all('joint')
        # self._joints = self._root_body.find_all('joint')[1:]

        self._actuators = self.mjcf_model.find_all('actuator')
        # self._actuators = self._root_body.find_all('actuator')

        # Check that joints and actuators match each other.
        assert len(self._joints) == len(self._actuators)
        for joint, actuator in zip(self._joints, self._actuators):
            assert joint == actuator.joint

        self.kp = 80.0
        if learn_kd:
            self.kd = None
        else:
            self.kd = 1.0

        self._prev_actions = deque(maxlen=action_history)
        self._prev_observations = deque(maxlen=action_history)
        # self._goal_obs = np.array([-0.31705597, 0.91705686])
        self.initialize_episode_mjcf(None)

    def get_random_ball_pos(self):
        rand_dist = np.random.uniform(0, 3, 1)
        rand_angle = np.random.uniform(0, 2 * np.pi, 1)
        return (rand_dist * np.cos(rand_angle)).item(), (rand_dist * np.sin(rand_angle)).item()

    def initialize_episode_mjcf(self, random_state):
        self._prev_actions.clear()
        self._prev_observations.clear()
        init_action = np.zeros_like(self._INIT_QPOS)
        # init_obs = np.concatenate([self._INIT_OBS, np.array([1, 1, 0.05])])
        for _ in range(self._prev_actions.maxlen):
            # self._prev_actions.append(self._INIT_QPOS)
            self._prev_actions.append(init_action)
            self._prev_observations.append(self._INIT_OBS)


    @cached_property
    def action_spec(self):
        pd_action_offset = np.array(
            [0.0000, 0.9000, -1.8000, 0.0000, 0.9000, -1.8000, 0.0000, 0.9000, -1.8000, 0.0000, 0.9000, -1.8000])
        pd_action_scale = np.array(
            [1.1239, 3.1416, 1.2526, 1.1239, 3.1416, 1.2526, 1.1239, 3.1416, 1.2526, 1.1239, 3.1416, 1.2526])
        minimum = pd_action_offset - pd_action_scale
        maximum = pd_action_offset + pd_action_scale
        # minimum = []
        # maximum = []
        # for joint_, actuator in zip(self.joints, self.actuators):
        #     joint = actuator.joint
        #     assert joint == joint_
        #
        #     minimum.append(joint.range[0])
        #     maximum.append(joint.range[1])
        # import ipdb; ipdb.set_trace()

        if self.kd is None:
            minimum.append(-1.0)
            maximum.append(1.0)

        return specs.BoundedArray(
            shape=(len(minimum), ),
            dtype=np.float32,
            minimum=minimum,
            maximum=maximum,
            name='\t'.join([actuator.name for actuator in self.actuators]))

    @cached_property
    def ctrllimits(self):
        minimum = []
        maximum = []
        for actuator in self.actuators:
            minimum.append(actuator.ctrlrange[0])
            maximum.append(actuator.ctrlrange[1])

        return minimum, maximum

    def update_observations(self, obs):
        # update previous observations
        self._prev_observations.append(obs)

    # def update_observations(self, physics):
    #     # update previous observations
    #     self._prev_observations.append(self.get_curr_observation(physics))

    def update_actions(self, action):
        # curr_action = self.unnormalize_actions(desired_qpos.copy())
        self._prev_actions.append(action.copy())

    def unnormalize_actions(self, actions):
        return actions / self._QPOS_OFFSET - self._INIT_QPOS

    def apply_action(self, physics, desired_qpos, random_state):

        joints_bind = physics.bind(self.joints)
        qpos = joints_bind.qpos
        qvel = joints_bind.qvel

        if self.kd is None:
            min_kd = 1
            max_kd = 10

            kd = (desired_qpos[-1] + 1) / 2 * (max_kd - min_kd) + min_kd
            desired_qpos = desired_qpos[:-1]
        else:
            kd = self.kd

        action = self.kp * (desired_qpos - qpos) - kd * qvel
        minimum, maximum = self.ctrllimits
        action = np.clip(action, minimum, maximum)

        physics.bind(self.actuators).ctrl = action

    def _build_observables(self):
        return A1Observables(self)

    @property
    def root_body(self):
        return self._root_body

    @property
    def joints(self):
        """List of joint elements belonging to the arm."""
        return self._joints

    @property
    def observable_joints(self):
        return self._joints

    @property
    def actuators(self):
        """List of actuator elements belonging to the arm."""
        return self._actuators

    @property
    def mjcf_model(self):
        """Returns the `mjcf.RootElement` object corresponding to this robot."""
        return self._mjcf_root

    @property
    def prev_action(self):
        return np.concatenate(self._prev_actions)

    @property
    def prev_observation(self):
        return np.concatenate(self._prev_observations)

    @property
    def goal_obs(self):
        return self._goal_obs

    def get_roll_pitch_yaw(self, physics):
        quat = physics.bind(self.mjcf_model.sensor.framequat).sensordata
        return np.rad2deg(quat_to_euler(quat))

    def get_velocity(self, physics):
        velocimeter = physics.bind(self.mjcf_model.sensor.velocimeter)
        return velocimeter.sensordata

    # def get_robot_observation(self, physics):
    #
    #     base_quat = np.array(physics.bind(self.root_body).xquat, dtype=float, copy=True)[[1, 2, 3, 0]]
    #     qpos = np.array(physics.bind(self.joints).qpos, dtype=float, copy=True)
    #     # base_quat = torch.from_numpy(base_quat).unsqueeze(0)
    #     # base_quat = compute_local_root_quat(base_quat)[0].numpy()
    #     # import ipdb; ipdb.set_trace()
    #     return np.concatenate([base_quat, qpos])

    def get_curr_observation(self, physics):

        base_quat = np.array(physics.bind(self.root_body).xquat, dtype=float, copy=True)[[1, 2, 3, 0]]
        qpos = np.array(physics.bind(self.joints).qpos, dtype=float, copy=True)
        # ball_pos = self.get_ball_pos(physics)
        # base_quat = torch.from_numpy(base_quat).unsqueeze(0)
        # base_quat = compute_local_root_quat(base_quat)[0].numpy()
        # print('ball_pos: ', ball_pos)
        return np.concatenate([base_quat, qpos])

    # def get_ball_pos(self, physics):
    #     base_quat = np.array(physics.bind(self.root_body).xquat, dtype=float, copy=True)[[1, 2, 3, 0]]
    #     base_pos = np.array(physics.bind(self.root_body).xpos, dtype=float, copy=True)
    #     # import ipdb; ipdb.set_trace()
    #     ball_pos = np.array(physics.bind(self._ball_body).xpos, dtype=float, copy=True)
    #     ball_pos_xy = compute_local_pos(
    #         torch.from_numpy(base_pos).unsqueeze(0),
    #         torch.from_numpy(ball_pos).unsqueeze(0),
    #         torch.from_numpy(base_quat).unsqueeze(0))[0].numpy()
    #     ball_pos[:2] = ball_pos_xy
    #     return ball_pos

    # def update_goal_obs(self, physics):
    #     ball_pos = self.get_ball_pos(physics)
    #     base_quat = np.array(physics.bind(self.root_body).xquat, dtype=float, copy=True)[[1, 2, 3, 0]]
    #     base_pos = np.array(physics.bind(self.root_body).xpos, dtype=float, copy=True)
    #     rand_dist = np.random.uniform(0, 3, 1)
    #     rand_angle = np.random.uniform(0, 2 * np.pi, 1)
    #     goal_obs = np.concatenate([
    #         rand_dist * np.cos(rand_angle) + ball_pos[0],
    #         rand_dist * np.sin(rand_angle) + ball_pos[1],
    #         np.array([0.0])
    #     ])
    #     goal_obs_xy = compute_local_pos(
    #         torch.from_numpy(base_pos).unsqueeze(0),
    #         torch.from_numpy(goal_obs).unsqueeze(0),
    #         torch.from_numpy(base_quat).unsqueeze(0))[0].numpy()
    #     self._goal_obs = goal_obs_xy

