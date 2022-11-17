from collections import OrderedDict
from typing import Optional, Tuple

import dm_control.utils.transformations as tr
import numpy as np
from dm_control import composer
from dm_control.locomotion import arenas
from dm_control.utils import rewards
from dm_control.composer.observation import observable

from sim.arenas import HField
from sim.tasks.utils import _find_non_contacting_height
from sim.robots.ball import Ball


import torch

from ..wrappers.residual import ResidualWrapper
from ..robots.robot_utils import compute_local_root_quat, compute_local_pos

DEFAULT_CONTROL_TIMESTEP = 0.03
DEFAULT_PHYSICS_TIMESTEP = 0.001

# observable class for task / how to write task_observables
# how to init prev_observation


def get_run_reward(x_velocity: float, move_speed: float, cos_pitch: float,
                   dyaw: float):
    reward = rewards.tolerance(cos_pitch * x_velocity,
                               bounds=(move_speed, 2 * move_speed),
                               margin=2 * move_speed,
                               value_at_margin=0,
                               sigmoid='linear')
    reward -= 0.1 * np.abs(dyaw)

    return 10 * reward  # [0, 1] => [0, 10]


class DribbleObservables(composer.Observables):
    @composer.observable
    def a_prev_observation(self):
        return observable.Generic(lambda _: self._entity.prev_observation)

    @composer.observable
    def curr_observation(self):
        return observable.Generic(
            lambda physics: self._entity.get_curr_observation(physics)
        )

    @composer.observable
    def goal_obs(self):
        return observable.Generic(lambda _: self._entity.goal_obs)


class Dribble(composer.Task):

    def __init__(self,
                 robot,
                 terminate_pitch_roll: Optional[float] = 30,
                 physics_timestep: float = DEFAULT_PHYSICS_TIMESTEP,
                 control_timestep: float = DEFAULT_CONTROL_TIMESTEP,
                 floor_friction: Tuple[float] = (1, 0.005, 0.0001),
                 randomize_ground: bool = True,
                 add_velocity_to_observations: bool = True):

        self.floor_friction = floor_friction
        if randomize_ground:
            self._floor = HField(size=(10, 10))
            self._floor.mjcf_model.size.nconmax = 400
            self._floor.mjcf_model.size.njmax = 2000
        else:
            self._floor = arenas.Floor(size=(10, 10))

        for geom in self._floor.mjcf_model.find_all('geom'):
            geom.friction = floor_friction

        self._robot = robot
        self._floor.add_free_entity(self._robot)
        self._ball = Ball()
        self._floor.add_free_entity(self._ball)

        self._goal_obs = np.array([-0.31705597, 0.91705686])

        # self._build_obs()
        # import ipdb;ipdb.set_trace()
        # observables = ([self._robot.observables.a_prev_observation] +
        #                [self._robot.observables.prev_action])
        # observables = ([self._robot.observables.curr_observation])

        for observable in self.get_observables():
            observable.enabled = True

        import ipdb; ipdb.set_trace()
        # look at observables

        if not add_velocity_to_observations:
            self._robot.observables.sensors_velocimeter.enabled = False

        if hasattr(self._floor, '_top_camera'):
            self._floor._top_camera.remove()
        self._robot.mjcf_model.worldbody.add('camera',
                                             name='side_camera',
                                             pos=[0, -1, 0.5],
                                             xyaxes=[1, 0, 0, 0, 0.342, 0.940],
                                             mode="trackcom",
                                             fovy=60.0)

        self.set_timesteps(physics_timestep=physics_timestep,
                           control_timestep=control_timestep)

        self._terminate_pitch_roll = terminate_pitch_roll

        self._move_speed = 0.5

    def get_reward(self, physics):
        xmat = physics.bind(self._robot.root_body).xmat.reshape(3, 3)
        _, pitch, _ = tr.rmat_to_euler(xmat, 'XYZ')
        velocimeter = physics.bind(self._robot.mjcf_model.sensor.velocimeter)

        gyro = physics.bind(self._robot.mjcf_model.sensor.gyro)

        return get_run_reward(x_velocity=velocimeter.sensordata[0],
                              move_speed=self._move_speed,
                              cos_pitch=np.cos(pitch),
                              dyaw=gyro.sensordata[-1])

    def initialize_episode_mjcf(self, random_state):
        super().initialize_episode_mjcf(random_state)

        # Terrain randomization
        if hasattr(self._floor, 'regenerate'):
            self._floor.regenerate(random_state)
            self._floor.mjcf_model.visual.map.znear = 0.00025
            self._floor.mjcf_model.visual.map.zfar = 50.

        new_friction = (random_state.uniform(low=self.floor_friction[0] - 0.25,
                                             high=self.floor_friction[0] +
                                             0.25), self.floor_friction[1],
                        self.floor_friction[2])
        for geom in self._floor.mjcf_model.find_all('geom'):
            geom.friction = new_friction

    def initialize_episode(self, physics, random_state):
        super().initialize_episode(physics, random_state)
        self._floor.initialize_episode(physics, random_state)

        self._failure_termination = False

        _find_non_contacting_height(physics,
                                    self._robot,
                                    qpos=self._robot._INIT_QPOS)

    def before_step(self, physics, action, random_state):
        # self._robot.update_actions(physics, action, random_state)
        pass

    def before_substep(self, physics, action, random_state):
        self._robot.apply_action(physics, action, random_state)

    def action_spec(self, physics):
        return self._robot.action_spec

    def after_step(self, physics, random_state):
        # self._robot.update_observations(physics)
        self._failure_termination = False

        if self._terminate_pitch_roll is not None:
            roll, pitch, _ = self._robot.get_roll_pitch_yaw(physics)

            if (np.abs(roll) > self._terminate_pitch_roll
                    or np.abs(pitch) > self._terminate_pitch_roll):
                self._failure_termination = True

    def should_terminate_episode(self, physics):
        return self._failure_termination

    def get_discount(self, physics):
        if self._failure_termination:
            return 0.0
        else:
            return 1.0

    @property
    def root_entity(self):
        return self._floor

    @property
    def robot(self):
        return self._robot

    @property
    def prev_observation(self):
        pass

    @property
    def goal_pos(self):
        return self._goal_obs

    def get_observables(self):
        drib_obs = DribbleObservables(self)
        obs = []
        for k, v in drib_obs.as_dict(fully_qualified=False).items():
            obs.append(v)
        obs.append(self._robot.observables.prev_action)
        return obs

    def get_curr_observation(self, physics):
        base_quat = np.array(physics.bind(self._robot.root_body).xquat, dtype=float, copy=True)[[1, 2, 3, 0]]
        base_pos = np.array(physics.bind(self._robot.root_body).xpos, dtype=float, copy=True)
        ball_pos = np.array(physics.bind(self._ball.ball_body).xpos, dtype=float, copy=True)
        ball_pos_xy = compute_local_pos(
            torch.from_numpy(base_pos).unsqueeze(0),
            torch.from_numpy(ball_pos).unsqueeze(0),
            torch.from_numpy(base_quat).unsqueeze(0))[0].numpy()
        ball_pos[:2] = ball_pos_xy
        qpos = np.array(physics.bind(self._robot.joints).qpos, dtype=float, copy=True)
        base_quat = torch.from_numpy(base_quat).unsqueeze(0)
        base_quat = compute_local_root_quat(base_quat)[0].numpy()
        # import ipdb; ipdb.set_trace()
        return np.concatenate([base_quat, qpos, ball_pos])

    def update_goal_pos(self):
        pass

