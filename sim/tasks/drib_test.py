from typing import Optional, Tuple

import dm_control.utils.transformations as tr
import ipdb
import numpy as np
from dm_control import composer
from dm_control.composer.observation import observable
from dm_control.locomotion import arenas
from dm_control.utils import rewards

from sim.arenas import HField, BallField
from sim.tasks.utils import _find_non_contacting_height, _find_non_contacting_height_with_ball
from sim.robots.robot_utils import compute_local_pos

DEFAULT_CONTROL_TIMESTEP = 0.03
DEFAULT_PHYSICS_TIMESTEP = 0.001
ENERGY_WEIGHT = 0.0
ENERGY_SCALE = 0.01


def get_run_reward(x_velocity: float, move_speed: float, cos_pitch: float,
                   dyaw: float):
    reward = rewards.tolerance(cos_pitch * x_velocity,
                               bounds=(move_speed, 2 * move_speed),
                               margin=2 * move_speed,
                               value_at_margin=0,
                               sigmoid='linear')
    reward -= 0.1 * np.abs(dyaw)

    return 10 * reward  # [0, 1] => [0, 10]


def get_dribble_reward(robot_pos, diff_root, ball_xy, diff_ball, goal_xy, torque, dof_vel, dt):
    root_xy = robot_pos[:2]
    # energy saving reward
    energy_sum = np.sum(np.square(torque * dof_vel))
    energy_reward = np.exp(- ENERGY_SCALE * energy_sum)

    v_char = diff_root / dt
    v_ball = diff_ball / dt

    diff_cb = ball_xy - root_xy
    dist_cb = diff_cb[0] ** 2 + diff_cb[1] ** 2
    d_cb = diff_cb / np.sqrt(dist_cb)

    diff_bg = goal_xy - ball_xy
    dist_bg = diff_bg[0] ** 2 + diff_bg[1] ** 2
    d_bg = diff_bg / np.sqrt(dist_bg)

    ball_vel = d_bg[0] * v_ball[0] + d_bg[1] * v_ball[1]
    actor_vel = d_cb[0] * v_char[0] + d_cb[1] * v_char[1]

    ball_vel_static = rewards.tolerance(ball_vel, bounds=(0., 0.), margin=0.05, sigmoid='linear')
    ball_vel_move = rewards.tolerance(ball_vel, bounds=(0.5, 1.), margin=0.2, sigmoid='linear')
    actor_vel_static = rewards.tolerance(actor_vel, bounds=(0., 0.), margin=0.1, sigmoid='linear')
    actor_vel_move = rewards.tolerance(actor_vel, bounds=(0.5, 1.), margin=1., sigmoid='linear')

    ball_vel_static = np.where(dist_bg < 0.04, ball_vel_static, np.zeros_like(ball_vel_static))
    ball_vel_move = np.where(dist_bg < 0.04, np.ones_like(ball_vel_move), ball_vel_move)
    actor_vel_static = np.where(dist_bg < 0.04, np.ones_like(actor_vel_static),
                                np.where(dist_cb < 0.04, actor_vel_static, np.zeros_like(actor_vel_static)))
    actor_vel_move = np.where(dist_bg < 0.04, np.ones_like(actor_vel_move),
                              np.where(dist_cb < 0.04, np.ones_like(actor_vel_move), actor_vel_move))
    reward = 0.1 * actor_vel_static + 0.1 * actor_vel_move + 0.4 * ball_vel_static + 0.4 * ball_vel_move
    total_reward = (1 - ENERGY_WEIGHT) * reward + ENERGY_WEIGHT * energy_reward
    return total_reward


class DribTest(composer.Task):

    def __init__(self,
                 robot,
                 object_params,
                 energy_weight,
                 terminate_pitch_roll: Optional[float] = 45,
                 physics_timestep: float = DEFAULT_PHYSICS_TIMESTEP,
                 control_timestep: float = DEFAULT_CONTROL_TIMESTEP,
                 floor_friction: Tuple[float] = (1, 0.005, 0.0001),
                 randomize_ground: bool = True,
                 add_velocity_to_observations: bool = True):

        _object_size = [float(i) for i in object_params['object_size']]
        self._object_height = _object_size[-1]
        _object_type = object_params['object_type']
        self.energy_weight = energy_weight
        self.qvel = None
        self.torque = None
        self._prev_ball_xy = None
        self._prev_robot_xy = np.array((0.0, 0.0))
        self.floor_friction = floor_friction
        self._floor = BallField(size=(10, 10))
        self._floor.mjcf_model.size.nconmax = 400
        self._floor.mjcf_model.size.njmax = 2000
        self._ball_frame = None

        for geom in self._floor.mjcf_model.find_all('geom'):
            geom.friction = floor_friction

        self._robot = robot
        self._floor.add_free_entity(self._robot)

        self._ball_frame = self._floor.attach(self._floor._ball)
        self._ball_frame.add('joint',
                             type='free',
                             damping="0.01",
                             armature="0.01",
                             frictionloss="0.2",
                             limited='False'
                             # limited='false',
                             # solreflimit="0.01 1",
                             # solimplimit="0.9 0.99 0.01"
                             )
        # import ipdb; ipdb.set_trace()
        self.ball_body = self._floor._ball.mjcf_model.find('body', 'ball')
        self._floor._ball.mjcf_model.find('geom', 'ball_geom').type = _object_type
        if _object_type == 'ball':
            _object_size = _object_size[0]
        self._floor._ball.mjcf_model.find('geom', 'ball_geom').size = _object_size

        self._add_goal_sensor(self._floor)
        self._goal_loc = self._floor.mjcf_model.find('site', 'target_goal').pos
        observables = (
                [self._robot.observables.body_position] +
                [self._robot.observables.body_rotation] +
                [self._robot.observables.joints_pos] +
                [self._robot.observables.prev_action])

        for observable in observables:
            observable.enabled = True

        if not add_velocity_to_observations:
            self._robot.observables.sensors_velocimeter.enabled = False

        if hasattr(self._floor, '_top_camera'):
            self._floor._top_camera.remove()
        self._robot.mjcf_model.worldbody.add('camera',
                                             name='side_camera',
                                             pos=[0, -1.0, 0.5],
                                             xyaxes=[1, 0, 0, 0, 0.342, 0.940],
                                             mode="trackcom",
                                             fovy=100.0)

        self.set_timesteps(physics_timestep=physics_timestep,
                           control_timestep=control_timestep)

        self._terminate_pitch_roll = terminate_pitch_roll

        self._move_speed = 0.5

    def _add_goal_sensor(self, floor):
        floor.mjcf_model.worldbody.add('site',
                                       name="target_goal",
                                       size=[0.1] * 3,
                                       pos=[-3.6, 0.0, .125],
                                       group=0)

    def get_reward(self, physics):
        robot_pos = np.array(physics.bind(self._robot.root_body).xpos, dtype=float)
        ball_pos = np.array(physics.bind(self.ball_body).xpos, dtype=float)
        goal_pos = self._goal_loc
        diff_root = robot_pos[:2] - self._prev_robot_xy
        diff_ball = ball_pos[:2] - self._prev_ball_xy
        self._prev_robot_xy[:] = robot_pos[:2]
        self._prev_ball_xy[:] = ball_pos[:2]

        return get_dribble_reward(robot_pos=robot_pos,
                                  diff_root=diff_root,
                                  ball_xy=ball_pos[:2],
                                  diff_ball=diff_ball,
                                  goal_xy=goal_pos[:2],
                                  torque=self.torque,
                                  dof_vel=self.qvel,
                                  dt=DEFAULT_CONTROL_TIMESTEP)

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
        # import ipdb; ipdb.set_trace()
        # self._floor._ball.mjcf_model.find('body', 'ball').pos = (-0.28089765, 2.7256093, 0.5)
        # sampled_x = np.random.uniform(-2.0, 2.0)
        # print("Setting ball x to:", sampled_x)
        self.ball_body.pos = self.sample_ball_pos(random_state, self._object_height)
        self._prev_ball_xy = self.ball_body.pos[:2]

        # randomize goal loc
        self.sample_goal(random_state, self.ball_body.pos[0], self.ball_body.pos[1])
        # self._goal_loc = (-0.31705597, 0.91705686, 0.125)
        goal_site = self._floor.mjcf_model.find('site', 'target_goal')
        goal_site.pos = self._goal_loc

    def initialize_episode(self, physics, random_state):
        super().initialize_episode(physics, random_state)
        self._floor.initialize_episode(physics, random_state)

        self._failure_termination = False

        _find_non_contacting_height(physics, self._robot, qpos=self._robot._INIT_QPOS)
        # _find_non_contacting_height_with_ball(physics, self._robot, self._floor._ball, self.ball_body.pos[0], self.ball_body.pos[1], qpos=self._robot._INIT_QPOS)

    def sample_goal(self, random_state, ball_x, ball_y):
        # import ipdb; ipdb.set_trace()
        x_pos = random_state.uniform(low=-3.0, high=3.0) + ball_x
        y_pos = random_state.uniform(low=-3.0, high=3.0) + ball_y
        self._goal_loc = np.array([x_pos, y_pos, 0.125], dtype=np.float32)

    def sample_ball_pos(self, random_state, h):
        x_pos = random_state.uniform(low=-5.0, high=5.0)
        y_pos = random_state.uniform(low=-5.0, high=5.0)
        return np.array([x_pos, y_pos, h+1e-3], dtype=np.float32)

    @property
    def task_observables(self):
        task_observables = super().task_observables
        ball_pos = observable.MJCFFeature('xpos', self.ball_body)
        goal_pos = observable.Generic(lambda _: self._goal_loc)
        # import ipdb; ipdb.set_trace()
        task_observables['ball_loc'] = ball_pos
        task_observables['goal_loc'] = goal_pos
        ball_pos.enabled = True
        goal_pos.enabled = True
        return task_observables

    def before_step(self, physics, action, random_state):
        # self._robot.update_actions(physics, action, random_state)
        pass

    def before_substep(self, physics, action, random_state):
        self.torque, self.qvel = self._robot.apply_action(physics, action, random_state)

    def action_spec(self, physics):
        return self._robot.action_spec

    def after_step(self, physics, random_state):
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
