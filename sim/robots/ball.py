import os
from collections import deque
from functools import cached_property
from typing import Optional

import numpy as np
from dm_control import composer, mjcf
from dm_control.composer.observation import observable
from dm_control.locomotion.walkers import base
from dm_control.entities import props
from dm_control.utils.transformations import quat_to_euler
from dm_env import specs

import torch

from ..wrappers.residual import ResidualWrapper
from ..robots.robot_utils import compute_local_root_quat, compute_local_pos

ASSETS_DIR = os.path.join(os.path.dirname(__file__), 'a1')
_BALL_XML_PATH = os.path.join(ASSETS_DIR, 'xml', 'ball.xml')


class BallObservables(composer.Observables):
    @composer.observable
    def goal_obs(self):
        return observable.MJCFFeature('xpos', self._entity.ball_body)


class Ball(props.Primitive):

    def _build(self,
               radius=0.097,
               mass=0.318,
               friction=(0.7, 0.075, 0.075),
               damp_ratio=1.0,
               name='ball'):
        """Initializes the JacoArm.

    Args:
      name: String, the name of this robot. Used as a prefix in the MJCF name
        name attributes.
    """
        self._mjcf_root = mjcf.from_path(_BALL_XML_PATH)

        self._ball_body = self._mjcf_root.find('body', 'ball')
        # self._ball_body.pos[0], self._ball_body.pos[1] = self.get_random_ball_pos()
        # import ipdb; ipdb.set_trace()
        self._ball_body.pos = (-0.28089765, 0.27256093, 0.1)

    def get_random_ball_pos(self):
        rand_dist = np.random.uniform(0, 3, 1)
        rand_angle = np.random.uniform(0, 2 * np.pi, 1)
        return (rand_dist * np.cos(rand_angle)).item(), (rand_dist * np.sin(rand_angle)).item()

    def _build_observables(self):
        return BallObservables(self)

    @property
    def ball_body(self):
        return self._ball_body

