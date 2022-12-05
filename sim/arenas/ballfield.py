# Copyright 2020 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Bowl arena with bumps."""

from dm_control import composer, mjcf
from dm_control.locomotion.arenas import assets as locomotion_arenas_assets
from dm_control.mujoco.wrapper import mjbindings
from scipy import ndimage
from dm_control.locomotion import arenas

mjlib = mjbindings.mjlib

import os

# Constants related to terrain generation.
_TERRAIN_SMOOTHNESS = .5  # 0.0: maximally bumpy; 1.0: completely smooth.
_TERRAIN_BUMP_SCALE = .2  # Spatial scale of terrain bumps (in meters).

ASSETS_DIR = os.path.dirname(__file__)
_BALL_XML_PATH = os.path.join(ASSETS_DIR, 'ball.xml')


class BallField(arenas.Floor):
    """A bowl arena with sinusoidal bumps."""

    def _build(self, size=(10, 10), reflectance=0., aesthetic='default',
               name='floor', top_camera_y_padding_factor=1.1,
               top_camera_distance=100):
        super()._build(size=size,
                       reflectance=reflectance,
                       aesthetic=aesthetic, name=name,
                       top_camera_distance=top_camera_distance,
                       top_camera_y_padding_factor=top_camera_y_padding_factor)
        assert size[0] == size[1]

        self._ball = composer.ModelWrapperEntity(mjcf.from_path(_BALL_XML_PATH))

        self._mjcf_root.visual.scale.set_attributes(
            forcewidth='0.01', contactwidth='0.06', contactheight='0.1')
