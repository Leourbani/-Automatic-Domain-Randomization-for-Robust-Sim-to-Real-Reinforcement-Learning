"""
Custom Walker2d environment with support for
domain randomization.

From: https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/mujoco/walker2d_v4.py

"""

import os
from typing import Dict, Tuple, Union, Optional

import numpy as np
import gymnasium as gym
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box


DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 2,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 1.0)),
    "elevation": -20.0,
}


class CustomWalker2d(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
            "rgbd_tuple"
        ],
        "render_fps": 125,
    }

    def __init__(
        self,
        xml_file: str = "walker2d.xml",
        frame_skip: int = 4,
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        forward_reward_weight: float = 1.0,
        ctrl_cost_weight: float = 1e-3,
        healthy_reward: float = 1.0,
        terminate_when_unhealthy: bool = True,
        healthy_z_range: Tuple[float, float] = (0.8, 2.0),
        healthy_angle_range: Tuple[float, float] = (-1.0, 1.0),
        reset_noise_scale: float = 5e-3,
        exclude_current_positions_from_observation: bool = True,
        domain: Optional[str] = None,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            forward_reward_weight,
            ctrl_cost_weight,
            healthy_reward,
            terminate_when_unhealthy,
            healthy_z_range,
            healthy_angle_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            domain,
            **kwargs,
        )

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._healthy_angle_range = healthy_angle_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        if xml_file == "walker2d.xml":
            xml_file = os.path.join(os.path.dirname(__file__), "assets/walker2d.xml")

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=None,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        obs_size = (
            self.data.qpos.size
            + self.data.qvel.size
            - exclude_current_positions_from_observation
        )

        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

        self.original_masses = np.copy(self.model.body_mass[1:])

        self._original_friction = np.copy(self.model.geom_friction)  # Default friction values

        # Domain shift
        if domain == "source":
            self.model.body_mass[1] -= -1.0


    @property
    def healthy_reward(self):
        return self.is_healthy * self._healthy_reward

    @property
    def is_healthy(self): # healthy if z and angle are in range
        z, angle = self.data.qpos[1:3]
        min_z, max_z = self._healthy_z_range
        min_angle, max_angle = self._healthy_angle_range
        
        is_healthy = (min_z < z < max_z) and (min_angle < angle < max_angle)

        return is_healthy

    def control_cost(self, action):
        return self._ctrl_cost_weight * np.sum(np.square(action))

    def _get_obs(self):
        qpos = self.data.qpos.flatten()
        qvel = np.clip(self.data.qvel.flatten(), -10.0, 10.0)

        if self._exclude_current_positions_from_observation:
            qpos = qpos[1:]

        observation = np.concatenate((qpos, qvel)).ravel()
        return observation

    def step(self, action):
        x_position_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        observation = self._get_obs()
        reward, reward_info = self._get_rew(x_velocity, action)
        terminated = not self.is_healthy and self._terminate_when_unhealthy

        info = {
            "x_position": x_position_after,
            "x_velocity": x_velocity,
            **reward_info,
        }

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, False, info

    def _get_rew(self, x_velocity, action):
        forward_reward = self._forward_reward_weight * x_velocity
        healthy_reward = self.healthy_reward
        ctrl_cost = self.control_cost(action)

        reward = forward_reward + healthy_reward - ctrl_cost

        reward_info = {
            "reward_forward": forward_reward,
            "reward_survive": healthy_reward,
            "reward_ctrl": -ctrl_cost,
        }

        return reward, reward_info

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            noise_low, noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            noise_low, noise_high, size=self.model.nv
        )

        self.set_state(qpos, qvel)
        observation = self._get_obs()
        return observation

    def _get_reset_info(self):
        return {
            "x_position": self.data.qpos[0],
        }

    ################### physical parameters #####################

    def get_link_masses(self):
        return np.array(self.model.body_mass[1:])

    def set_link_masses(self, masses):
        assert len(masses) == len(self.model.body_mass[1:])
        self.model.body_mass[1:] = masses

    def reset_link_masses(self):
        self.model.body_mass[1:] = self.original_masses.copy()

    def get_friction(self):
        return self.model.geom_friction.copy()

    def set_friction(self, friction):
        self.model.geom_friction[:] = friction

    def reset_friction(self):
        self.model.geom_friction[:] = self._nominal_friction


"""
    Registered environments
"""
gym.register(
    id="CustomWalker2d-v0",
    entry_point="env.custom_walker2d:CustomWalker2d",
    max_episode_steps=1000,
)

gym.register(
    id="CustomWalker2d-source-v0",
    entry_point="env.custom_walker2d:CustomWalker2d",
    max_episode_steps=1000,
    kwargs={"domain": "source"},
)

gym.register(
    id="CustomWalker2d-target-v0",
    entry_point="env.custom_walker2d:CustomWalker2d",
    max_episode_steps=1000,
    kwargs={"domain": "target"},
)
