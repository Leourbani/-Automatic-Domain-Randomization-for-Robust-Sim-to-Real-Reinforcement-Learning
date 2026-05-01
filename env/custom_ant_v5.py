"""
Custom Walker2d environment with support for
domain randomization.

From: https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/mujoco/ant_v5.py
"""

import os
from typing import Dict, Tuple, Union, Optional

import numpy as np
import gymnasium as gym
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box


DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 0.5)),
    "elevation": -20.0,
}


class CustomAnt(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 20,
    }

    def __init__(
        self,
        xml_file: str = "ant.xml",
        frame_skip: int = 5,
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        forward_reward_weight: float = 1.0,
        ctrl_cost_weight: float = 0.5,
        healthy_reward: float = 1.0,
        terminate_when_unhealthy: bool = True,
        healthy_z_range: Tuple[float, float] = (0.2, 1.0),
        reset_noise_scale: float = 0.1,
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
            reset_noise_scale,
            exclude_current_positions_from_observation,
            domain,
            **kwargs,
        )

        if xml_file == "ant.xml":
            xml_file = os.path.join(os.path.dirname(__file__), "assets/ant.xml")

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=None,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        excluded = 2 if exclude_current_positions_from_observation else 0 # to exclude x,y position, otherwise dimensional error
        
        obs_size = (
            self.data.qpos.size
            + self.data.qvel.size
            - excluded
        )

        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

        # Save nominal physical parameters
        self._nominal_masses = np.copy(self.model.body_mass[1:])
        self._nominal_friction = np.copy(self.model.geom_friction)

        # Optional sim-to-real mismatch
        if domain == "source":
            self.model.body_mass[1] -= 0.2  # torso mass reduced by 0.2 kg (nominal value is 0.35)


    @property
    def is_healthy(self):
        z = self.data.qpos[2]
        min_z, max_z = self._healthy_z_range # default (0.2, 1.0)
        return min_z <= z <= max_z # true if z inside the range (healthy), false if unhealthy

    @property
    def healthy_reward(self):
        return self.is_healthy * self._healthy_reward

    def control_cost(self, action):
        return self._ctrl_cost_weight * np.sum(np.square(action))

    def _get_obs(self):
        position = self.data.qpos.flatten()
        velocity = np.clip(self.data.qvel.flatten(), -10, 10)

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        return np.concatenate((position, velocity))

    def step(self, action):
        x_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_after = self.data.qpos[0]

        x_velocity = (x_after - x_before) / self.dt

        observation = self._get_obs()

        forward_reward = self._forward_reward_weight * x_velocity
        healthy_reward = self.healthy_reward
        ctrl_cost = self.control_cost(action)

        reward = forward_reward + healthy_reward - ctrl_cost

        terminated = (not self.is_healthy) and self._terminate_when_unhealthy # _terminate_when_unhealthy is True by default

        info = {
            "x_velocity": x_velocity,
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_survive": healthy_reward,
        }

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, False, info

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
        return self._get_obs()



    def get_link_masses(self):
        return np.array(self.model.body_mass[1:])

    def set_link_masses(self, masses):
        assert len(masses) == len(self.model.body_mass[1:])
        self.model.body_mass[1:] = masses

    def reset_link_masses(self):
        self.model.body_mass[1:] = self._nominal_masses.copy()

    def get_friction(self):
        return self.model.geom_friction.copy()

    def set_friction(self, friction):
        self.model.geom_friction[:] = friction

    def reset_friction(self):
        self.model.geom_friction[:] = self._nominal_friction.copy()




gym.register(
    id="CustomAnt-v5",
    entry_point="env.custom_ant_v5:CustomAnt",
    max_episode_steps=1000,
)

gym.register(
    id="CustomAnt-source-v5",
    entry_point="env.custom_ant_v5:CustomAnt",
    max_episode_steps=1000,
    kwargs={"domain": "source"},
)

gym.register(
    id="CustomAnt-target-v5",
    entry_point="env.custom_ant_v5:CustomAnt",
    max_episode_steps=1000,
    kwargs={"domain": "target"},
)
