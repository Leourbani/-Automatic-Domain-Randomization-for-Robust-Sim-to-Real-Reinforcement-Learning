"""
Custom Swimmer environment with ADR support (mass randomization)
Compatible with Gymnasium Swimmer-v5

From: https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/mujoco/walker2d_v4.py

"""

import os
from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box


DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 2,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 0.0)),
    "elevation": -20.0,
}


class CustomSwimmer(MujocoEnv, utils.EzPickle):

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 25,
    }

    def __init__(
        self,
        xml_file: str = "swimmer.xml",
        frame_skip: int = 4,
        forward_reward_weight: float = 1.0,
        ctrl_cost_weight: float = 1e-4,
        reset_noise_scale: float = 1e-2,
        exclude_current_positions_from_observation=True,
        domain: Optional[str] = None,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            forward_reward_weight,
            ctrl_cost_weight,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            domain,
        )

        if xml_file == "swimmer.xml":
            xml_file = os.path.join(
                os.path.dirname(__file__), "assets/swimmer.xml"
            )

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._reset_noise_scale = reset_noise_scale

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip=frame_skip,
            observation_space=None,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

        obs_size = self.data.qpos.size + self.data.qvel.size

        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

        # ---- store nominal parameters for ADR ----
        self._nominal_masses = np.copy(self.model.body_mass[1:])

        if domain == "source":
            # slight mismatch (example)
            self.model.body_mass[2] -= 1.0 # reducing the central body mass

    
    def _get_obs(self):
        return np.concatenate(
            (self.data.qpos.flatten(), self.data.qvel.flatten())
        )

    def control_cost(self, action):
        return self._ctrl_cost_weight * np.sum(np.square(action))

    def step(self, action):
        xpos_before = self.data.qpos[0]

        self.do_simulation(action, self.frame_skip)

        xpos_after = self.data.qpos[0]
        x_velocity = (xpos_after - xpos_before) / self.dt

        forward_reward = self._forward_reward_weight * x_velocity
        ctrl_cost = self.control_cost(action)

        reward = forward_reward - ctrl_cost
        

        observation = self._get_obs()
        info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "x_velocity": x_velocity,
        }

        if self.render_mode == "human":
            self.render()

        return observation, reward, False, False, info

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
        self.model.body_mass[1:] = self._nominal_masses

    def get_friction(self):
        return self.model.geom_friction.copy()

    def set_friction(self, friction):
        self.model.geom_friction[:] = friction

    def reset_friction(self):
        self.model.geom_friction[:] = self._nominal_friction


gym.register(
    id="CustomSwimmer-v0",
    entry_point="env.custom_swimmer:CustomSwimmer",
    max_episode_steps=2000,
)

gym.register(
    id="CustomSwimmer-source-v0",
    entry_point="env.custom_swimmer:CustomSwimmer",
    max_episode_steps=2000,
    kwargs={"domain": "source"},
)

gym.register(
    id="CustomSwimmer-target-v0",
    entry_point="env.custom_swimmer:CustomSwimmer",
    max_episode_steps=2000,
    kwargs={"domain": "target"},
)
