# per virtual env
# entrare in venv: source venv/bin/activate
# control+shif+P -> Python: Select Interpreter -> venv -> bin -> python3 (o python 3.10)

import gymnasium as gym
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt

from env.custom_hopper import *
from env.custom_walker2d import *
from env.custom_swimmer import *
from env.custom_ant_v5 import *

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback


ADR_CONFIGS = { # configurations for different environments
    "Hopper":   dict(init_range=(0.9, 1.1), step=0.05, threshold=100),
    "Walker2d": dict(init_range=(0.9, 1.1), step=0.05, threshold=100),
    "Swimmer":  dict(init_range=(0.95, 1.05), step=0.02, threshold=50),
    "Ant":      dict(init_range=(0.9, 1.1), step=0.05, threshold=100),
}

def make_adr_controller(env_id):
    for key, cfg in ADR_CONFIGS.items(): # "key" is the environment name, "cfg" is the value associated
        if key in env_id:
            print(f"Creating ADR controller for environment: {env_id} with config: {cfg} (key = {key})")
            return ADRController(**cfg) # calling the class. "**" is used to unpack the dictionary into keyword arguments


class ADRController: # ADR Controller (Hopper)

    def __init__(self, init_range=(0.9, 1.1), step=0.05, threshold=200):
        self.low, self.high = init_range
        self.step = step
        self.threshold = threshold

    def uniform_sample(self, size):
        return np.random.uniform(self.low, self.high, size) # try normal or triangular

    def normal_sample(self, size):
        low, high = sorted((self.low, self.high))  # to be sure
        mean = 1.0
        sigma = (high - low) / 2.0
        samples = np.random.normal(mean, sigma, size)
        samples = np.clip(samples, low, high)

        return samples

    def update(self, mean_reward):
        if mean_reward > self.threshold:
            self.threshold *= 1.5 # increase threshold
            self.low = max(0.5, self.low - self.step) # no under 0.5
            self.high = min(2.0, self.high + self.step)  # no over 2.0

        elif mean_reward < self.threshold*0.5: # decrease only if performance drops significantly
            self.low = min(self.low + self.step, 0.99)
            self.high = max(self.high - self.step, 1.01)



class ADRWrapper(gym.Wrapper): # ADR Wrapper

    def __init__(self, env, adr_controller):
        super().__init__(env)
        self.adr = adr_controller # adr controller

        assert hasattr(env.unwrapped, "get_link_masses") # check for physical parameter interface. "hasattr()" Return whether the object has an attribute with the given name.
        assert hasattr(env.unwrapped, "set_link_masses")

        self.nominal_masses = env.unwrapped.get_link_masses().copy() # store nominal masses
        self.nominal_frictions = env.unwrapped.get_friction().copy() # store nominal frictions

    def reset(self, **kwargs):# **kwargs is a generic way to pass variable length of arguments
        self.randomize_masses()
        self.randomize_frictions()
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def randomize_masses(self):
        scale = self.adr.uniform_sample(1)[0] # sample a scalar scaling factors (otherwise it gives me problems)
        self.env.unwrapped.set_link_masses(self.nominal_masses * scale) # setting new masses

    def randomize_frictions(self):
        scale = self.adr.uniform_sample(1)[0] # sample a scalar scaling factors (otherwise it gives me problems)
        self.env.unwrapped.set_friction(self.nominal_frictions * scale) # setting new frictions



class ADRCallback(BaseCallback): # ADR Callback

    def __init__(self, adr_controller, verbose=0):
        super().__init__(verbose)
        self.adr = adr_controller
        self.episode_rewards = []

    def _on_step(self):
        for info in self.locals["infos"]:
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
        return True

    def _on_rollout_end(self): # at the end of each rollout
        if len(self.episode_rewards) == 0: # to train sac I need this check
            return
        
        mean_reward = np.mean(self.episode_rewards)
        self.adr.update(mean_reward) # update ADR ranges based on performance
        self.episode_rewards.clear()

        env = self.training_env.envs[0].unwrapped # access the environment (Monitor → unwrap)

        masses = env.get_link_masses()
        frictions = env.get_friction()

        self.logger.record("adr/mean_reward", mean_reward) # printing parameters of the adr
        self.logger.record("adr/range_low", self.adr.low)
        self.logger.record("adr/range_high", self.adr.high)

        for i, m in enumerate(masses): # to print the masses
            self.logger.record(f"env/mass_{i}", m)

        for j,n in enumerate(frictions): # to print the frictions
            self.logger.record(f"env/friction_{j}", n)


################ FINE ADR ################

class ActionDelay(gym.Wrapper): # Action Delay Wrapper (UDR)

    def __init__(self, env, max_delay_steps=0):
        super().__init__(env)
        self.max_delay_steps = max_delay_steps
        self.delay_steps = 0
        self.buffer = []

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        if self.max_delay_steps > 0:
            self.delay_steps = np.random.randint(0, self.max_delay_steps + 1)
        else:
            self.delay_steps = 0
        print(f"Action delay steps = {self.delay_steps}")
        self.buffer = []
        return obs, info

    def step(self, action):
        self.buffer.append(action)
        if len(self.buffer) > self.delay_steps:
            action = self.buffer.pop(0) # .pop to remove and retourn the item at index 0
        else:
            action = np.zeros_like(action) # array of zeros
        return self.env.step(action)


def moving_average(values, window):
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, "valid")

        
def plot_results(log_folder, title="Learning Curve"):
 
    x, y = ts2xy(load_results(log_folder), "timesteps")
    y = moving_average(y, window=50)
    x = x[len(x) - len(y) :]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel("Number of Timesteps")
    plt.ylabel("Rewards")
    plt.title(title + " Smoothed")
    plt.show()

#NOISE
class SensorNoise(gym.ObservationWrapper):
    def __init__(self, env, noise_std=0.01):
        super().__init__(env)
        self.noise_std = noise_std

    def observation(self, obs):
        noise = np.random.normal(0, self.noise_std, size=obs.shape)
        return obs + noise




def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--test", "-t", type=str, default=None, help="Model to be tested")
    parser.add_argument("--env", type=str, default="CustomHopper-source-v0", help="Environments list: CustomWalker2d-source-v0, CustomSwimmer-source-v0, CustomAnt-source-v5")
    parser.add_argument("--total_timesteps", type=int, default=500000, help="The total number of samples to train on")
    parser.add_argument("--render_test", action='store_true', help="Render test")
    parser.add_argument('--lr', default=0.0005, type=float, help='Learning rate')
    parser.add_argument('--max_delay_steps', default=0, type=int, help='Number of action delay steps')
    parser.add_argument('--test_episodes', default=50, type=int, help='# episodes for test evaluations')
    parser.add_argument('--sensor_noise', default=0.0, type=float, help='Standard deviation of Gaussian sensor noise') # NOISE
    args = parser.parse_args()

    env = gym.make(args.env,                                           
                   render_mode='human' if args.render_test else None)
    #Noise
    if args.sensor_noise > 0.0:
        env = SensorNoise(env, noise_std=args.sensor_noise)

    print('State space:', env.observation_space)  # state-space
    print('Action space:', env.action_space)  # action-space
    print('Dynamics parameters:', env.unwrapped.get_link_masses())  # masses of each link of the Hopper

    log_dir = "./tmp/gym/"
    os.makedirs(log_dir, exist_ok=True)

    if args.test is None:
        try:
            
            env = ActionDelay(env, max_delay_steps=args.max_delay_steps)  # Add action delay
            adr = make_adr_controller(args.env)  # ADR controller definition
            env = ADRWrapper(env, adr)  # applying ADR to the environment
            env = Monitor(env, log_dir) # used to know the episode reward, length, time and other data

            model = PPO('MlpPolicy', env, learning_rate=args.lr, verbose=1)
            # model = SAC('MlpPolicy', env, learning_rate=args.lr, gamma=args.gamma, verbose=1)

            callback = ADRCallback(adr)

            model.learn(total_timesteps=args.total_timesteps, callback=callback)

            model.save("model_%s_%s_%s" % ('ppo', args.env, 'UDR_delay'))
            plot_results(log_dir)
        except KeyboardInterrupt:
            print("Interrupted!")
    else:
        print("Testing...")
        
        env = Monitor(env, log_dir) # used to know the episode reward, length, time and other data
        model = PPO.load(args.test, env=env)
        

        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=args.test_episodes, render=args.render_test)

        print(f"Test reward (avg +/- std): ({mean_reward} +/- {std_reward}) - Num episodes: {args.test_episodes}")

    env.close() 
    

if __name__ == '__main__':
    main()