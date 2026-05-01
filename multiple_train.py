import gymnasium as gym
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt

from env.custom_hopper import *
from env.custom_walker2d import *
from env.custom_swimmer import *
from env.custom_ant_v5 import *

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback


################ ADR CONFIG ################

ADR_CONFIGS = {
    "Hopper":   dict(init_range=(0.9, 1.1), step=0.05, threshold=100),
    "Walker2d": dict(init_range=(0.9, 1.1), step=0.05, threshold=100),
    "Swimmer":  dict(init_range=(0.95, 1.05), step=0.02, threshold=50),
    "Ant":      dict(init_range=(0.9, 1.1), step=0.05, threshold=100),
}

def make_adr_controller(env_id):
    for key, cfg in ADR_CONFIGS.items():
        if key in env_id:
            return ADRController(**cfg)


class ADRController:

    def __init__(self, init_range=(0.9, 1.1), step=0.05, threshold=200):
        self.low, self.high = init_range
        self.step = step
        self.threshold = threshold

    def uniform_sample(self, size):
        return np.random.uniform(self.low, self.high, size)

    def update(self, mean_reward):
        if mean_reward > self.threshold:
            self.threshold *= 1.5
            self.low = max(0.5, self.low - self.step)
            self.high = min(2.0, self.high + self.step)


class ADRWrapper(gym.Wrapper):

    def __init__(self, env, adr_controller):
        super().__init__(env)
        self.adr = adr_controller

        self.nominal_masses = env.unwrapped.get_link_masses().copy()
        self.nominal_frictions = env.unwrapped.get_friction().copy()

    def reset(self, **kwargs):
        scale = self.adr.uniform_sample(1)[0]
        self.env.unwrapped.set_link_masses(self.nominal_masses * scale)
        self.env.unwrapped.set_friction(self.nominal_frictions * scale)
        return self.env.reset(**kwargs)


class ADRCallback(BaseCallback):

    def __init__(self, adr_controller):
        super().__init__()
        self.adr = adr_controller
        self.episode_rewards = []

    def _on_step(self):
        for info in self.locals["infos"]:
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
        return True

    def _on_rollout_end(self):
        if len(self.episode_rewards) == 0:
            return
        self.adr.update(np.mean(self.episode_rewards))
        self.episode_rewards.clear()


################ EXPERIMENT ################

SEEDS = [0, 1, 2, 3, 4]

def train_and_evaluate(env_id, use_adr, total_timesteps):

    rewards = []
    log_dirs = []

    for seed in SEEDS:

        log_dir = f"./logs/{env_id}/{'adr' if use_adr else 'no_adr'}/seed_{seed}"
        os.makedirs(log_dir, exist_ok=True)

        env = gym.make(env_id)
        env.reset(seed=seed)

        if use_adr:
            adr = make_adr_controller(env_id)
            env = ADRWrapper(env, adr)

        env = Monitor(env, log_dir)

        model = PPO(
            "MlpPolicy",
            env,
            seed=seed,
            learning_rate=5e-4,
            verbose=0
        )

        callback = ADRCallback(adr) if use_adr else None

        model.learn(total_timesteps=total_timesteps, callback=callback)
        model.save(f"{log_dir}/model")

        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=50)
        rewards.append(mean_reward)
        log_dirs.append(log_dir)

        env.close()

    return np.array(rewards), log_dirs


def plot_mean_std(log_dirs, title):

    curves = []

    for d in log_dirs:
        x, y = ts2xy(load_results(d), "timesteps")
        curves.append(y)

    min_len = min(len(c) for c in curves)
    curves = np.array([c[:min_len] for c in curves])

    mean = curves.mean(axis=0)
    std = curves.std(axis=0)

    plt.figure()
    plt.plot(mean)
    plt.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.3)
    plt.title(title)
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.show()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CustomHopper-source-v0")
    parser.add_argument("--total_timesteps", type=int, default=500000)
    args = parser.parse_args()

    print("\n=== TRAINING NO ADR ===")
    no_adr_rewards, no_adr_logs = train_and_evaluate(
        args.env, use_adr=False, total_timesteps=args.total_timesteps
    )

    print("\n=== TRAINING WITH ADR ===")
    adr_rewards, adr_logs = train_and_evaluate(
        args.env, use_adr=True, total_timesteps=args.total_timesteps
    )

    print("\n===== RESULTS (SOURCE ENV) =====")
    print(f"No ADR  | mean={no_adr_rewards.mean():.2f} std={no_adr_rewards.std():.2f} min={no_adr_rewards.min():.2f}")
    print(f"ADR     | mean={adr_rewards.mean():.2f} std={adr_rewards.std():.2f} min={adr_rewards.min():.2f}")

    plot_mean_std(no_adr_logs, "No ADR - Learning Curve")
    plot_mean_std(adr_logs, "ADR - Learning Curve")


if __name__ == "__main__":
    main()
