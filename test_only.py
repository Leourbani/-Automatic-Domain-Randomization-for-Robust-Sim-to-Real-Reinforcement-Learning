import gymnasium as gym
import numpy as np
import os
import matplotlib.pyplot as plt

from env.custom_hopper import *
from env.custom_walker2d import *
from env.custom_swimmer import *
from env.custom_ant_v5 import *

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor


N_TESTS = 10
N_EVAL_EPISODES = 50
SEEDS = range(N_TESTS)

def coefficient_of_variation(x):
    return np.std(x) / np.mean(x) if np.mean(x) != 0 else 0

def plot_boxplot(results, title):
    plt.figure(figsize=(8, 5))
    plt.boxplot(results.values(), labels=results.keys(), showfliers=True)
    plt.ylabel("Mean Reward per Test")
    plt.title(title)
    plt.grid(True, axis="y", linestyle='--', alpha=0.7)
    plt.show()

def plot_seed_trend(results, title):
    plt.figure(figsize=(8, 5))
    for label, rewards in results.items():
        plt.plot(SEEDS, rewards, marker="o", label=label)
    plt.xlabel("Seed / Test Index")
    plt.ylabel("Mean Reward")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

def plot_cdf(results, title):
    plt.figure(figsize=(8, 5))
    for label, rewards in results.items():
        sorted_r = np.sort(rewards)
        cdf = np.arange(1, len(sorted_r) + 1) / len(sorted_r)
        plt.plot(sorted_r, cdf, label=label, marker='.')
    plt.xlabel("Reward")
    plt.ylabel("CDF")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

def print_robust_stats(results, name):
    print(f"\n===== {name} ROBUSTNESS STATS (Over {N_TESTS} Tests) =====")
    for k, v in results.items():
        worst_10 = np.percentile(v, 10)
        print(
            f"{k:7s} | mean={np.mean(v):.1f} "
            f"std={np.std(v):.1f} "
            f"CV={coefficient_of_variation(v):.3f} "
            f"min={np.min(v):.1f} "
            f"p10={worst_10:.1f}"
        )

def run_robustness_test(model_path, env_id, label):
    """Esegue N_TESTS valutazioni, ognuna con un seed diverso."""
    if not os.path.exists(model_path) and not os.path.exists(model_path + ".zip"):
        print(f"[ERROR] Model not found at: {model_path}")
        return np.zeros(N_TESTS)

    test_results = []
    
    model = PPO.load(model_path)

    print(f"Running {N_TESTS} tests for {label} on {env_id}...")
    
    for seed in SEEDS:
        env = gym.make(env_id)
        env = Monitor(env)
        env.reset(seed=seed)
        
        mean_reward, _ = evaluate_policy(
            model,
            env,
            n_eval_episodes=N_EVAL_EPISODES,
            deterministic=True
        )
        
        test_results.append(mean_reward)
        env.close()
        print(f" > Test {seed+1}/{N_TESTS} (Seed {seed}) completed. Mean Reward: {mean_reward:.2f}")

    return np.array(test_results)

def main():
    path_no_adr = "trained_models/model_ppo_no_adr_CustomHopper-source-v0"
    path_adr    = "trained_models/model_ppo_CustomHopper-source-v0"
    # path_delay  = "trained_models/model_ppo_CustomAnt-source-v5_UDR_delay"

    source_env = "CustomHopper-source-v0"
    target_env = "CustomHopper-target-v0"

    print("\n--- EVALUATING SOURCE DOMAIN ---")
    no_adr_source = run_robustness_test(path_no_adr, source_env, "NO_ADR")
    adr_source    = run_robustness_test(path_adr,    source_env, "ADR")
    # del_source    = run_robustness_test(path_delay,  source_env, "WITH_DELAY")

    print("\n--- EVALUATING TARGET DOMAIN ---")
    no_adr_target = run_robustness_test(path_no_adr, target_env, "NO_ADR")
    adr_target    = run_robustness_test(path_adr,    target_env, "ADR")
    # del_target    = run_robustness_test(path_delay,  target_env, "WITH_DELAY")

    results_source = {"NO_ADR": no_adr_source, "ADR": adr_source}#, "WITH_DELAY": del_source}
    results_target = {"NO_ADR": no_adr_target, "ADR": adr_target}#, "WITH_DELAY": del_target}

    plot_seed_trend(results_source, f"SOURCE ({source_env}) - Performance per Seed")
    plot_seed_trend(results_target, f"TARGET ({target_env}) - Performance per Seed")
    
    plot_boxplot(results_target, "TARGET - Distribution of Test Means")
    plot_cdf(results_target, "TARGET - Cumulative Distribution of Means")

    print_robust_stats(results_source, "SOURCE")
    print_robust_stats(results_target, "TARGET")

if __name__ == "__main__":
    main()