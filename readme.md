# Synchronous Automatic Domain Randomization with Action Delays for Robust Sim-to-Real Reinforcement Learning

> **Authors:** Alessandro Mosca, Leonardo Urbani, Gabriele Viceconte (Politecnico di Torino)

## Overview
This project investigates Automatic Domain Randomization (ADR) as a method to enhance the robustness and sim-to-real transfer of reinforcement learning agents in MuJoCo locomotion environments. By exposing agents to diverse simulated conditions, the goal is to bridge the gap between simulation and reality, mitigating issues such as rare events, complex real-world conditions, and simulation errors.

---

## The Approach
We propose a simplified and computationally efficient variant of ADR. 

*   **Synchronous Training Cycle:** The ranges of dynamic parameters are progressively expanded based on the agent's performance.
*   **No Separate Buffers:** Unlike OpenAI's original ADR approach, our method updates randomization ranges online at the end of each rollout, without requiring separate buffers or boundary-specific evaluations.
*   **Simultaneous Adaptation:** Our approach allows multiple parameters to vary simultaneously, effectively capturing interactions between physical properties.
*   **Action Delays:** We incorporate action delays alongside physical parameters like masses and frictions to simulate more realistic and challenging control dynamics.

---

## Training Details
The reinforcement learning framework is built for MuJoCo environments.

*   **Framework:** Implemented using Stable-Baselines3, supporting both standard training and ADR.
*   **Algorithms:** Policies were trained using PPO and SAC.
*   **Hyperparameters:** The learning rate was set to $5\times10^{-4}$, and training ran for 500,000 steps using default hyperparameters.
*   **Training Regimes:** We compared a `NO_ADR` regime (fixed nominal environment) against an `ADR` regime (parameters are randomly sampled and progressively adapted).

---

## Environment Evaluation & Results
We evaluated the policies across four MuJoCo locomotion tasks of increasing complexity, testing multiple metrics including average return, episode variability, and robustness to domain shifts. Each configuration used 10 seeds with 50 episodes per seed.

### 1. Hopper
*   Hopper is a fragile, underactuated system with a narrow stability region.
*   ADR intentionally sacrifices some performance in the source domain to achieve higher stability and lower variance under domain shifts (e.g., a +1 kg shift in "Torso" mass).
*   ADR significantly improves worst-case performance in this environment.

### 2. Walker2d
*   While underactuated, Walker2d is structurally simpler than Hopper.
*   The `NO_ADR` policy suffers a drastic performance drop and nearly doubles its coefficient of variation in the target environment.
*   ADR demonstrates superior generalization, actually improving target domain performance and maintaining a much higher 10th percentile compared to `NO_ADR` in the source domain.

### 3. Swimmer
*   Swimmer is a low-dimensional environment with highly constrained dynamics.
*   Testing with ADR or shifting masses produced no notable effects, making it a poor benchmark for this algorithm.

### 4. Ant-v5
*   Ant is the most complex environment tested, featuring a high-dimensional state space, actuation redundancy, and non-linear dynamics.
*   ADR leads to higher average rewards in both the source and target domains by preventing the optimization algorithm from prematurely converging to suboptimal local minima.
*   Applying explicit delay randomization causes a substantial drop in average performance, confirming delay as a highly disruptive factor.
*   Testing isolated parameters revealed that masses are the dominant factor; without mass randomization, ADR loses power.

---

## Noise Analysis
To evaluate sensor robustness, zero-mean Gaussian noise was applied independently to each dimension of the state vector with a standard deviation of $\sigma=0.02$ during the test phase.

*   **Hopper & Ant-v5:** ADR consistently mitigates the impact of noise, showing smaller increases in variance and maintaining higher average rewards compared to non-ADR baselines.
*   **Walker2d:** Both ADR and non-ADR policies suffer a dramatic collapse in performance, exposing the limits of ADR when the underlying dynamics rely heavily on precise state estimation.
*   **Swimmer:** Performance remains virtually unchanged under noise for both models, confirming its simplicity and robustness to perceptual noise.