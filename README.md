# RL-Odyssey - Deep Reinforcement Learning

A reinforcement learning experiments package for implementing and benchmarking state-of-the-art algorithms across various environments. Mainly focused on continuous control tasks for personal research and learning purposes.

## Table of Contents

- [RL-Odyssey - Deep Reinforcement Learning](#rl-odyssey---deep-reinforcement-learning)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Features](#features)
  - [Getting Started](#getting-started)
    - [Installation](#installation)
    - [Quick Start](#quick-start)
      - [1. Interactive Notebooks](#1-interactive-notebooks)
      - [2. Clean Experiment Scripts](#2-clean-experiment-scripts)
      - [3. Advanced Usage](#3-advanced-usage)
    - [Project Structure](#project-structure)
  - [Algorithm Comparison](#algorithm-comparison)
  - [License](#license)
  - [Acknowledgments](#acknowledgments)

## Overview

RL-Odyssey provides a flexible framework for reinforcement learning research and experimentation, with implementations of key algorithms (DDPG, TD3, SAC, PPO, TRPO, A2C, A3C, PG) and support for multiple environment types. The project emphasizes reproducibility, containerization, and efficient experiment management.

## Features

- Multiple algorithm implementations:
  - Policy Gradient (PG)
  - Advantage Actor-Critic (A2C)
  - Asynchronous Advantage Actor-Critic (A3C)
  - Trust Region Policy Optimization (TRPO)
  - Proximal Policy Optimization (PPO)
  - Deep Deterministic Policy Gradient (DDPG)
  - Twin Delayed DDPG (TD3)
  - Soft Actor-Critic (SAC)
- Environment support:
  - Gymnasium/OpenAI Gym (MuJoCo environments like Ant-v5, Humanoid-v5)
  - DeepMind Control Suite
  - Atari environments
- Experiment management:
  - Docker containerization for reproducibility (_in progress_)
  - Logging with TensorBoard and/or Weights & Biases
  - Experiment configuration through YAML or command-line arguments
  - Checkpointing for resuming training (_in progress_)

## Getting Started

### Installation

1. Clone the repository:

```bash
git clone https://github.com/legalaspro/rl-odyssey.git
cd rl-odyssey
```

2. Install the required dependencies:

```bash
pip install -e .
```

3. Optionallly: Build the Docker image for running experiments in a container:

```bash
docker build -f Dockerfile.cpu -t rl_odyssey:cpu .
```

### Quick Start

RL-Odyssey offers two ways to engage with the algorithms:

#### 1. Interactive Notebooks

Explore our Jupyter notebooks that combine mathematical explanations with complete implementations:

```bash
# Launch Jupyter to explore the notebooks
jupyter notebook notebooks/
```

Key notebooks include:

- **Policy Gradients**
  - `notebooks/0.PG_Pong` - Policy Gradient implementation for Pong
- **Advantage Actor-Critic (A2C)**

  - `notebooks/1.1_A2C-pong.ipynb` - A2C implementation for Pong
  - `notebooks/1.2_A2C-upgrade-pong.ipynb` - Enhanced A2C for Pong
  - `notebooks/1.3_A2C-continuous.ipynb` - A2C for continuous control tasks

- **Trust Region Policy Optimization (TRPO)**

  - `notebooks/3.TRPO+GAE-continuous.ipynb` - TRPO with GAE implementation and theory

- **Proximal Policy Optimization (PPO)**

  - `notebooks/4.PPO-Continuous.ipynb` - PPO for continuous control environments

- **Deep Deterministic Policy Gradient (DDPG)**

  - `notebooks/5.DDPG-Continuous.ipynb` - DDPG implementation with theoretical explanations

- **Twin Delayed DDPG (TD3)**

  - `notebooks/6.TD3-Continuous.ipynb` - TD3 algorithm with enhancements over DDPG

- **Soft Actor-Critic (SAC)**
  - `notebooks/7.SAC-Continuous.ipynb` - SAC implementation with entropy regularization theory

#### 2. Clean Experiment Scripts

For direct experimentation, use our focused Python implementations:

```bash
# Run SAC on Ant-v5
python rl_experiments/continuous_sac.py --env-id Ant-v5

# Run TD3 on Humanoid-v5
python rl_experiments/continuous_td3.py --env-id Humanoid-v5
```

#### 3. Advanced Usage

Customize your SAC experiment with specific hyperparameters:

```bash
python rl_experiments/continuous_sac.py \
    --env-id Humanoid-v5 \
    --hidden-dim 1024 \
    --batch-size 512 \
    --alpha-lr 3e-4 \
    --reward-scale 0.2
```

Experiment Monitoring:

Training logs are stored in the runs/ directory and can be visualized with TensorBoard:

```bash
tensorboard --logdir runs/
```

### Project Structure

```bash
rl-odyssey/
├── notebooks/ # Jupyter notebooks with theoretical explanations
│   ├── 0.PG_Pong/ # Policy Gradient implementations
│   ├── 1.1_A2C-pong.ipynb # A2C for Pong
│   └── ... # Other algorithm notebooks
├── rl_experiments/ # Clean Python implementations
│   ├── continuous_sac.py # SAC implementation
│   ├── continuous_td3.py # TD3 implementation
│   └── ... # Other algorithms
├── parallel/ # Multi-process implementations
│   ├── continuous_a3c.py # Asynchronous Advantage Actor-Critic
│   └── ... # Other algorithms
├── helpers/ # Utility functions and components
│   ├── envs.py # Environment wrappers
│   ├── buffers.py # Replay buffer implementations
│   ├── networks.py # Neural network architectures
│   └── utils/
│       ├── monitoring/ # Logging and visualization utilities
│       └── ... # Other utilities
├── Dockerfile.cpu # CPU container definition
├── Dockerfile.gpu # GPU container definition
└── setup.py # Package installation
```

## Algorithm Comparison

| Algorithm | Type                | On/Off Policy | Action Space        | Key Features                                                   |
| --------- | ------------------- | ------------- | ------------------- | -------------------------------------------------------------- |
| PG        | Policy Optimization | On-policy     | Discrete/Continuous | Simple, high variance, no value function                       |
| A2C       | Actor-Critic        | On-policy     | Discrete/Continuous | Synchronous, advantage function, value baseline                |
| A3C       | Actor-Critic        | On-policy     | Discrete/Continuous | Asynchronous, multiple workers, parallel training              |
| TRPO      | Policy Optimization | On-policy     | Discrete/Continuous | Trust region constraint, monotonic improvement guarantee       |
| PPO       | Policy Optimization | On-policy     | Discrete/Continuous | Clipped objective, stable updates, sample efficient            |
| DDPG      | Actor-Critic        | Off-policy    | Continuous          | Deterministic policy, experience replay, target networks       |
| TD3       | Actor-Critic        | Off-policy    | Continuous          | Twin critics, delayed policy updates, target smoothing         |
| SAC       | Actor-Critic        | Off-policy    | Continuous          | Entropy maximization, stochastic policy, temperature parameter |

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- OpenAI Gym/Gymnasium for providing the reinforcement learning environments
- PyTorch team for the deep learning framework
- Original papers for each algorithm implementation
