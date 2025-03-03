import math
from copy import deepcopy


import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from continuous_sac import Actor

def get_action_bias_scale(env_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Return action bias and scale."""
    action_space = gym.make(env_id).action_space
    action_upper_bound: np.ndarray = action_space.high
    action_lower_bound: np.ndarray = action_space.low
    action_bias = (action_lower_bound + action_upper_bound) / 2.0
    action_scale = (action_upper_bound - action_lower_bound) / 2.0
    return action_bias, action_scale

def get_state_action_dims(env_id: str) -> tuple[int, int]:
    """Return state and action dimension."""
    assert env_id in gym.registry
    env = gym.make(env_id)
    if "dm_control" in env_id:
        env = gym.wrappers.FlattenObservation(env)
    return env.observation_space.shape[0], env.action_space.shape[0]

def evaluate(env, actor, action_scale_bias, n_rollout=10):
    action_scale, action_bias = action_scale_bias
    episode_rewards = []
    for episode in range(n_rollout):
        state, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            state = torch.from_numpy(np.array(state, dtype=np.float32)).to(device)
            with torch.no_grad():
                action = actor.get_action(state, compute_log_pi=False)[0].cpu().numpy().reshape(-1)
                action = action * action_scale + action_bias

            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            state = next_state
            done = terminated or truncated
        
        episode_rewards.append(total_reward) # Store the total reward for the episode
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")
    
    # Calculate mean and standard deviation
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    print(f"Mean Reward: {mean_reward}, Standard Deviation: {std_reward}")

    return mean_reward, std_reward

if __name__ == "__main__":
    device = torch.device('cpu')
    env = gym.make('Humanoid-v5')
    actor = Actor(env).to(device)
    # actor_params, _, _ = torch.load('my_runs/Humanoid-v5_SAC_stableTANH_True_weights_False_1737117877/torch.model', map_location=device)
    # actor.load_state_dict(actor_params)
    # actor.eval()

    # action_bias, action_scale = get_action_bias_scale('Humanoid-v5')

    # evaluate(env, actor, (action_scale, action_bias))