import torch
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import NormalizeObservation

def eval_policy(env, policy, num_episodes=10):
    """Evaluate the policy over a number of episodes."""
    # Store rewards for each episode
    episode_rewards = []
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            state = torch.tensor(obs, dtype=torch.float32)
            if len(state) > 2:
                state = state.unsqueeze(0)
            # Select the action using the trained model
            action = policy(state).squeeze(0)

            # clip may not be needed here
            selected_actions = action.detach().numpy().clip(
                   env.action_space.low,
                   env.action_space.high)
           
            # Step the environment
            obs, reward, terminated, truncated, info = env.step(selected_actions)
            total_reward += reward
            done = terminated or truncated
           
        episode_rewards.append(total_reward) # Store the total reward for the episode
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

    # Calculate mean and standard deviation
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    print(f"Mean Reward: {mean_reward}, Standard Deviation: {std_reward}")

    return mean_reward, std_reward


def create_evaluation_env_model(env_id, model_class, checkpoint_path, device="cpu", **kwargs):
    """
    Set up an evaluation environment and load the trained policy model.
    :param env_id: Gymnasium environment ID
    :param model_class: The class of the model (e.g., TD3Actor, SACActor)
    :param checkpoint_path: Path to the saved model checkpoint
    :param device: Torch device for evaluation
    :param kwargs: Additional arguments for the model factory
    :return: Tuple of (environment, policy model)
    """
    env = gym.make(env_id, render_mode="rgb_array")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Add observation normalization if present
    if "obs_rms" in checkpoint:
        env = NormalizeObservation(env)
        env.update_running_mean = False
        mean = np.mean([obs_rms.mean for obs_rms in checkpoint["obs_rms"]], axis=0)
        var = np.mean([obs_rms.var for obs_rms in checkpoint["obs_rms"]], axis=0)
        env.obs_rms.mean = mean
        env.obs_rms.var = var

    action_low = env.action_space.low 
    action_high = env.action_space.high 
    # Initialize the policy model with dynamic arguments
    policy = model_class(env.observation_space.shape, env.action_space.shape[0], action_low, action_high, **kwargs).to(device)
    policy.load_state_dict(checkpoint["actor"])
    policy.eval()
    return env, policy