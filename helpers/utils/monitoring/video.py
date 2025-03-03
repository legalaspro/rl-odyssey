import os
import torch
import imageio
import numpy as np
# from IPython.display import Video, display

def record_video(env, policy, out_directory, out_name, fps=30, min_reward=None):
    """
    Generate a replay video of the agent and save it.
    :param env: Environment to record.
    :param policy: Policy used to determine actions.
    :param out_directory: Path to save the video.
    :param fps: Frames per second.
    :param min_reward: Record only episodes with reward >= min_reward
    """
    images = []
    obs, _ = env.reset()
    total_reward = 0
    episode_length = 0
    num_saved_episodes = 0

    while True:
        state = torch.tensor(obs, dtype=torch.float32)
        if len(state) > 2:
            state = state.unsqueeze(0)
        
        # Select the action using the trained model
        action = policy(state).squeeze(0)
        # clip may not be needed here
        selected_actions = action.detach().numpy().clip(
                env.action_space.low,
                env.action_space.high)
        
        obs, reward, terminated, truncated, _ = env.step(selected_actions)
        total_reward += reward
        images.append(env.render())
        episode_length += 1

        if terminated or truncated:
            obs, _ = env.reset()
            print(total_reward)
            if min_reward is None or total_reward >= min_reward:
                num_saved_episodes += 1
            else:
                images = images[:-episode_length]  # Remove unsatisfactory episode
            total_reward = 0
            episode_length = 0
            if num_saved_episodes == 2:  # Save 2 episodes
                break
    
    # Save the video
    video_path = os.path.join(out_directory, out_name)
    imageio.mimsave(video_path, images, fps=fps)
    return video_path

# def display_video(video_path):
#     """Display the video inside a Jupyter notebook."""
#     display(Video(video_path, embed=True, width=640, height=480))

# def record_and_display(env, policy, out_directory, out_name, fps=30, min_reward=None):
#     """Record the video  and then display inside a Jupyter notebook."""
#     video_path = record_video(env, policy, out_directory, out_name, fps, min_reward)
#     display_video(video_path)