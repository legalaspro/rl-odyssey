import gymnasium as gym
from gymnasium.envs.registration import register
from .dmc_wrapper import DMCWrapper

def make(
    domain_name: str,
    task_name: str,
    visualize_reward: bool = True,
    from_pixels: bool = False,
    height: int = 84,
    width: int = 84,
    camera_id: int = 0,
    frame_skip: int = 1,
    episode_length: int = 1000,
    task_kwargs: dict = None,
    environment_kwargs: dict = None,
) -> gym.Env:
    """
    Register and create a DM Control environment wrapped for Gymnasium.

    Args:
        domain_name (str): The DM Control domain name.
        task_name (str): The task name.
        visualize_reward (bool, optional): Whether to visualize the reward. Defaults to True.
        from_pixels (bool, optional): Whether to use pixel observations. Defaults to False.
        height (int, optional): Render height. Defaults to 84.
        width (int, optional): Render width. Defaults to 84.
        camera_id (int, optional): Render camera id. Defaults to 0.
        frame_skip (int, optional): Number of frames to skip. Defaults to 1.
        episode_length (int, optional): Maximum episode length. Defaults to 1000.
        task_kwargs (dict, optional): Additional task arguments. Defaults to {}.
        environment_kwargs (dict, optional): Additional environment arguments. Defaults to {}.

    Returns:
        gym.Env: The registered and created Gymnasium environment.
    """
    task_kwargs = task_kwargs or {}
    environment_kwargs = environment_kwargs or {}
    # Generate a unique environment ID
    env_id = f"dmc_{domain_name}_{task_name}-v1"

    if from_pixels:
        assert not visualize_reward, "Cannot use visualize_reward when learning from pixels."

    # Calculate maximum episode steps (adjust for frame skipping)
    max_episode_steps = (episode_length + frame_skip - 1) // frame_skip

    # Register environment if it's not already registered
    if env_id not in gym.envs.registration.registry:
        register(
            id=env_id,
            entry_point=lambda: DMCWrapper(
                domain_name=domain_name,
                task_name=task_name,
                task_kwargs=task_kwargs,
                environment_kwargs=environment_kwargs,
                visualize_reward=visualize_reward,
                from_pixels=from_pixels,
                height=height,
                width=width,
                camera_id=camera_id,
                frame_skip=frame_skip,
            ),
            max_episode_steps=max_episode_steps,
        )

    return gym.make(env_id)