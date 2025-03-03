

import numpy as np
import os
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
from gymnasium import core, spaces
from gymnasium.core import ObsType, RenderFrame, ActType
from dm_control import suite, manipulation


def _spec_to_box(spec, dtype) -> spaces.Box:
    # Convert DM Control specs to a gymnasium Box
    mins = [np.array(s.minimum) for s in spec]
    maxs = [np.array(s.maximum) for s in spec]
    low = np.concatenate(mins, axis=0).astype(dtype)
    high = np.concatenate(maxs, axis=0).astype(dtype)
    assert low.shape == high.shape
    return spaces.Box(low, high, dtype=dtype)

def _flatten_obs(obs: Dict[str, Any]) -> np.ndarray:
    """Flatten a dict of observations into a single vector."""
    obs_pieces = []
    for v in obs.values():
        flat = np.array([v]) if np.isscalar(v) else v.ravel()
        obs_pieces.append(flat)
    return np.concatenate(obs_pieces, axis=0)

class DMCWrapper(core.Env):
    """
    A Gymnasium-compatible wrapper for DeepMind Control Suite environments.
    
    This class adapts environments from the DM Control Suite to the Gymnasium API.
    It handles action denormalization, observation flattening (or pixel rendering), and 
    frame-skipping. It also defines the observation_space and action_space properties.
    """
    def __init__(
        self,
        domain_name: str,
        task_name: str,
        task_kwargs: Dict[str, Any] = {},
        environment_kwargs: Dict[str, Any] = {},
        rendering: str = "egl",
        height: int = 84,
        width: int = 84,
        camera_id: int = 0,
        from_pixels: bool = False,
        frame_skip: int = 1,
        visualize_reward: bool = False,
    ):
        """
        Initialize the DeepMind Control environment wrapper.
        
        Args:
            domain_name (str): The DM Control domain (e.g., "cartpole").
            task_name (str): The task name (e.g., "swingup").
            task_kwargs (dict, optional): Additional keyword arguments for task construction.
            environment_kwargs (dict, optional): Additional keyword arguments for environment construction.
            rendering (str, optional): Rendering backend ("egl", "glfw", or "osmesa"). Defaults to "egl".
            height (int, optional): Rendered image height (if from_pixels=True). Defaults to 84.
            width (int, optional): Rendered image width (if from_pixels=True). Defaults to 84.
            camera_id (int, optional): Camera id used for rendering. Defaults to 0.
            from_pixels (bool, optional): Return pixel observations instead of state vectors. Defaults to False.
            frame_skip (int, optional): Number of frames to accumulate per step. Defaults to 1.
            visualize_reward (bool, optional): Whether to visualize rewards. Defaults to False.
        
        Raises:
            AssertionError: If an unsupported rendering option is provided.
        """
        assert rendering in ["glfw", "egl", "osmesa"], f"Rendering '{rendering}' not supported."
        os.environ["MUJOCO_GL"] = rendering

        # Create task from DM Control suite
        if domain_name == "manipulation":
            self._env = manipulation.load(
                environment_name=task_name,
                seed=task_kwargs.get("random", 42),
            )
        else:
            self._env = suite.load(
                domain_name=domain_name,
                task_name=task_name,
                task_kwargs=task_kwargs,
                environment_kwargs=environment_kwargs,
                visualize_reward=visualize_reward,
            )

        # Gymnasium rendering settings
        self.render_mode = "rgb_array"
        self.render_height = height
        self.render_width = width
        self.render_camera_id = camera_id
        self._frame_skip = frame_skip
        self._from_pixels = from_pixels

        # Define the true action space using the DM Control spec,
        # and a normalized action space in [-1, 1] for the agent.
        self._true_action_space = _spec_to_box([self._env.action_spec()], np.float32)
        self._norm_action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=self._true_action_space.shape,
            dtype=np.float32,
        )

        # Create observation space
        if from_pixels:
            shape = (3, height, width)
            self._observation_space = spaces.Box(
                low=0, high=255, shape=shape, dtype=np.uint8
            )
        else:
            obs_spec = list(self._env.observation_spec().values())
            # print(obs_spec)
            # [Array(shape=(56,), dtype=dtype('float64'), name='joint_angles'), 
            #  Array(shape=(), dtype=dtype('float64'), name='head_height'), 
            #  Array(shape=(12,), dtype=dtype('float64'), name='extremities'), 
            #  Array(shape=(3,), dtype=dtype('float64'), name='torso_vertical'), 
            #  Array(shape=(3,), dtype=dtype('float64'), name='com_velocity'), 
            #  Array(shape=(62,), dtype=dtype('float64'), name='velocity')]
            # Compute the total number of elements in each observation element.
            observation_dim = sum(np.prod(s.shape) if len(s.shape) > 0 else 1 for s in obs_spec)
            self._observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(observation_dim,),
                dtype=np.float32,
            )
        
        self._action_space = self._norm_action_space
        self._visualize_reward = visualize_reward
        self._current_time_step = None

        self._true_low = self._true_action_space.low
        self._true_high = self._true_action_space.high
        self._norm_low = self._norm_action_space.low
        self._norm_high = self._norm_action_space.high
        self._scale = (self._true_high - self._true_low) / (self._norm_high - self._norm_low)
        self._offset = self._true_low - self._norm_low * self._scale

    
    def __getattr__(self, name):
        # Forward attribute lookups to the underlying DM Control environment
        return getattr(self._env, name)
    
    def _convert_action(self, action: np.ndarray) -> np.ndarray:
        # Convert normalized action to true DM Control action range
        return action * self._scale + self._offset  # Faster computation
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        assert self._action_space.contains(action), "Action out of range"
        # Denormalize action to true range
        true_action = self._convert_action(action)
        assert self._true_action_space.contains(true_action), "True action out of range"

        reward = 0.0
        for _ in range(self._frame_skip):
            time_step = self._env.step(true_action)
            reward += time_step.reward or 0.0
            if time_step.last():
                break
        self._current_time_step = time_step
        truncated = time_step.discount == 0
        terminated = time_step.last() and not truncated
        obs = self._get_obs()
        info = {"internal_state": self._env.physics.get_state().copy()}  # internal state prior to step
        info["discount"] = time_step.discount
        if terminated or truncated:
            info["TimeLimit.truncated"] = truncated
        return obs, float(reward), terminated, truncated, info
    
    def _get_obs(self) -> np.ndarray:
        if self._from_pixels:
            obs = self.render(
                height=self.render_height,
                width=self.render_width,
                camera_id=self.render_camera_id,
            )
            obs = np.transpose(obs, (2, 0, 1)).copy()
        else:
            obs = _flatten_obs(self._current_time_step.observation)
        return obs
    
    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[ObsType, Dict[str, Any]]:
        if seed is not None:
            self._env.task.random.seed(seed)
        time_step = self._env.reset()
        self._current_time_step = time_step
        obs = self._get_obs()
        info = {}
        return obs, info
    
    def render(self) -> np.ndarray:
        assert self.render_mode == "rgb_array", "Only rgb_array mode is supported."
        return self._env.physics.render(
            height=self.render_height,
            width=self.render_width,
            camera_id=self.render_camera_id,
        )
    
    @property
    def observation_space(self):
        return self._observation_space 

    @property
    def action_space(self):
        return self._action_space

def make_dmc(
    domain_name: str,
    task_name: str,
    frame_skip: int = 1,
    from_pixels: bool = False,
    visualize_reward: bool = False,
    rendering: str = "egl",
    height: int = 84,
    width: int = 84,
    camera_id: int = 0,
    environment_kwargs: Optional[Dict[str, Any]] = None,
    task_kwargs: Optional[Dict[str, Any]] = None,
) -> gym.Env:
    """
    Create a DeepMind Control environment adapted for Gymnasium.
    
    This helper function instantiates a DMCWrapper with the specified parameters.
    It applies a TimeLimit wrapper to enforce a maximum episode length for compatibility
    with Gymnasium interfaces.
    
    Args:
        domain_name (str): The DM Control domain name (e.g., "cartpole").
        task_name (str): The DM Control task name (e.g., "swingup").
        frame_skip (int, optional): Number of frames to skip per step. Defaults to 1.
        from_pixels (bool, optional): Determines whether the observations are pixel-based, 
                                      or state vectors. Defaults to False.
        visualize_reward (bool, optional): If True, visualizes the reward signal. Defaults to False.
        rendering (str, optional): The rendering backend ("egl", "glfw", "osmesa"). Defaults to "egl".
        height (int, optional): The height of the rendered image. Defaults to 84.
        width (int, optional): The width of the rendered image. Defaults to 84.
        camera_id (int, optional): The camera id to be used for rendering. Defaults to 0.
        environment_kwargs (Optional[Dict[str, Any]], optional): Additional arguments for environment creation.
        task_kwargs (Optional[Dict[str, Any]], optional): Additional arguments for task creation.
    
    Returns:
        gym.Env: A Gymnasium-compatible DeepMind Control environment wrapped with TimeLimit.
    """
        
    environment_kwargs = environment_kwargs or {}
    task_kwargs = task_kwargs or {}

    env = DMCWrapper(
        domain_name=domain_name,
        task_name=task_name,
        task_kwargs=task_kwargs,
        environment_kwargs=environment_kwargs,
        rendering=rendering,
        height=height,
        width=width,
        camera_id=camera_id,
        from_pixels=from_pixels,
        frame_skip=frame_skip,
        visualize_reward=visualize_reward,
    )
    # Add a time limit to ensure episodes have a max length
    env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
    return env