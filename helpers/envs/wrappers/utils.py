import torch 
from gymnasium import Wrapper
from gymnasium.vector import VectorWrapper
from .numpy_to_torch import NumpyToTorchVector, NumpyToTorch


def is_wrapped(env, wrapper_class):
    """
    Check if a gym environment is wrapped with a specific wrapper class.

    Args:
        env (gym.Env): The gym environment to check.
        wrapper_class (type): The wrapper class to check for.

    Returns:
        bool: True if the environment is wrapped with the specified wrapper class, False otherwise.
    """
    current_env = env
    while isinstance(current_env, Wrapper):
        if isinstance(current_env, wrapper_class):
            return True
        current_env = current_env.env  # Unwrap one level
    return False


def is_vector_wrapped(env, wrapper_class):
    """
    Check if a vectorized environment is wrapped with a specific wrapper class.

    Args:
        env (VectorEnv): The vectorized environment to check.
        wrapper_class (type): The wrapper class to check for.

    Returns:
        bool: True if the environment is wrapped with the specified wrapper class, False otherwise.
    """
    current_env = env
    while isinstance(current_env, VectorWrapper):
        if isinstance(current_env, wrapper_class):
            return True
        current_env = current_env.env  # Unwrap one level
    return False

def ensure_numpytotorch_wrapper(env, device: str | torch.device = "cpu"):
    """
    Ensure a gym environment is wrapped with the NumpyToTorch wrapper.

    Args:
        env (gym.Env): The gym environment to wrap.
        device (torch.device): The device for the NumpyToTorch wrapper.

    Returns:
        gym.Env: The wrapped environment.
    """
    if not is_wrapped(env, NumpyToTorch):
        env = NumpyToTorch(env, device=device)
    return env


def ensure_vector_numpytotorch_wrapper(env, device: str | torch.device = "cpu"):
    """
    Ensure a vectorized environment is wrapped with the NumpyToTorch wrapper.

    Args:
        env (VectorEnv): The vectorized environment to wrap.
        device (torch.device): The device for the NumpyToTorch wrapper.

    Returns:
        VectorEnv: The wrapped environment.
    """
    if not is_vector_wrapped(env, NumpyToTorchVector):
        env = NumpyToTorchVector(env, device=device)
    return env