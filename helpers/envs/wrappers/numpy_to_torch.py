import gymnasium as gym
import numpy as np
import torch

from gymnasium.vector import VectorWrapper, VectorEnv
from gymnasium.core import ActType, ObsType
from typing import Any

# Mainly these classes implemented to support MPS, MPS only works with torch.float32

class NumpyToTorch(gym.Wrapper):
    """Converts NumPy-based environment inputs/outputs to PyTorch Tensors."""

    def __init__(self, env: gym.Env,  device: str | torch.device = "cpu"):
        super().__init__(env)
        self.device = torch.device(device)

    def reset(self, *, seed=None, options=None):
        """Resets the environment and returns PyTorch-based observations."""
        obs, info = self.env.reset(seed=seed, options=self._to_numpy(options) if options else None)
        return self._to_torch(obs), info

    def step(self, action):
        """Takes a step using a PyTorch action and returns PyTorch-based observations."""
        obs, reward, terminated, truncated, info = self.env.step(self._to_numpy(action))
        return (
            self._to_torch(obs),
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
        )

    def _to_torch(self, value):
        """Convert NumPy to preallocated PyTorch tensor."""
        if isinstance(value, np.ndarray):
            return torch.tensor(value, dtype=torch.float32, device=self.device)
        return value

    def _to_numpy(self, value):
        """Convert PyTorch tensor to NumPy array."""
        if isinstance(value, torch.Tensor):
            # Convert back to NumPy only when absolutely necessary
            return value.cpu().numpy()
        return value


class NumpyToTorchVector(VectorWrapper):
    """Wraps a NumPy-based vector environment to use PyTorch Tensors."""

    def __init__(self, env: VectorEnv, device: str | torch.device = "cpu"):
        super().__init__(env)
        self.device = torch.device(device)

    def step(self, actions: ActType) -> tuple[ObsType, Any, Any, Any, dict]:
        """Step environment with PyTorch-based actions."""
        # Convert PyTorch tensor actions to NumPy
        np_actions = self._to_numpy(actions)
        obs, reward, terminated, truncated, info = self.env.step(np_actions)

        # Convert results back to PyTorch tensors on MPS
        return (
            self._to_torch(obs),
            self._to_torch(reward),
            self._to_torch(terminated),
            self._to_torch(truncated),
            info,
        )

    def reset(
        self,
        *,
        seed: int | list[int] | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        """Reset environment and convert observations to PyTorch Tensors."""
        options = self._to_numpy(options) if options else None

        # Reset environment and convert results to PyTorch tensors
        obs, info = self.env.reset(seed=seed, options=options)
        return self._to_torch(obs), info

    def _to_torch(self, value, dtype=torch.float32):
        """Convert NumPy to PyTorch tensor using preallocated space."""
        if isinstance(value, np.ndarray):
            return torch.tensor(value, dtype=dtype, device=self.device)
        return value

    def _to_numpy(self, value):
        """Convert PyTorch tensor to NumPy array."""
        if isinstance(value, torch.Tensor):
            return value.cpu().numpy()
        return value
