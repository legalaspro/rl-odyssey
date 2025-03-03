import numpy as np
import torch
import io
from collections import namedtuple


Transition = namedtuple(
    "Transition", ("states", "actions", "rewards", "next_states", "dones")
)

class ReplayBufferNumpy:
    """    
    Replay buffer used in off-policy algorithms like DDPG/SAC/TD3.
    
    :param obs_dim: Observation dimensions
    :param action_dim: Actions dimensions
    :param n_envs: Number of parallel environments 
    :param size: Max number of elements in the buffer
    :param device: Device (cpu, cuda, ...) on which the code should be run. 
    """
    def __init__(self,
                 obs_dim,
                 action_dim,
                 n_envs: int = 1,
                 size: int = 1e6,
                 device: torch.device = torch.device("cpu")):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        self.n_envs = n_envs

        self.pos = 0
        self.size = 0

        # Adjust buffer size 
        self.max_size = max(int(size // n_envs), 1)

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_dim) == 1 else np.uint8

        # Setup the data storage 
        
        self.obs = np.zeros((self.max_size, self.n_envs, *self.obs_dim), dtype=obs_dtype)
        self.next_obs = np.zeros((self.max_size, self.n_envs, *self.obs_dim), dtype=obs_dtype)
        self.actions = np.zeros((self.max_size, self.n_envs, *self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.max_size, self.n_envs), dtype=np.float32)
        self.terminates = np.zeros((self.max_size, self.n_envs), dtype=np.float32)

    def add(self,
            obs: np.ndarray,
            next_obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            terminates: np.ndarray
            ):
        np.copyto(self.obs[self.pos], obs)
        np.copyto(self.next_obs[self.pos], next_obs)
        np.copyto(self.actions[self.pos], action)
        np.copyto(self.rewards[self.pos], reward)
        np.copyto(self.terminates[self.pos], terminates)

        self.pos = (self.pos + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self,
               batch_size:int = 32):
        """
        Sample elements from the replay buffer.
        
        :param batch_size: Number of elements to sample
        """
        batch_indices = np.random.randint(0, self.size, size=batch_size)
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=batch_size) if self.n_envs > 1 \
            else np.zeros(batch_size, dtype=np.int64)
        # in the end we return exactly batch_size transitions collected even from different agents
        # Gather indices for the first two dimensions
        indices = (batch_indices, env_indices)
        
        
        obses=torch.as_tensor(self.obs[indices], dtype=torch.float32, device=self.device)
        next_obses=torch.as_tensor(self.next_obs[indices], dtype=torch.float32, device=self.device)
        actions=torch.as_tensor(self.actions[indices], dtype=torch.float32, device=self.device)
        rewards=torch.as_tensor(self.rewards[indices], dtype=torch.float32, device=self.device).unsqueeze(-1)
        # Only use dones that are not due to timeouts
        dones=torch.as_tensor(self.terminates[indices], dtype=torch.float32, device=self.device).unsqueeze(-1)
        
        return Transition(obses, actions, rewards, next_obses, dones)
        

    def _to_torch(self, data):
        return torch.tensor(data, dtype=torch.float32, device=self.device)
    
    def __len__(self):
        return self.size
    
    #
    # Internal helpers for dumping/loading via np.savez_compressed
    #
    def _dump_to_npz(self, file_obj):
        """Write the replay buffer data + metadata to a file-like object."""
        np.savez_compressed(
            file_obj,
            obs=self.obs,
            next_obs=self.next_obs,
            actions=self.actions,
            rewards=self.rewards,
            terminates=self.terminates,
            pos=self.pos,
            size=self.size,
            max_size=self.max_size,
            n_envs=self.n_envs
        )

    def _load_from_npz(self, file_obj):
        """Load the replay buffer data + metadata from a file-like object."""
        data = np.load(file_obj)
        self.obs = data["obs"]
        self.next_obs = data["next_obs"]
        self.actions = data["actions"]
        self.rewards = data["rewards"]
        self.terminates = data["terminates"]
        self.pos = int(data["pos"])
        self.size = int(data["size"])
        self.max_size = int(data["max_size"])
        self.n_envs = int(data["n_envs"])
    
    #
    # Public methods for file-based saving/loading
    #
    def save_as_numpy(self, file_path: str):
        """Save the replay buffer to a compressed .npz file on disk."""
        with open(file_path, "wb") as f:
            self._dump_to_npz(f)
        print(f"Replay buffer saved to {file_path}")

    def load_from_numpy(self, file_path: str):
        """Load the replay buffer from a compressed .npz file on disk."""
        with open(file_path, "rb") as f:
            self._load_from_npz(f)
        print(f"Replay buffer loaded from {file_path}")

    #
    # Public methods for in-memory (bytes) saving/loading
    #
    def save_as_bytes(self) -> bytes:
        """
        Serialize the replay buffer to an in-memory bytes object.
        Useful for storing in a single PyTorch checkpoint file.
        """
        buf = io.BytesIO()
        self._dump_to_npz(buf)
        return buf.getvalue()

    def load_from_bytes(self, replay_bytes: bytes):
        """
        Load the replay buffer data from a bytes object
        (the counterpart to save_as_bytes).
        """
        buf = io.BytesIO(replay_bytes)
        self._load_from_npz(buf)
        print("Replay buffer loaded from bytes")