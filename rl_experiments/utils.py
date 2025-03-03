import numpy as np

def get_action_bias_scale(action_space) -> tuple[np.ndarray, np.ndarray]:
    """Return action bias and scale."""
    action_upper_bound: np.ndarray = action_space.high
    action_lower_bound: np.ndarray = action_space.low
    action_bias = (action_lower_bound + action_upper_bound) / 2.0
    action_scale = (action_upper_bound - action_lower_bound) / 2.0
    return action_bias, action_scale


# def get_state_action_dims(env_id: str) -> tuple[int, int]:
#     """Return state and action dimension."""
#     assert env_id in gym.registry
#     env = gym.make(env_id)
#     if "dm_control" in env_id:
#         env = gym.wrappers.FlattenObservation(env)
#     return env.observation_space.shape[0], env.action_space.shape[0]