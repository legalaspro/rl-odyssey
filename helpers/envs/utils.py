import numpy as np
from helpers import Logger  # Adjust import as needed

def process_vector_env_info(infos: dict, logger: Logger, global_step: int) -> None:
    """
    Process the episode information from infos and log the reward and length.

    Args:
        infos: Dictionary that contains the key "episode" or "final_info" (in case of SAME_STEP autoreset mode)
              with episode-specific info and an optional "_episode" flag. 
        logger: Logger instance to record metrics.
        global_step: Current global step.
    """
    source_info = infos.get("episode") or infos.get("final_info", {})
    done_envs = np.where(source_info.get("_episode", []))[0]
    
    for env_index in done_envs:
        env_specific_data = {
            key: value[env_index]
            for key, value in source_info.items()
            if isinstance(value, np.ndarray)
        }
        reward = env_specific_data.get('r')
        length = env_specific_data.get('l')
        
        if reward is not None:
            logger.add_scalar("charts/episodic_return", reward, global_step)
        if length is not None:
            logger.add_scalar("charts/episodic_length", length, global_step)
