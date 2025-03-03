import argparse
import os
import time
from copy import deepcopy


import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from helpers import Logger
from helpers.envs import make_sync_vec, AutoresetMode

from replayBuffer import ReplayBufferNumpy, Transition


def weights_init_(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        nn.init.zeros_(m.bias)

def get_action_bias_scale(env_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Return action bias and scale."""
    action_space = gym.make(env_id).action_space
    action_upper_bound: np.ndarray = action_space.high
    action_lower_bound: np.ndarray = action_space.low
    action_bias = (action_lower_bound + action_upper_bound) / 2.0
    action_scale = (action_upper_bound - action_lower_bound) / 2.0
    return action_bias, action_scale

def sync_normalization_stats(test_env, envs):
    """Synchronize normalization stats for test environment based on training environments."""
    
    # Average normalization stats from training vectorized environments
    all_obs_rms = [e.obs_rms for e in envs.envs]
    mean = sum(obs_rms.mean for obs_rms in all_obs_rms) / len(all_obs_rms)
    var = sum(obs_rms.var for obs_rms in all_obs_rms) / len(all_obs_rms)

    # Set test environment stats before evaluation
    test_env.obs_rms.mean = mean
    test_env.obs_rms.var = var

    return test_env

def calculate_grad_norm(neural_network: nn.Module) -> float:
    """Calculate grad norm."""
    total_norm = 0.0
    for param in neural_network.parameters():
        if param.grad is not None:
            total_norm += torch.norm(param.grad.data, p=2)
    return total_norm.item()

def evaluate(env, actor, action_scale_bias, n_rollout=10):
    action_scale, action_bias = action_scale_bias
    tot_rw = 0
    for _ in range(n_rollout):
        state, _ = env.reset()
        done = False
        while not done:
            state = torch.from_numpy(np.array(state, dtype=np.float32)).to(device)
            with torch.no_grad():
                action = actor(state).cpu().numpy().reshape(-1)
                action = action * action_scale + action_bias

            next_state, reward, terminated, truncated, _ = env.step(action)
            tot_rw += reward
            state = next_state
            done = terminated or truncated
    return tot_rw / n_rollout

def save_model(env, actor, qf, logger, filename: str):
    """Save the current model state dictionaries and normalization info to a file in logger.dir_name."""
    model_path = os.path.join(logger.dir_name, filename)
    # Gather normalization statistics from each environment
    all_obs_rms = [env.obs_rms for env in env.envs]
    
    checkpoint = {
        'actor_state_dict': actor.state_dict(),
        'qf_state_dict': qf.state_dict(),
        'obs_rms': all_obs_rms,
    }
    torch.save(checkpoint, model_path)
    print(f"model saved to {model_path}")

# ALGO LOGIC: initialize agent here:
class Critic(nn.Module):
    def __init__(self, env, use_weights_init=False):
        super().__init__()
        self.fc1 = nn.Linear(env.single_observation_space.shape[0], 400)
        self.fc2 = nn.Linear(env.single_action_space.shape[0] + 400, 300)
        self.fc3 = nn.Linear(300, 1)

        if use_weights_init:
            self.apply(weights_init_)

    def forward(self, x, a):
        assert x.shape[0] == a.shape[0]
        assert len(x.shape) == 2 and len(a.shape) == 2
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(torch.cat([x, a], 1)))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, env, use_weights_init=False):
        super().__init__()

        self.fc1 = nn.Linear(env.single_observation_space.shape[0], 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc_mu = nn.Linear(300, env.single_action_space.shape[0])

        if use_weights_init:
            self.apply(weights_init_)
            # For final layer with tanh, use a small initialization:
            nn.init.xavier_uniform_(self.fc_mu.weight, gain=1)
            nn.init.zeros_(self.fc_mu.bias)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--env-id', type=str, default='Humanoid-v5')
    parser.add_argument('--total-timesteps', type=int, default=int(1e6))
    parser.add_argument('--buffer-size', type=int, default=int(1e6))
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005) 
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--learning-starts', type=int, default=int(25e3))
    parser.add_argument('--policy-lr', type=float, default=3e-4)
    parser.add_argument('--critic-lr', type=float, default=3e-4)
    parser.add_argument('--exploration-noise', type=float, default=0.1)
    parser.add_argument('--init-weights', type=bool, default=True)
    parser.add_argument('--eval-interval', type=int, default=10000)
    args = parser.parse_args()


    run_name = f"weights{args.init_weights}_{int(time.time())}"

    logger = Logger(run_name=run_name, env=args.env_id, algo="DDPG")
    logger.add_run_command()
    logger.log_all_hyperparameters(vars(args))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # env setup 
    envs = make_sync_vec(args.env_id, 
                    num_envs=1, 
                    wrappers=(gym.wrappers.RecordEpisodeStatistics, gym.wrappers.NormalizeObservation, ),
                    autoreset_mode=AutoresetMode.SAME_STEP)
    
    test_env = gym.make(args.env_id)
    test_env = gym.wrappers.NormalizeObservation(test_env)
    test_env._update_running_mean = False
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    action_bias, action_scale = get_action_bias_scale(args.env_id)

    policy = Actor(envs, args.init_weights).to(device)
    qf = Critic(envs, args.init_weights).to(device)
    target_qf = deepcopy(qf)
    target_policy = deepcopy(policy)

    optim_qf = optim.Adam(qf.parameters(), lr=args.critic_lr)
    optim_policy = optim.Adam(policy.parameters(), lr=args.policy_lr)

    buffer = ReplayBufferNumpy(
        obs_dim=envs.single_observation_space.shape,
        action_dim=envs.single_action_space.shape,
        n_envs=1,
        size=args.buffer_size,
        device=device,
    )

    start_time = time.time()
    global_step = 0
    train_iterations = 0
    best_eval_return = -np.inf

    obs, _ = envs.reset()
    for global_step in range(args.total_timesteps):
        if global_step < args.learning_starts:
            env_actions = envs.action_space.sample()
            actions = env_actions / action_scale - action_bias
        else:
            with torch.no_grad():
                actions = policy(torch.from_numpy(np.array(obs, dtype=np.float32)).to(device))
                # this noise is better then using OU Noise 
                # https://spinningup.openai.com/en/latest/algorithms/ddpg.html#exploration-vs-exploitation
                noise = torch.normal(0, args.exploration_noise, size=envs.single_action_space.shape, dtype=torch.float32, device=device)
                actions += noise
                actions = actions.cpu().detach().numpy()
                actions = np.clip(actions, -1.0, 1.0)
                env_actions = actions * action_scale + action_bias
        
        next_obs, rewards, terminations, truncations, infos = envs.step(env_actions)
        buffer.add(obs, next_obs, actions, rewards, terminations)

        obs = next_obs

        # updated for the AutoresetMode.SAME_STEP, info now in final_info
        if "final_info" in infos:
            source_info = infos["final_info"]
            done_envs = np.where(source_info.get("_episode", []))[0]
            for env_index in done_envs:
                env_specific_data = {
                    key: value[env_index]
                    for key, value in source_info["episode"].items()
                    if isinstance(value, np.ndarray)
                }
                reward = env_specific_data['r']
                length = env_specific_data['l']
                logger.add_scalar("charts/episodic_return", reward, global_step)
                logger.add_scalar("charts/episodic_length", length, global_step)
        
        if global_step > args.learning_starts:
            # Start training here
            data = buffer.sample(args.batch_size)

            ### Train critics
            with torch.no_grad():
                next_actions = target_policy(data.next_states)
                next_target_qf = target_qf(data.next_states, next_actions)
                q_target = data.rewards + args.gamma * next_target_qf * (1 - data.dones)
            
            # calculate the q value
            q_value = qf(data.states, data.actions)
            q_loss = F.mse_loss(q_value, q_target)

            optim_qf.zero_grad()
            q_loss.backward()
            optim_qf.step()

            policy_loss = - qf(data.states, policy(data.states)).mean()

            optim_policy.zero_grad()
            policy_loss.backward()
            gradient = calculate_grad_norm(policy)
            optim_policy.step()

            # update the target network
            for param, target_param in zip(policy.parameters(), target_policy.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
            for param, target_param in zip(qf.parameters(), target_qf.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            train_iterations += 1

            if train_iterations % 100 == 0:
                logger.add_scalar("losses/qf1_values", q_value.mean().item(), global_step)
                logger.add_scalar("losses/qf_loss", q_loss.item(), global_step)
                logger.add_scalar("losses/actor_loss", policy_loss.item(), global_step)
                logger.add_scalar("losses/actor_norm", gradient, global_step)
                # print("SPS:", int(global_step / (time.time() - start_time)), global_step)
                logger.add_scalar("charts/FPS", int(global_step / (time.time() - start_time)), global_step)

            if (global_step + 1) % args.eval_interval == 0:
                sync_normalization_stats(test_env, envs)
                eval_return = evaluate(test_env, policy, (action_scale, action_bias))
                if best_eval_return < eval_return:
                    best_eval_return = eval_return
                    save_model(envs, policy, qf, logger, "best-torch.model")
                logger.add_scalar("eval/returns", eval_return, global_step)
                logger.log_stdout()

    
    # ---- After training completes ----
    save_model(envs, policy, qf, logger, "torch.model")

    
    envs.close()
    logger.close()


