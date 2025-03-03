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
from helpers.envs import make_sync_vec, AutoresetMode, SyncVectorEnv
from helpers.envs.dmc2gym import make_dmc

from replayBuffer import ReplayBufferNumpy, Transition


def weights_init_(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        nn.init.zeros_(m.bias)

def get_action_bias_scale(action_space) -> tuple[np.ndarray, np.ndarray]:
    """Return action bias and scale."""
    action_upper_bound: np.ndarray = action_space.high
    action_lower_bound: np.ndarray = action_space.low
    action_bias = (action_lower_bound + action_upper_bound) / 2.0
    action_scale = (action_upper_bound - action_lower_bound) / 2.0
    return action_bias, action_scale

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

def save_model(actor, qf1, qf2, logger, filename: str):
    """Save the current model state dictionaries to a file in logger.dir_name."""
    model_path = os.path.join(logger.dir_name, filename)
    torch.save((actor.state_dict(), qf1.state_dict(), qf2.state_dict()), model_path)
    print(f"model saved to {model_path}")

# ALGO LOGIC: initialize agent here:
class Critic(nn.Module):
    def __init__(self, env, use_weights_init=False):
        super().__init__()
        inputs_dim = env.single_observation_space.shape[0] + env.single_action_space.shape[0]
        self._critic = nn.Sequential(
            nn.Linear(inputs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        if use_weights_init:
            self.apply(weights_init_)

    def forward(self, x, a):
        assert x.shape[0] == a.shape[0]
        assert len(x.shape) == 2 and len(a.shape) == 2
        x = torch.cat([x, a], 1)
        x = self._critic(x)
        return x


class Actor(nn.Module):
    def __init__(self, env, use_weights_init=False):
        super().__init__()
        self.fc1 = nn.Linear(env.single_observation_space.shape[0], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, env.single_action_space.shape[0])

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
    # Consolidated environment selection.
    parser.add_argument('--env-type', type=str, choices=['gym', 'dmc'], default='gym',
                        help='Environment type: "gym" uses Gymnasium env id (e.g. Mujoco), '
                             'while "dmc" uses DM Control (specify domain and task names).')
    parser.add_argument('--env-id', type=str, default='Humanoid-v5',
                        help='Gymnasium environment ID (used if env-type is gym)')
    parser.add_argument('--domain-name', type=str, default='humanoid_CMU',
                        help='DM Control domain name (used if env-type is dmc)')
    parser.add_argument('--task-name', type=str, default='run',
                        help='DM Control task name (used if env-type is dmc)')
    parser.add_argument('--reward-scale', type=float, default=1.0, 
                        help='Reward scaling factor for DM Control')
    # Other training Hyperparameters
    parser.add_argument('--total-timesteps', type=int, default=int(1e6))
    parser.add_argument('--buffer-size', type=int, default=int(1e6))
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--learning-starts', type=int, default=int(25e3))
    parser.add_argument('--policy-lr', type=float, default=3e-4)
    parser.add_argument('--critic-lr', type=float, default=3e-4)
    parser.add_argument('--exploration-noise', type=float, default=0.1)
    parser.add_argument('--target-policy-noise', type=float, default=0.2)
    parser.add_argument('--noise-clip', type=float, default=0.5)
    parser.add_argument('--policy-frequency', type=int, default=2)
    parser.add_argument('--init-weights', type=bool, default=True)
    parser.add_argument('--eval-interval', type=int, default=10000)
    args = parser.parse_args()

    run_name = (
        f"lr{args.policy_lr}"
        f"_bs{args.batch_size}"
        f"_gamma{args.gamma}"
        f"_{int(time.time())}"
    )
    # Define a run identifier using either env_id or the DM Control spec.
    env_identifier = args.env_id if args.env_type == "gym" else f"{args.domain_name}-{args.task_name}"

    logger = Logger(run_name=run_name, env=env_identifier, algo="TD3")
    logger.add_run_command()
    logger.log_all_hyperparameters(vars(args))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

     # Environment setup:
    if args.env_type == 'dmc':
        def make_env():
            # Build and wrap the DM Control env with RecordEpisodeStatistics.
            return gym.wrappers.RecordEpisodeStatistics(
                make_dmc(
                    domain_name=args.domain_name,
                    task_name=args.task_name,
                    frame_skip=1,
                    from_pixels=False,
                    visualize_reward=False,
                    rendering="egl",
                    height=84,
                    width=84,
                    camera_id=0
                )
            )
        # For vectorized environments.
        envs = SyncVectorEnv([make_env], autoreset_mode=AutoresetMode.SAME_STEP)
        test_env = make_env()
    else:
        # For Gymnasium environment (e.g. Mujoco) using env_id.
        envs = make_sync_vec(args.env_id, num_envs=1,
                             wrappers=(gym.wrappers.RecordEpisodeStatistics,),
                             autoreset_mode=AutoresetMode.SAME_STEP)
        test_env = gym.make(args.env_id)

    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    action_bias, action_scale = get_action_bias_scale(envs.single_action_space)

    policy = Actor(envs, args.init_weights).to(device)
    qf1 = Critic(envs, args.init_weights).to(device)
    qf2 = Critic(envs, args.init_weights).to(device)
    target_q1, target_q2 = deepcopy(qf1), deepcopy(qf2)
    target_policy = deepcopy(policy)
    optim_q_fns = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.critic_lr)
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
                actions = policy(torch.as_tensor(obs, dtype=torch.float32, device=device))
                noise = torch.normal(0, args.exploration_noise, size=envs.single_action_space.shape, dtype=torch.float32, device=device)
                actions += noise
                actions = actions.cpu().detach().numpy()
                actions = np.clip(actions, -1.0, 1.0)
                env_actions = actions * action_scale + action_bias
        
        next_obs, rewards, terminations, truncations, infos = envs.step(env_actions)
        if args.env_type == 'dmc':
            rewards = rewards * args.reward_scale
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
        
            # simple hack as we have single env in our vector env anyways 
            # We remove this hacks now we going to have our own SyncVectorEnv with AutoresetMode.SAME_STEP
            # obs, _ =  envs.reset()

        if global_step > args.learning_starts:
            # Start training here
            data = buffer.sample(args.batch_size)

            ### Train critics
            with torch.no_grad():
                clipped_noise =  (torch.randn_like(data.actions) * args.target_policy_noise).clamp(
                    -args.noise_clip, args.noise_clip
                )
                next_actions = (target_policy(data.next_states) + clipped_noise).clamp(-1.0, 1.0)
                next_target_q1 = target_q1(data.next_states, next_actions)
                next_target_q2 = target_q2(data.next_states, next_actions)
                next_target_q = torch.min(next_target_q1, next_target_q2)
                q_target = data.rewards + args.gamma * next_target_q * (1 - data.dones)
            
            # calculate the q value
            q1_value = qf1(data.states, data.actions)
            q2_value = qf2(data.states, data.actions)
            q1_loss = 0.5 * F.mse_loss(q1_value, q_target) #  torch.mean((q_target - q1_value)**2.0) * 0.5
            q2_loss = 0.5 * F.mse_loss(q2_value, q_target) #   torch.mean((q_target - q2_value)**2.0) * 0.5
            q_loss = q1_loss + q2_loss # check if last result better, then we can be happy that way

            optim_q_fns.zero_grad()
            q_loss.backward()
            optim_q_fns.step()

            ### delayed policy train 
            if train_iterations % args.policy_frequency == 0:
                actions = policy(data.states)
                q1 = qf1(data.states, actions)
                q2 = qf2(data.states, actions)
                policy_loss = - torch.min(q1, q2).mean()

                optim_policy.zero_grad()
                policy_loss.backward()
                gradient = calculate_grad_norm(policy)
                optim_policy.step()
            
                # update the target network
                for param, target_param in zip(policy.parameters(), target_policy.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf1.parameters(), target_q1.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), target_q2.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
            

            train_iterations += 1

            if train_iterations % 100 == 0:
                logger.add_scalar("losses/qf1_values", q1_value.mean().item(), global_step)
                logger.add_scalar("losses/qf2_values", q2_value.mean().item(), global_step)
                logger.add_scalar("losses/qf1_loss", q1_loss.item(), global_step)
                logger.add_scalar("losses/qf2_loss", q2_loss.item(), global_step)
                logger.add_scalar("losses/qf_loss", q_loss.item(), global_step)
                logger.add_scalar("losses/actor_loss", policy_loss.item(), global_step)
                logger.add_scalar("losses/actor_norm", gradient, global_step)
                # print("SPS:", int(global_step / (time.time() - start_time)), global_step)
                logger.add_scalar("charts/FPS", int(global_step / (time.time() - start_time)), global_step)

            if (global_step + 1) % args.eval_interval == 0:
                eval_return = evaluate(test_env, policy, (action_scale, action_bias))
                if best_eval_return < eval_return:
                    best_eval_return = eval_return
                    save_model(policy, qf1, qf2, logger, "best-torch.model")
                logger.add_scalar("eval/returns", eval_return, global_step)
                logger.log_stdout()

    
    # ---- After training completes ----
    save_model(policy, qf1, qf2, logger, "torch.model")

    
    envs.close()
    logger.close()


