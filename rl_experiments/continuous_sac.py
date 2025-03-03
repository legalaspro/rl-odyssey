import argparse
import os
import random
import time
import math
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

def evaluate(env, actor, action_scale_bias, n_rollout=10):
    action_scale, action_bias = action_scale_bias
    tot_rw = 0
    for _ in range(n_rollout):
        state, _ = env.reset()
        done = False
        while not done:
            state = torch.from_numpy(np.array(state, dtype=np.float32)).to(device)
            with torch.no_grad():
                action = actor.get_action(state, compute_log_pi=False)[0].cpu().numpy().reshape(-1)
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
        x = torch.cat([x, a], dim=1)
        x = self._critic(x)
        return x

LOG_STD_MAX = 2
LOG_STD_MIN = -20

class Actor(nn.Module):
    def __init__(self, env, use_stable_method=True, use_weights_init=False):
        super().__init__()
         
        self._actor = nn.Sequential(
            nn.Linear(env.single_observation_space.shape[0], 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * env.single_action_space.shape[0])
        )

        self.log2 = math.log(2)
        self.use_stable_method = use_stable_method

        if use_weights_init:
            self.apply(weights_init_)

            # For final layer with tanh, use a small initialization:
            nn.init.xavier_uniform_(self._actor[4].weight, gain=1)
            nn.init.zeros_(self._actor[4].bias)


    def forward(self, x):
        mean, log_std = self._actor(x).chunk(2, dim=-1)
        return mean, log_std

    def get_action(self, x, compute_log_pi=True, deterministic=False):
        mean, log_std = self(x)
        
        if deterministic:
            return torch.tanh(mean), None
        
        log_std = torch.clamp(log_std, min=LOG_STD_MIN, max=LOG_STD_MAX)

        normal = torch.distributions.Normal(mean, log_std.exp())
        arctanh_action = normal.rsample()
        squashed_action = torch.tanh(arctanh_action)

        if not compute_log_pi:
            return squashed_action, None
        
        # Enforcing Action Bound
        if self.use_stable_method:
            log_prob = normal.log_prob(arctanh_action).sum(-1, keepdim=True)
            log_prob = log_prob - (2*(self.log2 - arctanh_action - F.softplus(-2 * arctanh_action))).sum(-1, keepdim=True)
        else: 
            log_prob = normal.log_prob(arctanh_action).sum(-1, keepdim=True)
            log_prob = log_prob - torch.log(1 - squashed_action.pow(2) + 1e-6).sum(-1, keepdim=True)     

        assert len(log_prob.shape) == 2 and len(squashed_action.shape) == 2
        return squashed_action, log_prob
    
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
    # Other hyperparameters
    parser.add_argument('--total-timesteps', type=int, default=int(1e6))
    parser.add_argument('--buffer-size', type=int, default=int(1e6))
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--learning-starts', type=int, default=int(25e3))
    parser.add_argument('--policy-lr', type=float, default=3e-4)
    parser.add_argument('--critic-lr', type=float, default=3e-4)
    parser.add_argument('--autotune', action='store_true', default=True,
                   help='Enable temperature auto-tuning')
    parser.add_argument('--no-autotune', dest='autotune', action='store_false',
                   help='Disable temperature auto-tuning')
    parser.add_argument('--init-temperature', type=float, default=0.05)
    parser.add_argument('--stable-tanh-squash', type=bool, default=True)
    parser.add_argument('--init-weights', type=bool, default=True)
    parser.add_argument('--eval-interval', type=int, default=10000)
    args = parser.parse_args()

    run_name = f"autotune{args.autotune}_stableTANH{args.stable_tanh_squash}_weights{args.init_weights}_{int(time.time())}"
    # Define a run identifier using either env_id or the DM Control spec.
    env_identifier = args.env_id if args.env_type == "gym" else f"{args.domain_name}-{args.task_name}"

    logger = Logger(run_name=run_name, env=env_identifier, algo="SAC")
    logger.add_run_command()
    logger.log_all_hyperparameters(vars(args))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

     # Environment setup:
    if args.env_type == 'dmc':
        def make_env():
            # Build and wrap the DM Control env with RecordEpisodeStatistics.
            env = make_dmc(
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
            env = gym.wrappers.RecordEpisodeStatistics(env)
            return env
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

    actor = Actor(envs, 
                  use_stable_method=args.stable_tanh_squash, 
                  use_weights_init=args.init_weights).to(device)
    qf1 = Critic(envs,
                use_weights_init=args.init_weights).to(device)
    qf2 = Critic(envs,
                use_weights_init=args.init_weights).to(device)
    target_q1, target_q2 = deepcopy(qf1), deepcopy(qf2)
    optim_q_fns = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.critic_lr)
    optim_actor =  optim.Adam(actor.parameters(), lr=args.policy_lr)

    if args.autotune:
        target_entropy = -np.prod(envs.single_action_space.shape)
        log_alpha = torch.log(args.init_temperature * torch.ones(1, device=device)).requires_grad_(True)
        optim_alpha = optim.Adam([log_alpha], lr=args.critic_lr)
        alpha = log_alpha.exp().item()
    else:
        alpha = args.init_temperature

    buffer = ReplayBufferNumpy(
        obs_dim=envs.single_observation_space.shape,
        action_dim=envs.single_action_space.shape,
        n_envs=1,
        size=args.buffer_size,
        device=device,
    )

    start_time = time.time()
    global_step = 0
    best_eval_return = -np.inf

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset()
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            env_actions = envs.action_space.sample()
            actions = env_actions / action_scale - action_bias
        else:
            with torch.no_grad():
                actions, _  = actor.get_action(
                    torch.from_numpy(np.array(obs, dtype=np.float32)).to(device),
                    compute_log_pi=False
                )
                actions = actions.cpu().numpy()
                env_actions = actions * action_scale + action_bias
        
        next_obs, rewards, terminations, truncations, infos  = envs.step(env_actions)
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

        if global_step > args.learning_starts:
            # Start training here 
            data = buffer.sample(args.batch_size)
            ### Train critics
            with torch.no_grad():
                next_actions, next_log_prob  = actor.get_action(data.next_states, compute_log_pi=True)
                next_target_q1 = target_q1(data.next_states, next_actions)
                next_target_q2 = target_q2(data.next_states, next_actions)
                next_target_q = torch.minimum(next_target_q1, next_target_q2)
                next_target_q = next_target_q - alpha * next_log_prob
                q_target = data.rewards + args.gamma * next_target_q * (1 - data.dones)
            
            # calculate q value 
            q1_values = qf1(data.states, data.actions)
            q2_values = qf2(data.states, data.actions)
            q1_loss = 0.5 * F.mse_loss(q1_values, q_target) # torch.mean((q_target - q1_values)**2.0) * 0.5
            q2_loss = 0.5 * F.mse_loss(q2_values, q_target) # torch.mean((q_target - q2_values)**2.0) * 0.5
            q_loss = q1_loss + q2_loss # check if last result better, then we can be happy that way

            optim_q_fns.zero_grad()
            q_loss.backward()
            optim_q_fns.step()

            #### Train actor and then alpha 
            actions, log_probs = actor.get_action(data.states, compute_log_pi=True)
            q1 = qf1(data.states, actions)
            q2 = qf2(data.states, actions)
            q_value = torch.minimum(q1, q2)            
            actor_loss = torch.mean(alpha*log_probs - q_value)

            optim_actor.zero_grad()
            actor_loss.backward()
            optim_actor.step()

            if args.autotune:
                with torch.no_grad():
                    _, log_probs = actor.get_action(data.states, compute_log_pi=True)
                    log_probs = log_probs.mean()
                # try detach everything here
                alpha_loss = log_alpha.exp()*(-log_probs.detach() - target_entropy)
                entropy = -log_probs.detach()

                optim_alpha.zero_grad()
                alpha_loss.backward()
                optim_alpha.step()
                alpha = log_alpha.exp().item()  
            
            #### Update the target funcs
            with torch.no_grad():
                for param, target_param in zip(qf1.parameters(), target_q1.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), target_q2.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
            


            if global_step % 100 == 0:
                logger.add_scalar("losses/qf1_values", q1_values.mean().item(), global_step)
                logger.add_scalar("losses/qf2_values", q2_values.mean().item(), global_step)
                logger.add_scalar("losses/qf1_loss", q1_loss.item(), global_step)
                logger.add_scalar("losses/qf2_loss", q2_loss.item(), global_step)
                logger.add_scalar("losses/qf_loss", q_loss.item(), global_step)
                logger.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                # print("FPS:", int(global_step / (time.time() - start_time)), global_step)
                logger.add_scalar("charts/FPS", int(global_step / (time.time() - start_time)), global_step)
                if args.autotune:
                    logger.add_scalar("losses/entropy", entropy.item(), global_step)
                    logger.add_scalar("losses/alpha", log_alpha.exp().item(), global_step)
                    logger.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)
            
            if (global_step + 1) % args.eval_interval == 0:
                eval_return = evaluate(test_env, actor, (action_scale, action_bias))
                if best_eval_return < eval_return:
                    best_eval_return = eval_return
                    save_model(actor, qf1, qf2, logger, "best-torch.model")
                logger.add_scalar("eval/returns", eval_return, global_step)
                logger.log_stdout()

    # ---- After training completes ----
    save_model(actor, qf1, qf2, logger, "torch.model")

    
    envs.close()
    logger.close()
            




