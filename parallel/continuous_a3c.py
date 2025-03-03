import argparse
import os
import time

import gymnasium as gym
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp

from helpers import Logger
from shared_running_mean_std import SharedRunningMeanStd

# -- Helper Functions --
def weights_init_(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        nn.init.zeros_(m.bias)

def save_model(shared_obs_norm, actor_critic, logger, filename: str):
    """Save the current model state dictionaries and normalization info to a file in logger.dir_name."""
    model_path = os.path.join(logger.dir_name, filename)

    checkpoint = {
        'actor_critic_state_dict': actor_critic.state_dict(),
        'obs_rms':{
            'mean': shared_obs_norm.mean.clone().numpy(),
            'var': shared_obs_norm.var.clone().numpy(),
        }
    }
    torch.save(checkpoint, model_path)

def create_mujoco_env(env_id: str):
    """Create a Mujoco environment with some standard wrappers."""
    env = gym.make(env_id)
    env = gym.wrappers.NormalizeObservation(env)
    return env

# --- Actor-Critic Model ---
class ActorCritic(nn.Module):

    def __init__(self, 
                 env, 
                 fc_units=128):
        """Initialize parameters and build model."""
        super(ActorCritic, self).__init__()

        # Environment related constants
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.shape[0]

        # Action Scaling
        action_low = torch.tensor(env.action_space.low, dtype=torch.float32)
        action_high = torch.tensor(env.action_space.high, dtype=torch.float32)
        self.action_scale = (action_high - action_low) / 2.0
        self.action_bias = (action_high + action_low) / 2.0
        
        # Actor head
        self.fc_actor = nn.Sequential(
            nn.Linear(state_size, fc_units),
            nn.ReLU(),
            nn.Linear(fc_units, fc_units),
            nn.ReLU()
        )

        # Critic head
        self.fc_critic = nn.Sequential(
            nn.Linear(state_size, fc_units),
            nn.ReLU(),
            nn.Linear(fc_units, fc_units),
            nn.ReLU()
        )

        # Actor head
        self.fc_mean = nn.Linear(fc_units, action_size)
        self.log_std = nn.Parameter(torch.zeros(action_size))

        # Critic Head
        self.fc_value = nn.Linear(fc_units, 1)

        self.apply(weights_init_)

    def forward(self, state):
        """Forward method implementation."""
        
        value = self.fc_value(self.fc_critic(state))

        actor = self.fc_actor(state)
        action_mean = self.fc_mean(actor) # Shape: [batch_size, action_size]
        action_std = self.log_std.expand_as(action_mean).exp() # Expand to match batch size and Convert log-std to std
        distribution = torch.distributions.Normal(action_mean, action_std)
        
        action_raw = distribution.sample()
        action_tanh = torch.tanh(action_raw) # Squash action to [-1, 1]
        action = action_tanh * self.action_scale + self.action_bias

        log_prob = distribution.log_prob(action_raw)
        log_prob -= torch.log(self.action_scale * (1 - action_tanh.pow(2)) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)

        entropy = distribution.entropy().sum(-1, keepdim=True)

        return value, action, log_prob, entropy, action_mean

    def get_value(self, state):
        """Get the value of a state."""
        return self.fc_value(self.fc_critic(state))


# --- Logger Function ---
def logger(queue, global_model, shared_obs_norm, args, save_interval=10, 
           ema_alpha=0.99, track_best=True, min_episodes_for_best=0):
    """
    Logs training metrics from an A3C queue and saves models.
    
    Args:
        queue: multiprocessing.Queue for receiving worker data
        global_model: Shared PyTorch model to save
        shared_obs_norm: SharedRunningMeanStd for observation normalization
        args: Parsed command-line arguments
        save_interval: Episodes between periodic saves
        ema_alpha: Exponential moving average factor for rewards
        track_best: Whether to track and save the best model
        min_episodes_for_best: Minimum episodes before saving best model
        max_reward_history: Length of reward history for stats
    """

    run_name =  (
        f"lr{args.lr}"
        f"_gamma{args.gamma}"
        f"_gae{args.gae_lambda}"
        f"_entcoef{args.ent_coef}"
        f"_steps{args.num_steps}"
        f"_nproc{args.num_processes}"
        f"_{int(time.time())}"
    )

    sharedLogger = Logger(run_name=run_name, env=args.env_id, algo="A3C")
    sharedLogger.add_run_command()
    sharedLogger.log_all_hyperparameters(vars(args))
    
    # Tracking variables
    global_episodes = 0
    ema_reward = 0.0
    best_reward = float('-inf') if track_best else None

    while True:
        try:
            # Wait for data with a timeout to avoid blocking indefinitely
            if queue.empty():
                time.sleep(0.1)  # More relaxed sleep than 0.001
                continue
            data = queue.get(timeout=1.0)  # Non-blocking with timeout
            if data.get('type') == 'stop':
                break

            # Process data based on type
            if data['type'] == 'loss':
                losses = data['losses']
                global_steps = data['steps']
                sharedLogger.add_scalar("Loss/Entropy", losses['entropy_loss'], global_steps)
                sharedLogger.add_scalar("Loss/Value", losses['value_loss'], global_steps)
                sharedLogger.add_scalar("Loss/Policy", losses['policy_loss'], global_steps)
                sharedLogger.add_scalar("Loss/Total", losses['total_loss'], global_steps)

            elif data['type'] == 'reward':
                episode_reward = data['reward']
                worker_id = data.get('worker_id', 'unknown')  # Optional worker info
                global_steps = data['steps']

                # Update EMA reward
                ema_reward = ema_reward * ema_alpha + episode_reward * (1 - ema_alpha) if ema_reward else episode_reward
                
                # Log metrics
                sharedLogger.add_scalar("Reward/EMA", ema_reward, global_steps)
                sharedLogger.add_scalar(f"Reward/Worker_{worker_id}", episode_reward, global_steps)
                
                # Save best model
                if track_best and episode_reward > best_reward and global_episodes >= min_episodes_for_best:
                    best_reward = episode_reward
                    save_model(shared_obs_norm, global_model, sharedLogger, "best-torch.model")
                    print(f"Worker {worker_id} | Saved best model with reward {episode_reward:.2f} at episode {global_episodes}")

                # Periodic logging and saving
                if global_episodes % save_interval == 0:
                    save_model(shared_obs_norm, global_model, sharedLogger, "torch.model")
                    print(f"Ep: {global_episodes} | EMA Reward: {ema_reward:.2f}")

                global_episodes += 1

            # sharedLogger.flush() For real life updates

        except Exception as e:
            print(f"Logger error: {e}")
            time.sleep(1.0)  # Wait before retrying on error

    sharedLogger.close()  # Unreachable in infinite loop; consider a shutdown signal

# --- Worker Function ---
def worker(
    worker_id: int,
    args: argparse.Namespace,
    global_model: ActorCritic,
    optimizer: torch.optim.Adam,  # Shared optimizer
    shared_obs_norm: SharedRunningMeanStd,
    global_steps: mp.Value,
    results_queue: mp.Queue,
    global_model_lock: mp.Lock
):
    name = f'W{worker_id}'
    env = create_mujoco_env(args.env_id)

    local_model = ActorCritic(env)
    local_steps = 0
    print(f"Worker {name} initialized and running.")
    while global_steps.value < args.total_timesteps:
        # Reset environment and state
        obs, _ = env.reset()
        state = torch.from_numpy(obs).float()
        
        done = False
        episode_reward = 0
        while not done:
            # Sync with global model
            local_model.load_state_dict(global_model.state_dict())

            values = []
            log_probs = []
            entropies = []
            rewards = []

            # Collect rollouts
            for _ in range(args.num_steps):
                # Test the shape for everything 
                value, action, log_prob, entropy, _ = local_model(state)
                action_np = action.detach().numpy() # Convert to numpy for environment
                next_obs, reward, terminated, truncated, _ = env.step(action_np)
                done = terminated or truncated
                next_state = torch.from_numpy(next_obs).float()
                episode_reward += reward

                # Save rollout data
                values.append(value)
                log_probs.append(log_prob)
                entropies.append(entropy)
                rewards.append(reward)

                state = next_state

                local_steps += 1
                if done:
                    break
            
            # We care about the last value for bootstrapping 
            # (0 if terminated, otherwise the value of the last state)
            last_value = local_model.get_value(state) if not terminated else torch.tensor([0.0])
            
            # After collection, convert everything to tensors
            returns = torch.zeros(len(rewards)) 
            values_t = torch.cat(values + [last_value])  # Add last_value at the end
            log_probs_t = torch.cat(log_probs) 
            rewards_t = torch.tensor(rewards, dtype=torch.float32) 
            entropies_t = torch.cat(entropies)

            # Compute GAE returns with tensors
            gae = torch.zeros(1)
            for step in reversed(range(len(rewards))):
                delta = rewards_t[step] + args.gamma * values_t[step+1].detach() - values_t[step].detach()
                gae = delta + args.gamma * args.gae_lambda * gae
                returns[step] = gae + values_t[step].detach()

            
            # Compute losses
            advantages = returns - values_t[:-1]
            if len(advantages) > 1:  # Only normalize if more than one sample
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            policy_loss = -(log_probs_t * advantages.detach()).mean()
            value_loss = F.mse_loss(values_t[:-1], returns.detach())
            entropy_loss = -entropies_t.mean()

            total_loss  = policy_loss + value_loss * args.vf_coef + entropy_loss * args.ent_coef

            # Update global model
            optimizer.zero_grad()
            total_loss.backward()
            # Grad clipping helpful in continuous action spaces
            torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=args.max_grad_norm)   
            with global_model_lock:
                for local_param, global_param in zip(local_model.parameters(), global_model.parameters()):
                    if local_param.grad is not None:
                        global_param._grad = local_param.grad.clone()
                optimizer.step()

            # Log losses
            results_queue.put({
                'type': 'loss',
                "worker_id": name,
                "steps": global_steps.value + local_steps,
                'losses': {
                    'policy_loss': policy_loss.item(),
                    'value_loss': value_loss.item(),
                    'entropy_loss': entropy_loss.item(),
                    'total_loss': total_loss.item()
                }
            })
        
        # Only update global counter once per episode
        with global_steps.get_lock():
            global_steps.value += local_steps
            local_steps = 0
        
        # Update optimizer learning rate
        if args.lr_anneal:
            # Calculate fraction of training completed
            progress_fraction = min(1.0, global_steps.value / args.total_timesteps)
            # Linear annealing from initial LR to final LR
            new_lr = args.lr * (1.0 - progress_fraction * args.lr_anneal_factor)
            for param_group in optimizer.param_groups:
                param_group['lr'] = max(1e-7, new_lr)  # Don't go below minimum LR

        # Sync obs normalization parameters
        shared_obs_norm.sync_worker(
            worker_id, 
            env
        )

        results_queue.put({
            'type': 'reward',
            'steps': global_steps.value,
            'worker_id': name,
            'reward': episode_reward
        })


# --- Main Training Function ---
def main():
    parser = argparse.ArgumentParser()
    # Consolidated environment selection.
    parser.add_argument('--env-type', type=str, choices=['gym', 'dmc'], default='gym',
                        help='Environment type: "gym" uses Gymnasium env id (e.g. Mujoco), '
                             'while "dmc" uses DM Control (specify domain and task names).')
    parser.add_argument('--env-id', type=str, default='HalfCheetah-v5',
                        help='Gymnasium environment ID (used if env-type is gym)')
    parser.add_argument('--domain-name', type=str, default='humanoid_CMU',
                        help='DM Control domain name (used if env-type is dmc)')
    parser.add_argument('--task-name', type=str, default='run',
                        help='DM Control task name (used if env-type is dmc)')
    # Other training Hyperparameters
    parser.add_argument('--total-timesteps', type=int, default=int(1e6))
    parser.add_argument('--num-processes', type=int, default=4)
    parser.add_argument('--num-steps', type=int, default=20)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--ent-coef', type=float, default=0.001)
    parser.add_argument('--vf-coef', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=7e-4)
    parser.add_argument('--lr-anneal', type=bool, default=False)
    parser.add_argument('--lr-anneal-factor', type=float, default=0.8,
                   help='Final LR will be (1-factor)*initial_lr')
    parser.add_argument('--max-grad-norm', type=float, default=0.5)
    args = parser.parse_args()

    
    env = create_mujoco_env(args.env_id)

    assert isinstance(env.action_space, gym.spaces.Box), "only continuous action space is supported"
    
    global_model = ActorCritic(env)
    global_model.share_memory()

    shared_obs_norm = SharedRunningMeanStd(env.observation_space.shape, num_workers=args.num_processes)
    shared_optimizer = optim.Adam(global_model.parameters(), lr=args.lr)

    results_queue = mp.Queue()#  funny it support max like 37k I think 
    global_steps = mp.Value('i', 0) 
    global_model_lock = mp.Lock()
    
    # Start logger process
    logger_process = mp.Process(target=logger, args=(
        results_queue, 
        global_model, 
        shared_obs_norm,
        args))
    logger_process.start()

    # Start parallel workers
    processes = []
    for worker_id in range(args.num_processes):
        p = mp.Process(target=worker, args=(
            worker_id,
            args,
            global_model,
            shared_optimizer,
            shared_obs_norm,
            global_steps,
            results_queue,
            global_model_lock))
        p.start()
        processes.append(p)

    # Wait for workers to finish
    for p in processes:
        p.join()

    # Stop logger
    results_queue.put({'type': 'stop'})
    logger_process.join() 

if __name__ == "__main__":
    main()
    print('All finished')
