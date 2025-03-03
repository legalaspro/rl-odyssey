import argparse
import os
import time

import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp

from helpers import Logger

import gymnasium as gym
import ale_py
gym.register_envs(ale_py)

# -- Helper Functions --

def weights_init_(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=np.sqrt(2)) # For ReLU activations
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.orthogonal_(m.weight, gain=1.0)
        nn.init.zeros_(m.bias)

def save_model(actor_critic, logger, filename: str):
    """Save the current model state dictionaries and normalization info to a file in logger.dir_name."""
    model_path = os.path.join(logger.dir_name, filename)
    torch.save(actor_critic.state_dict(), model_path)

def create_atari_env(env_id: str):
    """Create an Atari environment with some standard preprocessing."""
    env = gym.make(env_id)
    env = gym.wrappers.AtariPreprocessing(
        env,
        noop_max=0,
        frame_skip=1,  # Deterministic envs already skip frames internally
        screen_size=42,
        terminal_on_life_loss=True,
        grayscale_obs=True,
        scale_obs=True # Normalize pixel values to [0,1]
    )
    env = gym.wrappers.FrameStackObservation(env, stack_size=4)
    return env


# --- Actor-Critic Model ---
class ActorCritic(nn.Module):
    def __init__(self, env):
        super().__init__()

        # Feature extraction - shared CNN backbone
        self.features = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=4, stride=2),   # (42 - 4) / 2 + 1 = 20
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2), # (20 - 3) / 2 + 1 = 9
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), # (9 - 3) / 1 + 1 = 7
            nn.ReLU(),
            nn.Flatten(start_dim=1)
        )

        # Calculate flattened size after convolutions
        feature_size = 64 * 7 * 7  #  64 channels with 7x7 spatial dimensions

        # Actor (policy) head
        self.actor = nn.Sequential(
            nn.Linear(feature_size, 256),
            nn.ReLU(),
            nn.Linear(256, env.action_space.n)  # Outputs logits for each discrete action
        )
        
        # Critic (value) head
        self.critic = nn.Sequential(
            nn.Linear(feature_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.apply(weights_init_)


    def forward(self, state):
        # state shape: (batch_size, channels, height, width)
        features = self.features(state)

        # Actor: logits for each action
        action_logits = self.actor(features)
        action_dist = torch.distributions.Categorical(logits=action_logits)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action).unsqueeze(-1)
        entropy = action_dist.entropy().unsqueeze(-1)

        # Critic: state value
        value = self.critic(features)
        return value, action, log_prob, entropy

    def get_value(self, state):
        features = self.features(state)
        return self.critic(features)

# --- Logger Function ---
def logger(queue, global_model, args, save_interval=10, 
           ema_alpha=0.99, track_best=True, min_episodes_for_best=0):
    """
    Logs training metrics from an A3C queue and saves models.
    
    Args:
        queue: multiprocessing.Queue for receiving worker data
        env: Environment to interact with
        global_model: Shared PyTorch model to save
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
                episode_length = data.get('length', 0)  # Get length
                worker_id = data.get('worker_id', 'unknown')  # Optional worker info
                global_steps = data['steps']

                # Update EMA reward
                ema_reward = ema_reward * ema_alpha + episode_reward * (1 - ema_alpha) if ema_reward else episode_reward
                
                # Log metrics
                sharedLogger.add_scalar("Reward/EMA", ema_reward, global_steps)
                sharedLogger.add_scalar("Episode/Length", episode_length, global_steps)  # Log raw length
                sharedLogger.add_scalar(f"Reward/Worker_{worker_id}", episode_reward, global_steps)
                # sharedLogger.add_scalar(f"Length/Worker_{worker_id}", episode_length, global_steps)


                # Save best model
                if track_best and episode_reward > best_reward and global_episodes >= min_episodes_for_best:
                    best_reward = episode_reward
                    save_model(global_model, sharedLogger, "best-torch.model")
                    print(f"Worker {worker_id} | Saved best model with reward {episode_reward:.2f} at episode {global_episodes}")

                # Periodic logging and saving
                if global_episodes % save_interval == 0:
                    save_model(global_model, sharedLogger, "torch.model")
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
    global_steps: mp.Value,
    results_queue: mp.Queue,
    global_model_lock: mp.Lock
):
    name = f'W{worker_id}'
    env = create_atari_env(args.env_id)

    local_model = ActorCritic(env)
    local_steps = 0
    sync_counter = 0
    episode_reward = 0
    episode_length = 0  # Add episode length counter

    obs, _ = env.reset()
    state = torch.from_numpy(obs).float()
    done = False

    print(f"Worker {name} initialized and running.")
    while global_steps.value < args.total_timesteps:
        # Sync with global model
        # if sync_counter % 20 == 0:  # Only sync every few episodes
        local_model.load_state_dict(global_model.state_dict())
        # sync_counter += 1

        values = []
        log_probs = []
        entropies = []
        rewards = []
        dones = []  # Track done states for proper bootstrapping

        # Collect rollouts
        for _ in range(args.num_steps):
            # Test the shape for everything 
            value, action, log_prob, entropy = local_model(state.unsqueeze(0))
            action_np = action.item() # Convert to int for discrete actions

            next_obs, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated
           
            # Save rollout data
            values.append(value)
            log_probs.append(log_prob)
            entropies.append(entropy)
            rewards.append(reward)
            dones.append(done) 

            episode_reward += reward
            episode_length += 1
            local_steps += 1
            
            next_state = torch.from_numpy(next_obs).float()
            state = next_state

            if done:
                # Log episode results
                results_queue.put({
                    'type': 'reward',
                    'steps': global_steps.value,
                    'worker_id': name,
                    'reward': episode_reward,
                    'length': episode_length  # Add episode length to data
                })

                # Reset episode stats
                episode_reward = 0
                episode_length = 0

                # Reset environment to start a new episode
                obs, _ = env.reset()
                state = torch.from_numpy(obs).float()
                done = False
            
        # After collecting rollouts, bootstrap from last state
        last_value = torch.zeros(1, 1) if dones[-1] else local_model.get_value(state.unsqueeze(0))
    
        # Convert collections to tensors
        values_t = torch.cat(values + [last_value])  # Include bootstrap value
        log_probs_t = torch.cat(log_probs)
        rewards_t = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        entropies_t = torch.cat(entropies)
        dones_t = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

        # Compute GAE returns with tensors
        returns = torch.zeros(len(rewards), 1)
        gae = torch.zeros(1, 1)
        for step in reversed(range(len(rewards))):
            next_value = values_t[step + 1] * (1.0 - dones_t[step])
            delta = rewards_t[step] + args.gamma * next_value.detach() - values_t[step].detach()
        
            gae = delta + args.gamma * args.gae_lambda * gae * (1.0 - dones_t[step])
            returns[step] = gae + values_t[step].detach()

        
        # Compute losses
        advantages = returns - values_t[:-1]
        # advantages = torch.clamp(advantages, min=-10.0, max=10.0) 
        # if len(advantages) > 1:  # Only normalize if more than one sample
        #     advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        policy_loss = -(log_probs_t * advantages.detach()).mean()
        value_loss = F.mse_loss(values_t[:-1], returns.detach())
        entropy_loss = -entropies_t.mean()

        total_loss  = policy_loss + value_loss * args.vf_coef + entropy_loss * args.ent_coef

        # Update global model
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=args.max_grad_norm)
        # Copy gradients from local to global
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


# --- Main Training Function ---

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--env-id', type=str, default='PongDeterministic-v4')
    parser.add_argument('--total-timesteps', type=int, default=int(1e6))
    parser.add_argument('--num-processes', type=int, default=4)
    parser.add_argument('--num-steps', type=int, default=128)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--ent-coef', type=float, default=0.01)
    parser.add_argument('--vf-coef', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=7e-4)
    parser.add_argument('--lr-anneal', action='store_true', default=False)
    parser.add_argument('--lr-anneal-factor', type=float, default=0.75, 
                   help='Final LR will be (1-factor)*initial_lr')
    parser.add_argument('--max-grad-norm', type=float, default=0.5)
    args = parser.parse_args()

    
    env = create_atari_env(args.env_id)
    assert isinstance(env.action_space, gym.spaces.Discrete), "only discrete action space is supported"
    assert 'Deterministic' in args.env_id, "Use Deterministic environment variants"

    global_model = ActorCritic(env)
    global_model.share_memory()

    # shared_optimizer = optim.RMSprop(global_model.parameters(), lr=args.lr, alpha=0.99, eps=1e-5)
    shared_optimizer = optim.Adam(global_model.parameters(), lr=args.lr)

    results_queue = mp.Queue()#  funny it support max like 37k I think 
    global_steps = mp.Value('i', 0) 
    global_model_lock = mp.Lock()

    
    # Start logger process
    logger_process = mp.Process(target=logger, args=(
        results_queue,
        global_model, 
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

    print('All finished')