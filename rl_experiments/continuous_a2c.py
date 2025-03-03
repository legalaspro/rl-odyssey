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

from rollout_storage import RolloutStorage, RolloutBatch

#TODO: Remove it once Mujoco code fixed in the gymnasium
# or downgrade numpy to 1.21.0
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def weights_init_(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        nn.init.zeros_(m.bias)

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

def evaluate(env, actor_critic, n_rollout=10):
    """Evaluate the policy over a number of episodes."""
    tot_rw = 0
    for _ in range(n_rollout):
        state, _ = env.reset()
        done = False
        while not done:
            state = torch.from_numpy(np.array(state, dtype=np.float32)).to(device)
            with torch.no_grad():
                _, action, _ = actor_critic(state, deterministic=True)

            next_state, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
            tot_rw += reward
            state = next_state
            done = terminated or truncated
    return tot_rw / n_rollout

def save_model(env, actor_critic, logger, filename: str):
    """Save the current model state dictionaries and normalization info to a file in logger.dir_name."""
    model_path = os.path.join(logger.dir_name, filename)
    # Gather normalization statistics from each environment
    all_obs_rms = [env.obs_rms for env in env.envs]
    
    checkpoint = {
        'actor_critic_state_dict': actor_critic.state_dict(),
        'obs_rms': all_obs_rms,
    }
    torch.save(checkpoint, model_path)
    print(f"model saved to {model_path}")

# ALGO LOGIC: initialize agent here:
class ActorCritic(nn.Module):
    def __init__(self, env, fc_units=128):
        """Initialize parameters and build model."""
        super(ActorCritic, self).__init__()

        # Actor head
        self.fc_actor = nn.Sequential(
            nn.Linear(env.single_observation_space.shape[0], fc_units),
            nn.ReLU(),
            nn.Linear(fc_units, fc_units),
            nn.ReLU()
        )

        # Critic head
        self.fc_critic = nn.Sequential(
            nn.Linear(env.single_observation_space.shape[0], fc_units),
            nn.ReLU(),
            nn.Linear(fc_units, fc_units),
            nn.ReLU()
        )

        # Actor presented in form of mean and logstd to use Gaussian Distribution 
        self.fc_mean = nn.Linear(fc_units, env.single_action_space.shape[0]) 
        self.log_std = nn.Parameter(torch.zeros(env.single_action_space.shape[0]))  # standard initialization std = exp(log_std) = exp(0) = 1  

        # Critic calculates single value function result per state
        self.fc_critic_out = nn.Linear(fc_units, 1) 

        # Apply weights initialization for all linear layers
        self.apply(weights_init_)

    def _get_action_distribution(self, state):
        """Compute action mean, log_std, and distribution."""
        action_mean = self.fc_mean(self.fc_actor(state))
        action_log_std = self.log_std.expand_as(action_mean)  # Expand to match batch size
        action_std = action_log_std.exp()  # Convert log-std to std
        distribution = torch.distributions.Normal(action_mean, action_std)
        return distribution

    def evaluate_actions(self, state, action):
        """Evaluate given actions (used for computing log-prob and entropy)."""
        distribution = self._get_action_distribution(state)
        value = self.fc_critic_out(self.fc_critic(state))

        log_prob = distribution.log_prob(action).sum(-1, keepdim=True)  # Sum over action dimensions
        entropy = distribution.entropy().sum(-1, keepdim=True)  # Entropy of the distribution

        return value, log_prob, entropy
    

    def forward(self, state, deterministic=False):
        """Forward method implementation."""
        distribution = self._get_action_distribution(state)
        value = self.fc_critic_out(self.fc_critic(state))

        if deterministic:
            action = distribution.mean  # will use only mean for evaluation 
        else: 
            action = distribution.sample()
        
        log_prob = distribution.log_prob(action).sum(-1, keepdim=True)  # sums each action log prob into single result per state

        return value, action, log_prob

    def get_value(self, state):
        """Calculate only value for given state"""
        value = self.fc_critic_out(self.fc_critic(state))
        return value


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
    parser.add_argument('--num-envs', type=int, default=1)
    parser.add_argument('--num-steps', type=int, default=1024)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--hidden-neurons', type=int, default=128)
    parser.add_argument('--normalize-advantages', type=bool, default=True)
    parser.add_argument('--ent-coef', type=float, default=0.01)
    parser.add_argument('--vf-coef', type=float, default=0.5) # not used really
    parser.add_argument('--max-grad-norm', type=float, default=0.5)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--anneal-lr', type=bool, default=False)
    parser.add_argument('--init-weights', type=bool, default=True)
    parser.add_argument('--eval-interval', type=int, default=10000)
    args = parser.parse_args()


    run_name =  (
        f"lr{args.learning_rate}"
        f"_anlr{args.anneal_lr}"
        f"_gamma{args.gamma}"
        f"_entcoef{args.ent_coef}"
        f"_gae{args.gae_lambda}"
        f"_steps{args.num_steps}"
        f"_nenvs{args.num_envs}"
        f"_{int(time.time())}"
    )

    logger = Logger(run_name=run_name, env=args.env_id, algo="A2C")
    logger.add_run_command()
    logger.log_all_hyperparameters(vars(args))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    envs = make_sync_vec(args.env_id, 
                    num_envs=args.num_envs, 
                    wrappers=(gym.wrappers.RecordEpisodeStatistics, 
                              gym.wrappers.ClipAction,
                              gym.wrappers.NormalizeObservation,),
                    autoreset_mode=AutoresetMode.SAME_STEP)
    
    test_env = gym.make(args.env_id)
    test_env = gym.wrappers.ClipAction(test_env)
    test_env = gym.wrappers.NormalizeObservation(test_env)
    test_env.update_running_mean = False

    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    actor_critic = ActorCritic(envs).to(device)
    optimizer = optim.Adam(actor_critic.parameters(), lr=args.learning_rate)

    rollout_storage = RolloutStorage(
        obs_shape=envs.single_observation_space.shape,
        action_dim=envs.single_action_space.shape,
        num_steps=args.num_steps,
        n_envs=args.num_envs,
        device=device
    )

    start_time = time.time()
    global_step = 0
    train_iterations = 0
    best_eval_return = -np.inf

    obs, _ = envs.reset()
    steps_per_iter = int(args.num_steps * args.num_envs)
    num_iterations = args.total_timesteps // steps_per_iter
    eval_interval = args.eval_interval // steps_per_iter + 1

    for iteration in  range(num_iterations):

        if args.anneal_lr:
            frac = 1.0 - iteration / num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
        
        rollout_storage.reset() # important to reset storage before starting new rollout
        for _ in range(args.num_steps):

            with torch.no_grad():
                obs = torch.from_numpy(np.array(obs, dtype=np.float32)).to(device)
                value, action, log_prob = actor_critic(obs)
            
            # Take action in env and look the results
            next_obs, rewards, terminations, truncations, infos = envs.step(action.cpu().numpy())

            #Updating global step
            global_step += envs.num_envs

            dones = (terminations | truncations)
            masks = torch.tensor(1-dones, dtype=torch.float32).unsqueeze(-1).to(device)
            rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1).to(device)
            truncations = torch.tensor(truncations, dtype=torch.bool).unsqueeze(-1).to(device)

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
            
            rollout_storage.add(
                obs,
                action,
                log_prob,
                value,
                rewards,
                masks, 
                truncations)
            
            obs = next_obs
        
        # compute the last state value
        with torch.no_grad():
            obs = torch.from_numpy(np.array(obs, dtype=np.float32)).to(device)
            last_value = actor_critic.get_value(obs)
        
        rollout_storage.compute_returns_and_advantages(
            last_value,
            args.gamma,
            args.gae_lambda,
            args.normalize_advantages
        )

        # pass over the full rollout storage once - for A2C it's enough
        for rollout_data in rollout_storage.get_mini_batch(batch_size=None):
            values, log_probs, entropies = actor_critic.evaluate_actions(rollout_data.states, rollout_data.actions)
            advantages = rollout_data.advantages
            returns = rollout_data.returns

            # 1. Policy gradient loss or Actor gradient loss
            actor_loss = -(advantages * log_probs).mean()

            # 2. Value function loss or Critic loss
            value_loss = F.mse_loss(values, returns)

            # 3. Entropy loss to favor more exploration
            entropy_loss = -entropies.mean()

            # 4. Combine all losses
            total_loss = actor_loss + args.ent_coef * entropy_loss + args.vf_coef * value_loss

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(actor_critic.parameters(), args.max_grad_norm)
            optimizer.step()


        # logger.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        logger.add_scalar("losses/value_loss", value_loss.item(), global_step)
        logger.add_scalar("losses/policy_loss", actor_loss.item(), global_step)
        logger.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        logger.add_scalar("losses/std", torch.exp(actor_critic.log_std).mean().item(), global_step)
        logger.add_scalar("charts/FPS", int(global_step / (time.time() - start_time)), global_step)

        if (iteration+1) % eval_interval == 0:
            sync_normalization_stats(test_env, envs)
            eval_return = evaluate(test_env, actor_critic)
            if best_eval_return < eval_return:
                best_eval_return = eval_return
                save_model(envs, actor_critic, logger, "best-torch.model")
            logger.add_scalar("eval/returns", eval_return, global_step)
            logger.log_stdout()


    # ---- After training completes ----
    save_model(envs, actor_critic, logger, "torch.model")

    
    envs.close()
    logger.close()
            
