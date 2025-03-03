import argparse
import os
import time
from copy import deepcopy
from functools import partial



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

def evaluate(env, actor, n_rollout=10):
    """Evaluate the policy over a number of episodes."""
    tot_rw = 0
    for _ in range(n_rollout):
        state, _ = env.reset()
        done = False
        while not done:
            state = torch.from_numpy(np.array(state, dtype=np.float32)).to(device)
            with torch.no_grad():
                action, _ = actor(state, deterministic=True)

                action = action.cpu().numpy().clip(env.action_space.low, 
                                                   env.action_space.high)

            next_state, reward, terminated, truncated, _ = env.step(action)
            tot_rw += reward
            state = next_state
            done = terminated or truncated
    return tot_rw / n_rollout

def save_model(env, actor, critic, logger, filename: str):
    """Save the current model state dictionaries and normalization info to a file in logger.dir_name."""
    model_path = os.path.join(logger.dir_name, filename)
    # Gather normalization statistics from each environment
    all_obs_rms = [env.obs_rms for env in env.envs]
    
    checkpoint = {
        'actor': actor.state_dict(),
        'critic': critic.state_dict(),
        'obs_rms': all_obs_rms,
    }
    torch.save(checkpoint, model_path)
    print(f"model saved to {model_path}")

# ALGO LOGIC: initialize agent here:
def compute_flattened_gradients(output, parameters, retain_graph=False, create_graph=False):
    """Compute the gradients of the output w.r.t. the parameters and flatten them into a single vector.
    Args:
        output: Output tensor to compute gradients.
        parameters: List of parameters to compute gradients.
        retain_graph: Whether to retain the computation graph. If False, the graph is freed after computing gradients.
        create_graph: Whether to create the computation graph. If True, the graph is retained for further computations.
    """
    if create_graph:
        retain_graph = True

    grads = torch.autograd.grad(output, parameters, retain_graph=retain_graph, create_graph=create_graph)
    grads = torch.cat([grad.view(-1) for grad in grads])
    return grads

def hessian_vector_product(params, grad_kl, vector):
    """ Compute the Hessian-vector product with the Fisher information matrix.
    g(θ) = ∇₍θ₎Dkl(θ)
    1st operation H(θ)v = ∇₍θ₎(∇₍θ₎Dkl(θ)ᵀv) = ∇₍θ₎(g(θ)ᵀv)
    2nd operation is adding damping term to the Hessian-vector product to improve numerical stability. 
    Used in following regularized system (∇²D(θ) + cg_damping·I)·v = rhs  
    Args:
        params: List of parameters to compute the Hessian-vector product.
        grad_kl: Gradient of the KL divergence between old and new policy, flattened.
        vector: Vector to compute the dot product with the Hessian.
    """
    jacobian_vector_product = compute_flattened_gradients(grad_kl @ vector, params, retain_graph=True)
    hessian_vector_product = jacobian_vector_product + args.cg_damping * vector
    return hessian_vector_product

def conjugate_gradient(HVP, g, nsteps, residual_tol=1e-8):
    """ Conjugate gradient algorithm to solve Ax = b 
    We want to solve Hx = g, where H is the Hessian matrix and g is the gradient vector.
    x is the search direction, which is the solution we are looking for.
    Args:
        HVP: Function that computes Hessian-vector product, HVP(x) = Hx
        g: Gradient vector, b in Ax = b
        nsteps: Number of iterations
        residual_tol: Tolerance for stopping condition
    """
    x = torch.zeros_like(g)
    residual = g - HVP(x) # Residual = b - Ax, where A is the Hessian matrix and x direction
    d = residual.clone() # seach direction (start with d0 = residual)
    residual_dot_residual = torch.dot(residual, residual) # initial residual norm (squared)
    for i in range(nsteps):
        Hx = HVP(d) # Hessian-vector product
        d_Hx = torch.dot(d, Hx)
        # Avoid division by very small value for stability
        if torch.abs(d_Hx) < 1e-10:
            break
        # Compute step size alpha = residual_dot_residual / d^T Hx
        # x = x + alpha * d
        alpha = residual_dot_residual / d_Hx # step size
        x += alpha * d # update x

        residual -= alpha * Hx # update residual = residual - alpha * Hx
        
        new_residual_dot_residual = torch.dot(residual, residual)
        if new_residual_dot_residual < residual_tol:
            break

        # Compute beta = residual_dot_residual_new / residual_dot_residual 
        # which is new_residual^T new_residual / residual^T residual
        beta = torch.dot(residual, residual) / residual_dot_residual # update beta
        d = residual + beta * d # update search direction
        residual_dot_residual = new_residual_dot_residual # update residual norm

    return x, i + 1, residual

class Actor(nn.Module):
    def __init__(self, env, fc_units=128):
        """Initialize parameters and build model."""
        super(Actor, self).__init__()

        # Actor head
        self.fc_actor = nn.Sequential(
            nn.Linear(env.single_observation_space.shape[0], fc_units),
            nn.ReLU(),
            nn.Linear(fc_units, fc_units),
            nn.ReLU()
        )

        # Actor presented in form of mean and logstd to use Gaussian Distribution 
        self.fc_mean = nn.Linear(fc_units, env.single_action_space.shape[0]) 
        self.log_std = nn.Parameter(torch.zeros(env.single_action_space.shape[0]))  # standard initialization std = exp(log_std) = exp(0) = 1  

        self.apply(weights_init_)  # initialize weights

    def get_action_distribution(self, state):
        """Compute action mean, log_std, and distribution."""
        action_mean = self.fc_mean(self.fc_actor(state))
        action_log_std = self.log_std.expand_as(action_mean)  # Expand to match batch size
        action_std = action_log_std.exp()  # Convert log-std to std
        distribution = torch.distributions.Normal(action_mean, action_std)
        return distribution

    def forward(self, state, deterministic=False):
        """Forward method implementation."""
        distribution = self.get_action_distribution(state)

        if deterministic:
            action = distribution.mean  # will use only mean for evaluation 
        else: 
            action = distribution.sample()
        
        log_prob = distribution.log_prob(action).sum(-1, keepdim=True)  # sums each action log prob into single result per state

        return action, log_prob
    
    def evaluate_actions(self, state, action):
        """Evaluate given actions (used for computing log-prob and entropy)."""
        distribution = self.get_action_distribution(state)

        log_prob = distribution.log_prob(action).sum(-1, keepdim=True)  # Sum over action dimensions
        entropy = distribution.entropy().sum(-1, keepdim=True)  # Entropy of the distribution

        return log_prob, entropy

class Critic(nn.Module):

    def __init__(self, env, fc_units=128):
        """Initialize parameters and build model."""
        super(Critic, self).__init__()

        # Critic head
        self.fc_critic = nn.Sequential(
            nn.Linear(env.single_observation_space.shape[0], fc_units),
            nn.ReLU(),
            nn.Linear(fc_units, fc_units),
            nn.ReLU()
        )

        # Critic calculates single value function result per state
        self.fc_critic_out = nn.Linear(fc_units, 1) 

        self.apply(weights_init_)  # initialize weights

    def forward(self, state):
        """Forward method implementation."""
        value = self.fc_critic_out(self.fc_critic(state))
        return value

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--env-id', type=str, default='Humanoid-v5')
    parser.add_argument('--total-timesteps', type=int, default=int(1e6))
    parser.add_argument('--num-envs', type=int, default=1)
    parser.add_argument('--num-steps', type=int, default=2048)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--normalize-advantages', type=bool, default=True)
    parser.add_argument('--cg-max-steps', type=int, default=15, help="Maximum number of iterations in Conjugate Gradient")
    parser.add_argument('--cg-damping', type=float, default=0.1, help="Damping factor in Conjugate Gradient")
    parser.add_argument('--line-search-step-decay', type=float, default=0.8, help="Line search step decay")
    parser.add_argument('--max-num-linesearch-steps', type=int, default=10, help="Maximum number of line search steps")
    parser.add_argument('--n-critic-updates', type=int, default=10, help="Number of critic updates per iteration")
    parser.add_argument('--target-kl', type=float, default=0.01, help="Target KL divergence (Dkl(π₍θ_old₎ || π₍θ₎) < δ)")
    parser.add_argument('--ent-coef', type=float, default=0.0)
    parser.add_argument('--vf-coef', type=float, default=0.5) # not used really
    parser.add_argument('--max-grad-norm', type=float, default=0.5)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--anneal-lr', type=int, default=False)
    parser.add_argument('--eval-interval', type=int, default=10000)
    args = parser.parse_args()

    run_name = (
        f"lr{args.learning_rate}"
        f"_gamma{args.gamma}"
        f"_gae{args.gae_lambda}"
        f"_nenvs{args.num_envs}"
        f"_steps{args.num_steps}"
        f"_cg{args.cg_max_steps}"
        f"_kl{args.target_kl}"
        f"_lsdecay{args.line_search_step_decay}",
        f"_{int(time.time())}"
    )
    run_name = "".join(run_name)

    logger = Logger(run_name=run_name, env=args.env_id, algo="TRPO")
    logger.add_run_command()
    logger.log_all_hyperparameters(vars(args))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # env setup 
    envs = make_sync_vec(args.env_id, 
                    num_envs=args.num_envs, 
                    wrappers=(gym.wrappers.RecordEpisodeStatistics, 
                              gym.wrappers.ClipAction,
                              gym.wrappers.NormalizeObservation, ),
                    autoreset_mode=AutoresetMode.SAME_STEP)
    
    test_env = gym.make(args.env_id)
    test_env = gym.wrappers.NormalizeObservation(test_env)
    test_env.update_running_mean = False

    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    actor = Actor(envs).to(device)
    critic = Critic(envs).to(device)
    value_optimzier = optim.Adam(critic.parameters(), lr=args.learning_rate)

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
    iterations = 0 
    steps_per_iter = int(args.num_steps * args.num_envs)
    num_iterations = args.total_timesteps // steps_per_iter
    eval_interval = args.eval_interval // steps_per_iter

    # ---- Rollout collection ----

    for iteration in range(num_iterations):
        
        rollout_storage.reset() # important to reset storage before starting new rollout
        for _ in range(args.num_steps):
            with torch.no_grad():
                obs = torch.from_numpy(np.array(obs, dtype=np.float32)).to(device)
                action, log_prob = actor(obs)
                value = critic(obs)
            
            # Take action in env and look the results
            next_obs, rewards, terminations, truncations, infos = envs.step(action.cpu().numpy())

            # Updating global step
            global_step += envs.num_envs

            dones = (terminations | truncations)
            masks = torch.tensor(1-dones, dtype=torch.int8).unsqueeze(-1).to(device)
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
            last_value = critic(obs)
        
        rollout_storage.compute_returns_and_advantages(
            last_value,
            args.gamma,
            args.gae_lambda,
            args.normalize_advantages
        )

        # ---- After rollout collection, before training:
        
        # After rollout collection, before training:
        old_actor = deepcopy(actor)

        # pass over the full rollout storage once - for TRPO it's enough
        for rollout_data in rollout_storage.get_mini_batch(batch_size=None):

            actions = rollout_data.actions

            with torch.no_grad():
                old_distribution = old_actor.get_action_distribution(rollout_data.states)
            
            distribution = actor.get_action_distribution(rollout_data.states)
            log_prob = distribution.log_prob(actions).sum(-1, keepdim=True)

            # Compute ratio (pi_theta / pi_theta_old) for surrogate objective, equal 1 for first iteration
            ratio = torch.exp(log_prob - rollout_data.log_probs)
            surrogate_objective = (ratio * rollout_data.advantages + 
                                   args.ent_coef * distribution.entropy().sum(-1, keepdim=True)).mean()
            
            # Compute Kullback-Leibler divergencbetween old and new policy using pytorch kl_divergence D_KL(pi_theta_old || pi_theta)
            kl_div = torch.distributions.kl.kl_divergence(old_distribution, distribution).mean()

            parameters = list(actor.parameters())

            policy_objective_gradients = compute_flattened_gradients(surrogate_objective, parameters, retain_graph=True)
            # Create graph, because we need to compute Hessian-vector product
            kl_div_gradients = compute_flattened_gradients(kl_div, parameters, create_graph=True)

            # Hessian-vector product function with the Fisher information matrix 
            HVP = partial(hessian_vector_product, parameters, kl_div_gradients)

            # Compute search direction using Conjugate Gradient and Hessian-vector product
            search_direction, cg_steps, residual = conjugate_gradient(
                HVP, 
                policy_objective_gradients, 
                args.cg_max_steps)
            
            # δθ = β * search_direction, where β is the step size, we searching max step size
            # β = √(2 * δ / (δθ)ᵀH(δθ))
            line_search_max_step_size = torch.sqrt(2 * args.target_kl / (search_direction @ HVP(search_direction)))
            max_step = line_search_max_step_size * search_direction

            # ---- Line search ----

            # Line search to find the best step size
            line_search_initial_step = 1.0
            is_line_search_successful = False
            original_parameters = [p.detach().clone() for p in parameters]
            with torch.no_grad():
                # Line seach backtracking
                for i in range(args.max_num_linesearch_steps):
                    # Applying the scaled step direction
                    n = 0
                    for param, original_param in zip(parameters, original_parameters):
                        n_params = param.numel()
                        param.data = (
                            original_param 
                            + line_search_initial_step
                            * max_step[n: n+n_params].view(param.shape)
                        )
                        n += n_params

                    # Recompute new distribution and log probabilties
                    distribution = actor.get_action_distribution(rollout_data.states)
                    log_prob = distribution.log_prob(actions).sum(-1, keepdim=True)

                    # Compute ratio (pi_theta / pi_theta_old) for surrogate objective and new surrogate objective
                    ratio = torch.exp(log_prob - rollout_data.log_probs)
                    new_surrogate_objective = (ratio * rollout_data.advantages 
                                               + args.ent_coef * distribution.entropy().sum(-1, keepdim=True)).mean()

                    # New KL-divergence between old and new policy
                    kl_div = torch.distributions.kl.kl_divergence(old_distribution, distribution).mean()

                    # Constraint criteria: 
                    # KL-divergence should be less than target KL
                    # and new surrogate objective should be greater than old surrogate objective
                    if kl_div < args.target_kl and new_surrogate_objective >= surrogate_objective:
                        is_line_search_successful = True
                        break

                    # If line search is not successful, reduce the step size
                    line_search_initial_step *= args.line_search_step_decay


                if not is_line_search_successful:
                    # If line search is not successful, revert to original parameters
                    for param, original_param in zip(parameters, original_parameters):
                        param.data = original_param.data.clone()
                    
        # ---- After line search completes ----
        value_losses = []
        # Critic update
        for _ in range(args.n_critic_updates):
            for rollout_data in rollout_storage.get_mini_batch(batch_size=args.batch_size):
                value_loss = F.mse_loss(critic(rollout_data.states), rollout_data.returns)
                value_losses.append(value_loss.item())
                value_optimzier.zero_grad()
                value_loss.backward()
                value_optimzier.step()

        # ---- After critic update completes ----

        # Log the training metrics
        logger.add_scalar("losses/cg_steps", cg_steps, global_step)
        logger.add_scalar("losses/residual", residual.mean().item(), global_step)
        logger.add_scalar("losses/surrogate_objective", new_surrogate_objective.item(), global_step)
        logger.add_scalar("losses/line_search_step", line_search_initial_step, global_step)
        logger.add_scalar("losses/value_loss",np.mean(value_losses), global_step)
        logger.add_scalar("losses/kl_divergance", kl_div.item(), global_step)
        logger.add_scalar("losses/is_line_search_success", int(is_line_search_successful), global_step)
        logger.add_scalar("losses/std", torch.exp(actor.log_std).mean().item(), global_step)
        logger.add_scalar("charts/FPS", int(global_step / (time.time() - start_time)), global_step)

        if (iteration+1) % eval_interval == 0:
            sync_normalization_stats(test_env, envs)
            eval_return = evaluate(test_env, actor)
            if best_eval_return < eval_return:
                best_eval_return = eval_return
                save_model(envs, actor, critic, logger, "best-torch.model")
            logger.add_scalar("eval/returns", eval_return, global_step)
            logger.log_stdout()


    # ---- After training completes ----
    save_model(envs, actor, critic, logger, "torch.model")

    
    envs.close()
    logger.close()
            
