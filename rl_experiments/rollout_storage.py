import torch 
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import numpy as np 
from collections import namedtuple


RolloutBatch = namedtuple("RolloutBatch", 
    ["states", "actions", "values", "log_probs", "returns", "advantages"])

class RolloutStorage:
    """
    Rollout buffer used in on-policy algorithms like A2C/TRPO/PPO.
    It corresponds to ``buffer_size`` transitions collected
    using the current policy or n-steps as we like to call it in TD.
    This experience will be discarded after the policy update.
    In order to use PPO objective, we also store the current value of each state
    and the log probability of each taken action.

    It is only involved in policy and value function training but not action selection.
    """
    def __init__(self, 
                 obs_shape: any,
                 action_dim: any,
                 num_steps: int = 1, 
                 n_envs: int = 1,
                 device: torch.device = torch.device("cpu")):
        
        self.num_steps = num_steps
        self.n_envs = n_envs

        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.device = device

        # setup all the data   
        self.reset()  
    
    def reset(self) -> None:
        """
        Call reset, whenever we starting to collect next n-step rollout of data. 
        """
        self.obs = torch.zeros(self.num_steps, self.n_envs, *self.obs_shape, dtype=torch.float32, device=self.device)
        self.rewards = torch.zeros(self.num_steps, self.n_envs, 1,  dtype=torch.float32, device=self.device)
        self.values = torch.zeros(self.num_steps, self.n_envs,  1,  dtype=torch.float32, device=self.device)
        self.log_probs = torch.zeros(self.num_steps, self.n_envs, 1, dtype=torch.float32, device=self.device)
        self.actions = torch.zeros(self.num_steps, self.n_envs, *self.action_dim, dtype=torch.float32, device=self.device)
        self.masks = torch.ones(self.num_steps, self.n_envs, 1, dtype=torch.int8, device=self.device)
        self.truncates = torch.zeros(self.num_steps, self.n_envs, 1, dtype=torch.bool, device=self.device)
        self.advantages = torch.zeros(self.num_steps, self.n_envs, 1, dtype=torch.float32, device=self.device)
        self.returns = torch.zeros(self.num_steps, self.n_envs, 1, dtype=torch.float32, device=self.device)

        self.step = 0
        self.generator_ready = False
    
    def add(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        log_probs: torch.Tensor,
        values: torch.Tensor,
        rewards: torch.Tensor,
        masks: torch.Tensor,
        truncates: torch.Tensor,
    ) -> None:
        """
        :param obs: Observations
        :param action: Actions
        :param log_probs: log probability of the action
            following the current policy.
        :param values: estimated value of the current state
            following the current policy.
        :param entropies: entropy calculated for the current step
        :param rewards: rewards
        :param masks: indicate env is still active (terminated or truncated)
        :param truncated: indicate env is truncated, needed to calculated Advantages correctly
        """
        self.obs[self.step].copy_(obs)
        self.actions[self.step].copy_(actions)
        self.log_probs[self.step].copy_(log_probs) 
        self.values[self.step].copy_(values)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step].copy_(masks)
        self.truncates[self.step].copy_(truncates)

        self.step = (self.step + 1) % self.num_steps # hopefully thios % is actually not needed here

    def compute_returns_and_advantages(
            self,
            last_values: torch.Tensor,
            gamma: float = 0.99,
            gae_lambda: float = 1.0,
            normalize: bool = True) -> None:
        """
        Post-processing step: compute the advantages A using TD(n) error method, to use in the gradient calculation in future
            - TD(1) or A_1 is one-step estimate with bootstrapping delta_t = (r_{t+1} + gamma * v(s_{t+1}) - v(s_t))
            ....
            - TD(n) or A_n is n-step estimate with bootstrapping SUM_{l=0}^{n}(gamma^{l}*delta_{t+l})
               (r_{t+1} + gamma*r_{t+2} + gamma^2*r_{t+3} + .....+ gamma^(n+1)*v(s_{t+n+1}) - v(s_t))
        
        We using Generalized Advantage Estimation, in this case advantage calculated next way:

            - A_t^gae(gamma,lambda) = SUM_{l=0}^{\infty}( (gamma*lambda)^{l} * \delta_{t+l})

        :param last_values: state values estimation for the last step (one for each env)
        :param gamma: discount to be used in reward estimation
        :param use_gae:  use Generalized Advantage Estimation 
        :param gae_lambada: factor for trade off of bias vs variance for GAE
        """
        gae = 0
        for step in reversed(range(self.num_steps)):
            if step == self.num_steps - 1:
                next_values = last_values.detach()
            else:
                next_values = self.values[step+1].detach()

            # Handle truncated episodes by incorporating next state value
            # https://github.com/DLR-RM/stable-baselines3/issues/633#issuecomment-961870101
            adjusted_rewards = self.rewards[step].clone()  # Start with original rewards
            adjusted_rewards[self.truncates[step]] += gamma * next_values[self.truncates[step]]
            
            delta = adjusted_rewards + gamma * self.masks[step] * next_values - self.values[step].detach() #td_error
            gae = delta + gamma * gae_lambda * self.masks[step] * gae
            self.advantages[step] = gae.detach()

        #R_t = A_t{GAE} + V(s_t) 
        self.returns = self.advantages + self.values.detach()    

        # Normalize advantages to reduce skewness and improve convergence
        if normalize:
            self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def get_mini_batch(self, batch_size):
        
        indices = np.random.permutation(self.num_steps * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            self.obs_ = self._swap_and_flatten(self.obs)
            self.actions_ = self._swap_and_flatten(self.actions)
            self.values_ = self._swap_and_flatten(self.values)
            self.log_probs_ = self._swap_and_flatten(self.log_probs)
            self.returns_ = self._swap_and_flatten(self.returns)
            self.advantages_ = self._swap_and_flatten(self.advantages)

            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.num_steps * self.n_envs

        start_idx = 0
        while start_idx < self.num_steps * self.n_envs:
            batch_indices = indices[start_idx:start_idx + batch_size]
            batch = RolloutBatch(
                states=self.obs_[batch_indices],
                actions=self.actions_[batch_indices],
                values=self.values_[batch_indices],
                log_probs=self.log_probs_[batch_indices],
                returns=self.returns_[batch_indices],
                advantages=self.advantages_[batch_indices]
            )
            yield batch
            start_idx += batch_size
    
    # same to above code but running on data stored directly using view
    def get_mini_batch_sampler(self, num_mini_batch):
        batch_size = self.num_steps * self.n_envs

        mini_batch_size = batch_size // num_mini_batch

        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            mini_batch_size,
            drop_last=True)
    
        for indices in sampler:
            obs_batch = self.obs.view(-1, *self.obs.size()[2:])[indices]
            actions_batch = self.actions.view(-1, self.actions.size(-1))[indices]
            value_preds_batch = self.values.view(-1, 1)[indices]
            return_batch = self.returns.view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1,1)[indices]
            advantages_batch = self.advantages.view(-1, 1)[indices]

            yield obs_batch, actions_batch, value_preds_batch, \
                  old_action_log_probs_batch, return_batch, advantages_batch

    def _swap_and_flatten(self, tensor):
        """
        Swap the first two axes and flatten the tensor.
        """
        shape = tensor.shape  # e.g., (num_steps, n_envs, feature_dim)
        return tensor.swapaxes(0, 1).reshape(-1, *shape[2:])  # Flatten into (num_steps * n_envs, feature_dim)