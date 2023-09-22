import math
import torch
import torch.nn as nn
from types import SimpleNamespace
from torch.distributions.normal import Normal
from .normalise import NormaliseTorchScript
from .activations import SymLog, LiSHT
from .utils import init_weights, symlog

'''
    Based on https://github.com/vwxyzjn/cleanrl
    PPO agent with RPO option https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py
'''

class ReplayBuffer:
    """
        Call reset or flush at the end of a update
        Must call .store_transition() to advance pointer
    """
    def __init__(self, obs_dim, action_dim, num_steps, num_env, device):
        
        self.obs_dim    = obs_dim
        self.action_dim = action_dim

        # Storage setup
        self.obs        = torch.zeros((num_steps, num_env) + (obs_dim,)).to(device)
        self.actions    = torch.zeros((num_steps, num_env) + (action_dim,)).to(device)
        self.logprobs   = torch.zeros((num_steps, num_env)).to(device)
        self.rewards    = torch.zeros((num_steps, num_env)).to(device)
        self.dones      = torch.zeros((num_steps, num_env)).to(device)

        # Size in steps 
        self.max_size   = num_steps        
        self.size       = 0

    def store_choice(self, obs, action, logprob):
        self.obs[self.size]        = obs
        self.actions[self.size]    = action
        self.logprobs[self.size]   = logprob
            
    def store_transition(self, reward, done):
        self.rewards[self.size]    = reward
        self.dones[self.size]      = done
        self.size += 1

    def get_ppo_update(self):
        s           = int(self.size)
        # Flatten the batch
        b_obs       = self.obs[0:s].reshape((-1,) + (self.obs_dim,)) 
        b_actions   = self.actions[0:s].reshape((-1,) + (self.action_dim,)) 
        b_logprobs  = self.logprobs[0:s].reshape(-1) 
        return b_obs, b_actions, b_logprobs

    def get_gae(self):
        s           = int(self.size)
        # Don't flatten for GAE
        b_obs       = self.obs[0:s]
        b_rewards   = self.rewards[0:s]
        b_dones     = self.dones[0:s]
        return b_obs, b_rewards, b_dones

    def reset(self):
        self.size = 0

    def flush(self):
        self.obs.zero_()
        self.actions.zero_()
        self.logprobs.zero_()
        self.rewards.zero_()
        self.dones.zero_()
        self.size   = 0


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=64, rpo_alpha=0.0, device='cpu'):
        super(ActorCritic, self).__init__()

        self.device = device
        critic_mult = 4

        # agent input is dim (batch, channels)

        # Actor network
        self.fc0p = init_weights(nn.Linear(obs_dim, hidden_dim))
        self.fc1p = init_weights(nn.Linear(hidden_dim, hidden_dim))
        self.outp = init_weights(nn.Linear(hidden_dim, action_dim), std=0.01) # Last layer init near zero (C57, https://arxiv.org/abs/2006.05990)
        self.nonlinp = nn.Tanh() # tanh preferred (C55, https://arxiv.org/abs/2006.05990)

        # ReZero for deep networks https://arxiv.org/abs/2003.04887
        self.r0p = nn.Parameter(torch.zeros(1))

        # Adds stochasticity to action https://arxiv.org/abs/2212.07536
        # rpo_alpha = 0.5 –> better than PPO 93% of environments, rpo_alpha=0.01 –> better in 100%
        # https://docs.cleanrl.dev/rl-algorithms/rpo/#implementation-details 
        self.rpo_alpha = rpo_alpha

        # Critic network (C47, independent critic performs better https://arxiv.org/abs/2006.05990)
        self.fc0v = init_weights(nn.Linear(obs_dim, hidden_dim * critic_mult)) # wider critic preferred (https://arxiv.org/abs/2006.05990)
        self.fc1v = init_weights(nn.Linear(hidden_dim * critic_mult, hidden_dim)) 
        self.outv = init_weights(nn.Linear(hidden_dim, 1), std=1.0) # Last layer init near one (C57, https://arxiv.org/abs/2006.05990)
        self.nonlinv = nn.Tanh()

        # ReZero for deep networks https://arxiv.org/abs/2003.04887
        # self.r0v = nn.Parameter(torch.zeros(6)) # needs constant hidden dim

        # Actor logstd (initial standard dev = 0.5 https://arxiv.org/abs/2006.05990)
        self.logstd = nn.Parameter(torch.ones(1, action_dim) * math.log(0.5))

    # Actor       
    def get_action(self, obs, action=None):
        # neural net
        x = self.nonlinp(self.fc0p(obs))
        x = x + self.r0p[0] * self.nonlinp(self.fc1p(x))
        action_mean = self.outp(x)

        # expand to match shape of action_mean (e.g. batch dim)
        action_logstd = self.logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)

        if action is None:
            probs  = Normal(action_mean, action_std)
            action = probs.sample()
        else: # new to RPO
            z           = torch.FloatTensor(action_mean.shape).uniform_(-self.rpo_alpha, self.rpo_alpha).to(self.device)
            action_mean = action_mean + z
            probs       = Normal(action_mean, action_std)

        log_prob = probs.log_prob(action).sum(1)
        entropy  = probs.entropy().sum(1) # important: is there a lenght dim to consider in sum axis?

        # beware environment action ranges, clip where appropriate or use tanh for ±1 if necessary
        return action, log_prob, entropy

    # Critic
    def get_value(self, obs):
        # neural net
        x = self.nonlinv(self.fc0v(obs))
        x = self.nonlinv(self.fc1v(x))
        v = self.outv(x)
        return v


class Agent:
    def __init__(self, obs_dim, action_dim, num_steps=8192, num_env=1, device='cpu', seed=42):
        
        # Make global
        self.name           = "acw_v01_baseline1"       # name for logging
        self.obs_dim        = obs_dim                   # environment inputs for agent
        self.action_dim     = action_dim                # agent outputs to environment
        self.device         = device                    # gpu or cpu

        # All seeds default to 42
        torch.manual_seed(torch.tensor(seed))
        torch.backends.cudnn.deterministic = True
        
        # Hyperparameters
        hyperparameters = {
        "eps_clip"       : 0.2,           # (def: 0.2) clip parameter for PPO 
        "gamma"          : 0.99,          # (def: 0.99) Key parameter should be tuned for each environment https://arxiv.org/abs/2006.05990 (C20)
        "gae_lambda"     : 0.95,          # (def: 0.95) the lambda for the general advantage estimation
        "clip_coef"      : 0.25,          # (def: 0.25) try 0.1 to 0.5 depending on environment (https://arxiv.org/abs/2006.05990)
        "ent_coef"       : 0.001,         # (def: 0.001) coefficient of the entropy. 0.01 is better for WalkerHardcore.
        "vf_coef"        : 0.5,           # (def: 0.5) coefficient of the value function
        "max_grad_norm"  : 0.5,           # (def: 0.5) the maximum norm for the gradient clipping
        "max_kl"         : 0.02,          # (def: 0.02) stop policy optimisation early early if target exceeded. approx_kl generally < 0.02 when algo is working well
        "adam_lr"        : 0.0003,        # (def: 0.0003) Adam optimiser learning rate 0.0003 "safe default" but tuning recommeneded https://arxiv.org/abs/2006.05990
        "adam_eps"       : 1e-7,          # (def: 1e-7) Adam optimiser epsilon
        "adam_betas"     : (0.9, 0.999),  # (def: (0.9, 0.999) Adam optimiser betas
        "weight_decay"   : 0.0,           # (def: 0.0) AdamW weight decay for regularisation
        "norm_adv"       : False,         # (def: False) Normalise advantage of each batch (note not minibatch, lost source)
        "rpo_alpha"      : 0.01,          # (def: 0.01) mean of the action distribution is perturbed using a random number drawn from a Uniform distribution
        "symlog_norm"    : False,         # (def: False) use symlog normalisation instead of mean/var
        "gae_recalc"     : True,          # (def: True) recalculate GAE in each update epoch
        "update_epochs"  : 10,            # (def: 10) the K epochs to update the policy
        "mb_size"        : 64,            # (def: 64) the size of mini batches. CleanRL multiplies this by num_envs when vectorised.
        }
        self.h = SimpleNamespace(**hyperparameters)

        # Loggin & debugging
        self.approx_kl      = 0
        self.clipfracs      = 0
        self.p_loss         = 0
        self.v_loss         = 0
        self.loss           = 0
        self.entropy_loss   = 0
        self.ppo_updates    = 0
        self.grad_norm      = 0

        # Storage setup
        self.replay_buffer = ReplayBuffer(self.obs_dim, self.action_dim, num_steps, num_env, device=self.device)

        # Normalise state observations and rewards
        # https://arxiv.org/pdf/2006.05990.pdf (C64) and https://arxiv.org/pdf/2005.12729.pdf
        self.normalise_observations = torch.jit.script(NormaliseTorchScript(obs_dim, num_env, device=self.device))
        self.normalise_rewards      = torch.jit.script(NormaliseTorchScript(1, num_env, device=self.device))
        
        # Instantiate actor critic network
        self.policy     = torch.compile(ActorCritic(self.obs_dim, self.action_dim, hidden_dim=64, rpo_alpha=self.h.rpo_alpha, device=device).to(self.device))
        self.optimizer  = torch.optim.AdamW(self.policy.parameters(), lr=self.h.adam_lr, eps=self.h.adam_eps, betas=self.h.adam_betas, weight_decay=self.h.weight_decay)

    # Values from environments must be pytorch tensors of shape (batch, channels)
    def choose_action(self, obs):
        if self.h.symlog_norm:           
            obs = symlog(obs)
        else:
            obs = self.normalise_observations.new(obs)
        
        with torch.no_grad():
            action, logprob, _ = self.policy.get_action(obs)
        
        self.replay_buffer.store_choice(obs, action, logprob)
        return action # return shape is also (batch, channels)
    
    def store_transition(self, reward, done):
        if self.h.symlog_norm:
            reward = symlog(reward)
        else:
            reward = self.normalise_rewards.new(reward)
        self.replay_buffer.store_transition(reward, done)

    # Generalised advantage estimation
    def gae(self):
        b_obs, b_rewards, b_dones = self.replay_buffer.get_gae()
        b_size = self.replay_buffer.size

        with torch.no_grad():
            b_values   = self.policy.get_value(b_obs).squeeze(2) # latest critic values
            next_value = b_values[b_size - 1].reshape(1,-1)
        
        b_advantages = torch.zeros_like(b_rewards).to(self.device)
        lastgaelam = 0
        for t in reversed(range(b_size)):
            if t == b_size - 1:
                nextnonterminal = 1.0 - b_dones[b_size - 1]
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - b_dones[t + 1]
                nextvalues = b_values[t + 1]
            delta = b_rewards[t] + self.h.gamma * nextvalues * nextnonterminal - b_values[t]
            b_advantages[t] = lastgaelam = delta + self.h.gamma * self.h.gae_lambda * nextnonterminal * lastgaelam        
        b_returns = b_advantages + b_values

        # Flatten on return
        return b_returns.reshape(-1), b_advantages.reshape(-1), b_values.reshape(-1)


    # Optimize actor/policy and critic/value networks
    def update(self):
        
        b_obs, b_actions, b_logprobs = self.replay_buffer.get_ppo_update()
        batch_end = self.replay_buffer.size - 1 # index to last element

        clipfracs = torch.zeros(0).to(self.device)
        self.ppo_updates = 0
        for epoch in range(self.h.update_epochs):

            # Update GAE once or in each epoch for fresh advantages (https://arxiv.org/abs/2006.05990)
            if (self.h.gae_recalc) or (epoch == 0):
                b_returns, b_advantages, b_values = self.gae()
                if self.h.norm_adv:
                    b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

            b_inds = torch.randperm(batch_end) # shuffled indices of the batch
            for start in range(0, batch_end, self.h.mb_size):
                end     = min(start + self.h.mb_size, batch_end)
                mb_inds = b_inds[start:end]

                # Get minibatch set
                mb_obs          = b_obs[mb_inds]
                mb_actions      = b_actions[mb_inds]  
                mb_advantages   = b_advantages[mb_inds]

                # From latest policy
                _, newlogprob, entropy = self.policy.get_action(mb_obs, action=mb_actions)
                mb_newvalues = self.policy.get_value(mb_obs)

                # Ratio for policy loss, logratio for estimating kl
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                # Debugging & info
                with torch.no_grad():
                    # approx Kullback-Leibler divergence, usually < 0.02 when policy not changing too quickly
                    self.approx_kl = ((ratio - 1) - logratio).mean()
                    
                    # fraction of training data that triggered clipped objective
                    frac       = torch.gt(torch.abs(ratio - 1.0), self.h.clip_coef)
                    clipfracs  = torch.cat([clipfracs, frac])

                # PPO policy loss
                p_loss1 = -mb_advantages * ratio
                p_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.h.clip_coef, 1 + self.h.clip_coef)
                self.p_loss = torch.max(p_loss1, p_loss2).mean()

                # Value loss
                mb_newvalues = mb_newvalues.view(-1)
                self.v_loss = 0.5 * ((mb_newvalues - b_returns[mb_inds]) ** 2).mean()
                
                # Entropy and overall loss
                self.entropy_loss = entropy.mean()
                self.loss = self.p_loss - self.h.ent_coef * self.entropy_loss + self.v_loss * self.h.vf_coef

                # Skip this minibatch update just before applying .step() if max kl exceeded 
                if self.approx_kl > self.h.max_kl: break

                self.optimizer.zero_grad()
                self.loss.backward()
                self.grad_norm = nn.utils.clip_grad_norm_(self.policy.parameters(), self.h.max_grad_norm)
                self.optimizer.step()
                self.ppo_updates += 1

        # the explained variance for the value function
        y_pred, y_true = b_values, b_returns
        var_y = torch.var(y_true)
        self.explained_var = torch.nan if var_y == 0 else 1 - torch.var(y_true - y_pred) / var_y
        
        # the fraction of the training data that triggered the clipped objective
        self.clipfracs = torch.mean(clipfracs)
    
        # reset replay buffer
        self.replay_buffer.flush()
