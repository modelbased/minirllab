import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace
from torch.distributions.normal import Normal
from .normalise import NormaliseTorchScript
from .utils import init_layer, symlog, avg_weight_magnitude, count_dead_units
from .buffers import ReplayBufferPPO 

'''
    Based on https://github.com/vwxyzjn/cleanrl
    PPO agent with RPO option https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py
    Note: CleanRL version uses gym.wrappers.NormalizeReward(), which materially improves performance in some environments.
'''


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=64, rpo_alpha=0.0, device='cpu'):
        super(Actor, self).__init__()
        ''' agent input assumes shape: (batch, channels)'''

        self.device = device

        # Actor network
        self.fc0 = init_layer(nn.Linear(obs_dim, hidden_dim))
        self.fc1 = init_layer(nn.Linear(hidden_dim, hidden_dim))
        self.out = init_layer(nn.Linear(hidden_dim, action_dim), std=0.01) # Last layer init near zero (C57, https://arxiv.org/abs/2006.05990)
        self.nonlin = nn.Tanh() # tanh preferred (C55, https://arxiv.org/abs/2006.05990)

        # ReZero for deep networks https://arxiv.org/abs/2003.04887
        # Worse performance for such small nns
        # self.r0 = nn.Parameter(torch.zeros(1))

        # Adds stochasticity to action https://arxiv.org/abs/2212.07536
        # "rpo_alpha=0.5 –> better than PPO 93% of environments, rpo_alpha=0.01 –> better in 100%"
        # https://docs.cleanrl.dev/rl-algorithms/rpo/#implementation-details 
        self.rpo_alpha = rpo_alpha

        # Actor logstd (initial standard dev = 0.5 https://arxiv.org/abs/2006.05990)
        self.logstd = nn.Parameter(torch.ones(1, action_dim) * math.log(0.5))

    def forward(self, obs, action=None):
        x = self.nonlin(self.fc0(obs))
        x = self.nonlin(self.fc1(x))
        action_mean = self.out(x)

        # expand to match shape of action_mean (e.g. batch dim)
        action_logstd = self.logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)

        if action is None:
            probs  = Normal(action_mean, action_std, validate_args=False)
            action = probs.rsample()
        else: # RPO option
            z           = (torch.rand((action_mean.shape), device=self.device) - 0.5) * 2.0 * self.rpo_alpha # z = -rpo..+rpo
            action_mean = action_mean + z
            probs       = Normal(action_mean, action_std, validate_args=False)

        log_prob = probs.log_prob(action).sum(1)
        entropy  = probs.entropy().sum(1) # important: is there a lenght dim to consider in sum axis?

        # Consider environment action ranges, clip where appropriate or use tanh for ±1 if necessary
        return action, log_prob, entropy


class Critic(nn.Module):
    def __init__(self, obs_dim, hidden_dim=64):
        super(Critic, self).__init__()

        # Critic network (C47, independent critic performs better https://arxiv.org/abs/2006.05990)
        # Wider than actor preferred for critic (~4x) (https://arxiv.org/abs/2006.05990)
        self.fc0 = init_layer(nn.Linear(obs_dim, hidden_dim))
        self.fc1 = init_layer(nn.Linear(hidden_dim, hidden_dim)) 
        self.out = init_layer(nn.Linear(hidden_dim, 1), std=1.0) # Last layer init near one (C57, https://arxiv.org/abs/2006.05990)
        self.nonlin = nn.Tanh()

        # ReZero for deep networks https://arxiv.org/abs/2003.04887
        # Worse performance when networks are small
        # self.r0 = nn.Parameter(torch.zeros(1))

    def forward(self, obs):
        x = self.nonlin(self.fc0(obs))
        x = self.nonlin(self.fc1(x))
        v = self.out(x)
        return v


class Agent:
    def __init__(self, env_spec, buffer_size, num_env=1, device='cpu', seed=42):
        
        # Make global
        self.name           = "ppo_v01_baseline"     # name for logging
        self.obs_dim        = env_spec['obs_dim']    # environment inputs for agent
        self.action_dim     = env_spec['act_dim']    # agent outputs to environment
        self.act_max        = env_spec['act_max']    # action range, scalar or vector
        self.act_min        = env_spec['act_min']    # action range, scalar or vector
        self.device         = device                 # gpu or cpu

        # All seeds default to 42
        torch.manual_seed(torch.tensor(seed))
        torch.backends.cudnn.deterministic = True
        # torch.cuda.set_sync_debug_mode(1)            # Set to 1 to receive warnings

        
        # Hyperparameters
        hyperparameters = {
        "eps_clip"       : 0.2,           # (def: 0.2) clip parameter for PPO 
        "gamma"          : 0.99,          # (def: 0.99) Key parameter should be tuned for each environment https://arxiv.org/abs/2006.05990 (C20)
        "gae_lambda"     : 0.95,          # (def: 0.95) the lambda for the general advantage estimation
        "clip_coef"      : 0.2,           # (def: 0.2) try 0.1 to 0.5 depending on environment (https://arxiv.org/abs/2006.05990)
        "ent_coef"       : 0.0,           # (def: 0.0) coefficient of the entropy. 0.01 is better for WalkerHardcore.
        "max_grad_norm"  : 0.5,           # (def: 0.5) the maximum norm for the gradient clipping
        "max_kl"         : None,          # (def: 0.02) skip actor minibatch update if target exceeded. approx_kl generally < 0.02 when algo is working well
        "adam_lr"        : 0.0003,        # (def: 0.0003) Adam optimiser learning rate 0.0003 "safe default" but tuning recommeneded https://arxiv.org/abs/2006.05990
        "adam_eps"       : 1e-5,          # (def: 1e-7) Adam optimiser epsilon
        "weight_decay"   : 0.0,           # (def: 0.0) AdamW weight decay for regularisation (AdamW >> Adam)
        "norm_adv"       : False,         # (def: False) Normalise advantage of each batch (note not minibatch like CleanRL, lost source)
        "rpo_alpha"      : 0.0,           # (def: 0.0) In Box2D and Mujoco Gym environments a value of 0.5 was found to be worse. Perhaps due to differences between this and CleanRL's version.
        "gae_recalc"     : False,         # (def: False) recalculate GAE in each update epoch
        "update_epochs"  : 10,            # (def: 10) the K epochs to update the policy
        "mb_size"        : 64,            # (def: 64) the size of mini batches. CleanRL multiplies this by num_envs when vectorised
        "update_step"    : 2048,          # (def: 2048) perform update after this many environmental steps
        "dead_hurdle"    : 0.001,         # (def: 0.001) units with greater variation in output over one batch of data than this are not dead in plasticity terms
        "a_hidden_dim"   : 64,            # (def: 64) actor's hidden layers dim
        "c_hidden_dim"   : 64,            # (def: 64) critic's hidden layers dim
        }
        self.h = SimpleNamespace(**hyperparameters)

        # Loggin & debugging
        self.approx_kl          = 0       # estimated kl divergence
        self.clipfracs          = 0       # fraction of training data that triggered clipping objective  
        self.p_loss             = 0       # policy/actor loss
        self.v_loss             = 0       # value/critic loss
        self.entropy_loss       = 0       # provides entropy bonus through ent_coeff parameter
        self.ppo_updates        = 0       # number of minibatch updates performed
        self.actor_grad_norm    = 0       # actor model gradient norm
        self.critic_grad_norm   = 0       # critic model gradient norm
        self.actor_avg_wgt_mag  = 0       # average weight magnitude of model parameters
        self.critic_avg_wgt_mag = 0       # average weight magnitude of model parameters
        self.actor_dead_pct     = 0       # percentage of units which are dead by some threshold
        self.critic_dead_pct    = 0       # percentage of units which are dead by some threshold

        # Buffer only needs to be as large as update_step, so replay_buffer is redundand and kept for api compatibility
        self.rb = ReplayBufferPPO(self.obs_dim, self.action_dim, self.h.update_step, num_env, device=self.device)
        self.global_step = 0

        # Normalise state observations. Use symlog for reward "normalisation" – experimentally best results on Box2D and Mujoco
        # https://arxiv.org/pdf/2006.05990.pdf (C64) and https://arxiv.org/pdf/2005.12729.pdf
        self.normalise_observations = torch.jit.script(NormaliseTorchScript(self.obs_dim, num_env, device=self.device))
        # self.normalise_rewards      = torch.jit.script(NormaliseTorchScript(1, num_env, device=self.device))

        # Instantiate actor and critic networks, same optimiser parameters
        self.actor   = Actor(self.obs_dim, self.action_dim, hidden_dim=self.h.a_hidden_dim, rpo_alpha=self.h.rpo_alpha, device=device).to(self.device)
        self.optim_a = torch.optim.AdamW(self.actor.parameters(), lr=self.h.adam_lr, eps=self.h.adam_eps, weight_decay=self.h.weight_decay)
        
        self.critic  = Critic(self.obs_dim, hidden_dim=self.h.c_hidden_dim).to(self.device)
        self.optim_c = torch.optim.AdamW(self.critic.parameters(), lr=self.h.adam_lr, eps=self.h.adam_eps, weight_decay=self.h.weight_decay)


    # Values from environments must be pytorch tensors of shape (batch, channels)
    def choose_action(self, obs):
        obs = self.normalise_observations.new(obs) # normalise better than symlog (or none) for obs
        
        with torch.no_grad():
            action, logprob, _ = self.actor(obs)
        
        self.rb.store_choice(obs, action, logprob)
        return action # return shape is also (batch, channels)
    
    def store_transition(self, reward, done):
        reward = symlog(reward) #symlog better than Normalise (or None) for rewards
        self.rb.store_transition(reward, done)
        self.global_step += 1

    # Generalised advantage estimation
    def gae(self):
        b_obs, b_rewards, b_dones = self.rb.get_gae()
        b_size = self.rb.size

        with torch.no_grad():
            b_values   = self.critic(b_obs).squeeze(2) # latest critic values
            next_value = b_values[b_size - 1].reshape(1,-1)
        
            b_advantages = torch.zeros_like(b_rewards, device=self.device)
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
        
        # Bookeeping
        updated = False
        chronos_total = 0.0

        if self.global_step % self.h.update_step == 0 and self.global_step != 0:

            updated = True
            chronos_start = time.time()

            b_obs, b_actions, b_logprobs = self.rb.get_ppo_update()
            batch_end = self.rb.size - 1 # index to last element

            clipfracs = torch.zeros(0, device=self.device)
            self.ppo_updates = 0
            for epoch in range(self.h.update_epochs):

                # Update GAE once or in each epoch for fresh advantages (https://arxiv.org/abs/2006.05990)
                if (self.h.gae_recalc) or (epoch == 0):
                    b_returns, b_advantages, b_values = self.gae()
                    if self.h.norm_adv:
                        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

                b_inds = torch.randperm(batch_end, device=self.device) # shuffled indices of the batch
                for start in range(0, batch_end, self.h.mb_size):
                    end     = min(start + self.h.mb_size, batch_end)
                    mb_inds = b_inds[start:end]

                    # Get minibatch set
                    mb_obs          = b_obs[mb_inds]
                    mb_actions      = b_actions[mb_inds]  
                    mb_advantages   = b_advantages[mb_inds]

                    # From latest policy
                    _, newlogprob, entropy = self.actor(mb_obs, action=mb_actions)

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
                    self.entropy_loss = entropy.mean()
                    self.p_loss = torch.max(p_loss1, p_loss2).mean() - self.h.ent_coef * self.entropy_loss

                    # Value loss
                    mb_newvalues = self.critic(mb_obs).view(-1)
                    self.v_loss = F.mse_loss(mb_newvalues, b_returns[mb_inds])

                    # Skip this minibatch update just before applying .step() if max kl exceeded 
                    if self.h.max_kl is not None:
                        if self.approx_kl > self.h.max_kl: break

                    # Update actor model
                    self.optim_a.zero_grad()
                    self.p_loss.backward()
                    self.actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.h.max_grad_norm)
                    self.optim_a.step()
                    
                    # Update critic model
                    self.optim_c.zero_grad()
                    self.v_loss.backward()
                    self.critic_grad_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), self.h.max_grad_norm)
                    self.optim_c.step()
                    
                    self.ppo_updates += 1

            # the explained variance for the value function
            y_pred, y_true = b_values, b_returns
            var_y = torch.var(y_true)
            self.explained_var = 1 - torch.var(y_true - y_pred) / var_y
            
            # the fraction of the training data that triggered the clipped objective
            self.clipfracs = torch.mean(clipfracs)
        
            # Loss of Plasticity in Deep Continual Learning: https://arxiv.org/abs/2306.13812
            self.actor_avg_wgt_mag  = avg_weight_magnitude(self.actor)
            self.critic_avg_wgt_mag = avg_weight_magnitude(self.critic)
            
            b_obs = self.rb.get_obs()
            _, _, self.actor_dead_pct  = count_dead_units(self.actor, in1=b_obs, threshold=self.h.dead_hurdle)
            _, _, self.critic_dead_pct = count_dead_units(self.critic, in1=b_obs, threshold=self.h.dead_hurdle)
            
            # reset replay buffer and calc time taken
            self.rb.flush()
            chronos_total = time.time() - chronos_start

        return updated, chronos_total

