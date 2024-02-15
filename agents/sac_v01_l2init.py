import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace
from .utils import symlog, avg_weight_magnitude, count_dead_units

'''
    Based on https://github.com/vwxyzjn/cleanrl
    SAC agent https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py

    This L2Init variant tests concepts related to plasticity loss and increasing sample efficiency by "training harder"

    Influnced by the following papers:
    - Maintaining Plasticity in Continual Learning via Regenerative Regularization https://arxiv.org/abs/2308.11958
    - Loss of Plasticity in Deep Continual Learning https://arxiv.org/abs/2306.13812
    - Sample-Efficient Reinforcement Learning by Breaking the Replay Ratio Barrier https://openreview.net/forum?id=OpC-9aBBVJe
    – Bigger, Better, Faster: Human-level Atari with human-level efficiency https://arxiv.org/abs/2305.19452
'''


# Credit GPT4: Class to store initial parameters and compute L2 Init loss
# https://arxiv.org/abs/2308.11958
class L2NormRegularizer:
    def __init__(self, model, lambda_reg):
        self.lambda_reg = lambda_reg
        self.initial_params = {name: param.detach().clone() for name, param in model.named_parameters()}
        
    def loss(self, model):
        l2_loss = 0.0
        for name, param in model.named_parameters():
            initial_param = self.initial_params[name]
            l2_loss += ((param - initial_param)**2).sum() # squared L2 Norm
        return self.lambda_reg * l2_loss


class ReplayBuffer:
    """
        Call reset or flush at the end of a update
        Must call .store_transition() to advance pointer
    """
    def __init__(self, obs_dim, action_dim, max_size, num_env, device):
        
        self.obs_dim    = obs_dim
        self.action_dim = action_dim
        self.num_env    = num_env

        # Storage setup
        self.obs        = torch.zeros((max_size, num_env) + (obs_dim,)).to(device)
        self.actions    = torch.zeros((max_size, num_env) + (action_dim,)).to(device)
        self.rewards    = torch.zeros((max_size, num_env)).to(device)
        self.dones      = torch.zeros((max_size, num_env)).to(device)

        # Size in steps 
        self.max_size   = max_size        
        self.ptr        = 0
        self.size       = 0

    def store_choice(self, obs, action):
        self.obs[self.ptr]        = obs
        self.actions[self.ptr]    = action
            
    def store_transition(self, reward, done):
        self.rewards[self.ptr]    = reward
        self.dones[self.ptr]      = done
        self.ptr = (self.ptr + 1) % self.max_size 
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        s    = int(self.size) - 2                 # allow for obs_next

        # Guarantees uniqueness, but slow and gets slower with time
        # inds = torch.randperm(s)                  # shuffled index over entire buffer
        # inds = inds[:batch_size // self.num_env]  # limit sample to batch size when vectorised
        
        # Faster, but no uniqueness guarantee
        inds = torch.randint(0, s, (batch_size // self.num_env,))

        # Flatten the batch (global_step, env_num, channels) -> (b_step, channels)
        b_obs       = self.obs[inds].reshape((-1,) + (self.obs_dim,))
        b_actions   = self.actions[inds].reshape((-1,) + (self.action_dim,))
        b_obs_next  = self.obs[inds + 1].reshape((-1,) + (self.obs_dim,)) 
        b_rewards   = self.rewards[inds].reshape(-1) 
        b_dones     = self.dones[inds].reshape(-1) 
        return b_obs, b_actions, b_obs_next, b_rewards, b_dones

    def plasticity_data(self, batch_size):
        inds        = torch.randint(0, self.size - 1, (batch_size // self.num_env,))
        b_obs       = self.obs[inds].reshape((-1,) + (self.obs_dim,))
        b_actions   = self.actions[inds].reshape((-1,) + (self.action_dim,))
        return b_obs, b_actions

    def reset(self):
        self.ptr  = 0
        self.size = 0

    def flush(self):
        self.obs.zero_()
        self.actions.zero_()
        self.rewards.zero_()
        self.dones.zero_()
        self.ptr  = 0
        self.size = 0


class SoftQNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.fc0    = nn.Linear(obs_dim + action_dim, hidden_dim)
        self.fc1    = nn.Linear(hidden_dim, hidden_dim)
        self.fc2    = nn.Linear(hidden_dim, 1)
        self.nonlin = nn.LeakyReLU()
        self.norm   = nn.LayerNorm(hidden_dim)

    def forward(self, x, a=None):
        if a is not None:
            x = torch.cat([x, a], 1)
        x = self.norm(self.nonlin(self.fc0(x)))
        x = self.norm(self.nonlin(self.fc1(x)))
        x = self.fc2(x)
        return x


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.fc0        = nn.Linear(obs_dim, hidden_dim)
        self.fc1        = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean    = nn.Linear(hidden_dim, action_dim)
        self.fc_logstd  = nn.Linear(hidden_dim, action_dim)
        self.nonlin     = nn.LeakyReLU()
        self.norm       = nn.LayerNorm(hidden_dim)

        # HACK: Manual scaling to environment
        self.action_scale   = 1.0
        self.action_bias    = 0.0
        self.log_std_max    = 2
        self.log_std_min    = -5

    def forward(self, x):
        x       = self.norm(self.nonlin(self.fc0(x)))
        x       = self.norm(self.nonlin(self.fc1(x)))
        mean    = self.fc_mean(x)
        log_std = self.fc_logstd(x)

        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)  # From SpinUp / Denis Yarats
        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        
        return action, log_prob, mean


class Agent:
    def __init__(self, obs_dim, action_dim, run_steps=int(1e6), num_env=1, device='cpu', seed=42, l2init_lambda=0.0, rr=1, q_lr=1e-3, actor_lr=3e-4):
        
        # Make global
        self.name           = "sac_v01_l2init"          # name for logging
        self.obs_dim        = obs_dim                   # environment inputs for agent
        self.action_dim     = action_dim                # agent outputs to environment
        self.device         = device                    # gpu or cpu

        # All seeds default to 42
        torch.manual_seed(torch.tensor(seed))
        torch.backends.cudnn.deterministic = True

        # Hyperparameters
        hyperparameters = {
        "gamma"             : 0.99,     # (def: 0.99) Discount factor
        "q_lr"              : q_lr,     # (def: 1e-3) Q learning rate 
        "actor_lr"          : actor_lr, # (def: 3e-4) Policy learning rate
        "learn_start"       : int(5e3), # (def: 5e3) Start updating policies after this many global steps
        "batch_size"        : 256,      # (def: 256) Batch size of sample from replay buffer
        "policy_freq"       : 2,        # (def: 2) the frequency of training policy (delayed)
        "target_net_freq"   : 1,        # (def: 1) Denis Yarats' implementation delays this by 2
        "tau"               : 0.005,    # (def: 0.005) target smoothing coefficient
        "dead_hurdle"       : 0.01,     # (def: 0.01) units with greater variation in output over one batch of data than this are not dead in plasticity terms
        "symlog_norm"       : False,    # (def: False) normalise obs and rewards with symlog
        "actor_hidden_dim"  : 256,      # (def: 256) size of actor's hidden layer(s)
        "q_hidden_dim"      : 256,      # (def: 256) size of Q's hidden layer(s)
        "l2init_lambda"     : l2init_lambda,
        "replay_ratio"      : round(rr),
        }
        self.h = SimpleNamespace(**hyperparameters)

        # Loggin & debugging
        self.qf1_a_values       = torch.tensor([0.0])
        self.qf2_a_values       = torch.tensor([0.0])
        self.qf1_loss           = 0
        self.qf2_loss           = 0
        self.qf_loss            = 0
        self.actor_loss         = 0
        self.alpha_loss         = 0
        self.actor_avg_wgt_mag  = 0     # average weight magnitude of model parameters
        self.qf1_avg_wgt_mag    = 0     # average weight magnitude of model parameters
        self.qf2_avg_wgt_mag    = 0     # average weight magnitude of model parameters
        self.actor_dead_pct     = 0     # percentage of units which are dead by some threshold
        self.qf1_dead_pct       = 0     # percentage of units which are dead by some threshold
        self.qf2_dead_pct       = 0     # percentage of units which are dead by some threshold

        # Instantiate actor and Q networks, optimisers
        self.qf1        = SoftQNetwork(obs_dim, action_dim, self.h.q_hidden_dim).to(device)
        self.qf2        = SoftQNetwork(obs_dim, action_dim, self.h.q_hidden_dim).to(device)
        self.qf1_target = SoftQNetwork(obs_dim, action_dim, self.h.q_hidden_dim).to(device)
        self.qf2_target = SoftQNetwork(obs_dim, action_dim, self.h.q_hidden_dim).to(device)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())
        self.q_optimizer = torch.optim.AdamW(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=self.h.q_lr)
        
        self.actor           = Actor(obs_dim, action_dim, self.h.q_hidden_dim).to(device)
        self.actor_optimizer = torch.optim.AdamW(list(self.actor.parameters()), lr=self.h.actor_lr)

        # Use automatic entropy tuning
        self.target_entropy     = -torch.prod(torch.Tensor(action_dim).to(device)).item()
        self.log_alpha          = torch.zeros(1, requires_grad=True, device=device)
        self.alpha              = self.log_alpha.exp().item()
        self.alpha_optimizer    = torch.optim.AdamW([self.log_alpha], lr=self.h.q_lr)

        # Storage setup
        self.rb = ReplayBuffer(self.obs_dim, self.action_dim, run_steps, num_env, device=self.device)
        self.global_step = 0

        # L2 Init for each model
        self.l2_qf1     = L2NormRegularizer(self.qf1, lambda_reg=l2init_lambda)
        self.l2_qf2     = L2NormRegularizer(self.qf2, lambda_reg=l2init_lambda)
        self.l2_actor   = L2NormRegularizer(self.actor, lambda_reg=l2init_lambda)

    def choose_action(self, obs):
        if self.h.symlog_norm:
            obs = symlog(obs)
        with torch.no_grad():
            action, _, _ = self.actor.get_action(obs)
        self.rb.store_choice(obs, action)
        return action

    def store_transition(self, reward, done):
        if self.h.symlog_norm:
            reward = symlog(reward)
        self.rb.store_transition(reward, done)
        self.global_step += 1


    def update(self):
        
        ''' Call every step from learning script, agent decides if it is time to update '''

        # Bookeeping
        updated = False
        chronos_total = 0.0

        if self.global_step > self.h.learn_start:

            updated = True
            chronos_start = time.time()

            for replay in range(0,self.h.replay_ratio):
    
                b_obs, b_actions, b_obs_next, b_rewards, b_dones = self.rb.sample(self.h.batch_size)

                with torch.no_grad():
                    next_state_actions, next_state_log_pi, _ = self.actor.get_action(b_obs_next)
                    qf1_next_target = self.qf1_target(b_obs_next, next_state_actions)
                    qf2_next_target = self.qf2_target(b_obs_next, next_state_actions)
                    min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
                    next_q_value = b_rewards.flatten() + (1 - b_dones.flatten()) * self.h.gamma * (min_qf_next_target).view(-1)

                self.qf1_a_values = self.qf1(b_obs, b_actions).view(-1)
                self.qf2_a_values = self.qf2(b_obs, b_actions).view(-1)
                self.qf1_loss = F.mse_loss(self.qf1_a_values, next_q_value)
                self.qf2_loss = F.mse_loss(self.qf2_a_values, next_q_value)
                self.qf_loss = self.qf1_loss + self.qf2_loss

                # L2Init loss
                self.qf_loss += self.l2_qf1.loss(self.qf1)
                self.qf_loss += self.l2_qf2.loss(self.qf2)

                # consider reducing learning rate (instability -> exploding gradients)
                if torch.isnan(self.qf_loss):
                    # print("QF loss is nan, skipping update")
                    break

                self.q_optimizer.zero_grad()
                self.qf_loss.backward()
                self.q_optimizer.step()

                # update actor network and alpha parameter
                if self.global_step % self.h.policy_freq == 0:  # TD 3 Delayed update support
                    for _ in range(self.h.policy_freq):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                        pi, log_pi, _ = self.actor.get_action(b_obs)
                        qf1_pi = self.qf1(b_obs, pi)
                        qf2_pi = self.qf2(b_obs, pi)
                        min_qf_pi = torch.min(qf1_pi, qf2_pi).view(-1)
                        self.actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

                        # L2 Init loss
                        self.actor_loss += self.l2_actor.loss(self.actor)

                        # consider reducing learning rate (instability -> exploding gradients)
                        if torch.isnan(self.actor_loss):
                            # print("Actor loss is nan, skipping update")
                            break

                        self.actor_optimizer.zero_grad()
                        self.actor_loss.backward()
                        self.actor_optimizer.step()

                        # Autotune alpha 
                        with torch.no_grad():
                            _, log_pi, _ = self.actor.get_action(b_obs)
                        self.alpha_loss = (-self.log_alpha.exp() * (log_pi + self.target_entropy)).mean()

                        self.alpha_optimizer.zero_grad()
                        self.alpha_loss.backward()
                        self.alpha_optimizer.step()
                        self.alpha = self.log_alpha.exp().item()

                # update the target networks
                if self.global_step % self.h.target_net_freq == 0:
                    for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
                        target_param.data.copy_(self.h.tau * param.data + (1 - self.h.tau) * target_param.data)
                    for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
                        target_param.data.copy_(self.h.tau * param.data + (1 - self.h.tau) * target_param.data)
            
            # Plasticity metrics occasionally
            if self.global_step % 2048 == 0 or self.global_step == self.h.learn_start:
                self.actor_avg_wgt_mag  = avg_weight_magnitude(self.actor)
                self.qf1_avg_wgt_mag    = avg_weight_magnitude(self.qf1)
                self.qf2_avg_wgt_mag    = avg_weight_magnitude(self.qf2)

                b_obs, b_actions = self.rb.plasticity_data(2048) # a representative sample 
                _, _, self.actor_dead_pct   = count_dead_units(self.actor, b_obs, self.h.dead_hurdle)
                _, _, self.qf1_dead_pct     = count_dead_units(self.qf1, torch.cat((b_obs, b_actions), dim=1), self.h.dead_hurdle)
                _, _, self.qf2_dead_pct     = count_dead_units(self.qf2, torch.cat((b_obs, b_actions), dim=1), self.h.dead_hurdle)

            chronos_total = time.time() - chronos_start
        
        return updated, chronos_total
    