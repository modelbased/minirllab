import time, math
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace
from .utils import avg_weight_magnitude, count_dead_units, init_model
from .buffers import ReplayBufferSAC

'''
    Based on https://github.com/vwxyzjn/cleanrl
    SAC agent https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py

    DroQ extensions to SAC based on:
    - Original paper: https://openreview.net/forum?id=xCVJMsPv3RT
    – Berkely implemetation on a quadruped (Walk In the Park (WITP)): https://arxiv.org/abs/2208.07860

    Plasticity metrics and regularisation influnced by the following papers:
    - Maintaining Plasticity in Continual Learning via Regenerative Regularization https://arxiv.org/abs/2308.11958
    - Loss of Plasticity in Deep Continual Learning https://arxiv.org/abs/2306.13812
    - Sample-Efficient Reinforcement Learning by Breaking the Replay Ratio Barrier https://openreview.net/forum?id=OpC-9aBBVJe
    – Bigger, Better, Faster: Human-level Atari with human-level efficiency https://arxiv.org/abs/2305.19452
'''


class SoftQNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.Dropout(0.01), # Droq uses 0.01 for dropout
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.Dropout(0.01),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # WITP and DroQ implementations both use xavier_uniform initialisations
        # Correct initialisation is important for stable training as replay ratio increases 
        init_model(self.mlp, init_method='xavier_uniform_')

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = self.mlp(x)
        return x


class Actor(nn.Module):
    def __init__(self, env_spec, hidden_dim=256):
        super().__init__()
        obs_dim = env_spec['obs_dim']
        act_dim = env_spec['act_dim']

        self.fc0       = nn.Linear(obs_dim, hidden_dim)
        self.fc1       = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean   = nn.Linear(hidden_dim, act_dim)
        self.fc_logstd = nn.Linear(hidden_dim, act_dim)
        self.nonlin    = nn.ReLU() # If the non-linearity has state (e.g. trainable parameter) we need one per use

        action_high       = nn.Parameter(torch.tensor(env_spec['act_max']), requires_grad=False)
        action_low        = nn.Parameter(torch.tensor(env_spec['act_min']), requires_grad=False)
        self.action_scale = nn.Parameter((action_high - action_low) * 0.5, requires_grad=False)
        self.action_bias  = nn.Parameter((action_high + action_low) * 0.5, requires_grad=False)
        self.log_std_max  = 2
        self.log_std_min  = -5

    def forward(self, x):
        x       = self.nonlin(self.fc0(x))
        x       = self.nonlin(self.fc1(x))
        mean    = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)  # From SpinUp / Denis Yarats
        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std      = log_std.exp()
        normal   = torch.distributions.Normal(mean, std, validate_args=False) # validation forces a cuda<>cpu sync
        x_t      = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t      = torch.tanh(x_t)
        action   = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob  = log_prob.sum(1, keepdim=True)
        mean      = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


class Agent:
    def __init__(self, 
                 env_spec, 
                 buffer_size=int(1e6), 
                 num_env=1, 
                 device='cpu', 
                 seed=42, 
                 rr=20, # DroQ paper
                 q_lr=1e-3, 
                 actor_lr=3e-4, 
                 alpha_lr=1e-3,
                 ):
        
        # Make global
        self.name           = "sac_v01_droq"          # name for logging
        self.obs_dim        = env_spec['obs_dim']    # environment inputs for agent
        self.action_dim     = env_spec['act_dim']    # agent outputs to environment
        self.act_max        = env_spec['act_max']    # action range, scalar or vector
        self.act_min        = env_spec['act_min']    # action range, scalar or vector
        self.device         = device                 # gpu or cpu

        # All seeds default to 42
        torch.manual_seed(torch.tensor(seed))
        torch.backends.cudnn.deterministic = True
        # torch.cuda.set_sync_debug_mode(1)               # Set to 1 to receive warnings

        # Hyperparameters
        hyperparameters = {
        "gamma"             : 0.99,     # (def: 0.99) Discount factor
        "q_lr"              : q_lr,     # (def: 1e-3) Q learning rate 
        "a_lr"              : actor_lr, # (def: 3e-4) Policy learning rate
        "alpha_lr"          : alpha_lr, # (def: 1e-3) alpha auto entropoty tuning learning rate
        "learn_start"       : int(5e3), # (def: 5e3) Start updating policies after this many global steps
        "batch_size"        : 256,      # (def: 256) Batch size of sample from replay buffer
        "policy_freq"       : 2,        # (def: 2) the frequency of training policy (delayed)
        "target_net_freq"   : 1,        # (def: 1) Denis Yarats' implementation delays this by 2
        "tau"               : 0.005,    # (def: 0.005) target smoothing coefficient
        "dead_hurdle"       : 0.01,     # (def: 0.01) units with greater variation in output over one batch of data than this are not dead in plasticity terms
        "a_hidden_dim"      : 256,      # (def: 256) size of actor's hidden layer(s)
        "q_hidden_dim"      : 256,      # (def: 256) size of Q's hidden layer(s)
        "q_max_grad_norm"   : 1000.0,   # (def: 1000) qf maximum norm for the gradient clipping
        "a_max_grad_norm"   : 1000.0,   # (def: 1000) actor and alpha maximum norm for the gradient clipping
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
        # AdamW (may have) resulted in more stable training than Adam
        self.qf1        = SoftQNetwork(self.obs_dim, self.action_dim, self.h.q_hidden_dim).to(device)
        self.qf2        = SoftQNetwork(self.obs_dim, self.action_dim, self.h.q_hidden_dim).to(device)
        self.qf1_target = SoftQNetwork(self.obs_dim, self.action_dim, self.h.q_hidden_dim).to(device)
        self.qf2_target = SoftQNetwork(self.obs_dim, self.action_dim, self.h.q_hidden_dim).to(device)
        self.q_optim    = torch.optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=self.h.q_lr)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())
        
        self.actor       = Actor(env_spec, self.h.a_hidden_dim).to(device)
        self.actor_optim = torch.optim.Adam(list(self.actor.parameters()), lr=self.h.a_lr)
        init_model(self.actor, init_method='xavier_uniform_') # Correct initialisation essential for stable training with increasing replay ratio
        
        # Use automatic entropy tuning
        # WITP initialises alpha = 0.1, this seems to help stabilise training
        self.target_entropy = -torch.prod(torch.Tensor((self.action_dim,)).to(device)).item()
        self.log_alpha      = torch.tensor((math.log(0.1)), requires_grad=True, device=device)
        self.alpha          = self.log_alpha.exp().item()
        self.alpha_optim    = torch.optim.Adam([self.log_alpha], lr=self.h.alpha_lr)

        # Storage setup
        self.rb          = ReplayBufferSAC(self.obs_dim, self.action_dim, buffer_size, num_env, device=self.device)
        self.global_step = 0


    def choose_action(self, obs):
        # Random uniform actions before learn_start can speed up training over using the agent's inital randomness.
        if self.global_step < self.h.learn_start:
            # actions are rand_uniform of shape (obs_batch_size, action_dim)
            action = (torch.rand((obs.size(0), self.action_dim), device=self.device) - 0.5) * 2.0  # rand_uniform -1..+1
            action = action * self.actor.action_scale + self.actor.action_bias # apply scale and bias
        else:
            with torch.no_grad():
                action, _, _ = self.actor.get_action(obs)
        self.rb.store_choice(obs, action)
        return action

    def store_transition(self, reward, done):
        self.rb.store_transition(reward, done)
        self.global_step += 1


    def update(self):
        
        ''' Call every step from learning script, agent decides if it is time to update '''

        # Bookeeping
        updated       = False
        chronos_total = 0.0

        if self.global_step > self.h.learn_start:
            updated       = True
            chronos_start = time.time()

            for replay in range(0,self.h.replay_ratio):
    
                b_obs, b_actions, b_obs_next, b_rewards, b_dones = self.rb.sample(self.h.batch_size)

                with torch.no_grad():
                    next_state_actions, next_state_log_pi, _ = self.actor.get_action(b_obs_next)
                    qf1_next_target    = self.qf1_target(b_obs_next, next_state_actions)
                    qf2_next_target    = self.qf2_target(b_obs_next, next_state_actions)
                    min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
                    next_q_value       = b_rewards.flatten() + (1 - b_dones.flatten()) * self.h.gamma * (min_qf_next_target).view(-1)

                self.qf1_a_values = self.qf1(b_obs, b_actions).view(-1)
                self.qf2_a_values = self.qf2(b_obs, b_actions).view(-1)
                self.qf1_loss     = F.mse_loss(self.qf1_a_values, next_q_value)
                self.qf2_loss     = F.mse_loss(self.qf2_a_values, next_q_value)
                self.qf_loss      = self.qf1_loss + self.qf2_loss

                self.q_optim.zero_grad()
                self.qf_loss.backward()
                nn.utils.clip_grad_norm_(self.qf1.parameters(), self.h.q_max_grad_norm) # Almost certainly unecessary
                nn.utils.clip_grad_norm_(self.qf2.parameters(), self.h.q_max_grad_norm)
                self.q_optim.step()

                # update the target networks within repay loop
                if self.global_step % self.h.target_net_freq == 0:
                    for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
                        target_param.data.copy_(self.h.tau * param.data + (1.0 - self.h.tau) * target_param.data)
                    for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
                        target_param.data.copy_(self.h.tau * param.data + (1.0 - self.h.tau) * target_param.data)

            # Replay Ratio does not apply to actor nor to alpha 
            # update actor network and alpha parameter
            b_obs, _, _, _, _ = self.rb.sample(self.h.batch_size) 

            if self.global_step % self.h.policy_freq == 0:  # TD 3 Delayed update support
                for _ in range(self.h.policy_freq):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _   = self.actor.get_action(b_obs)
                    qf1_pi          = self.qf1(b_obs, pi) #BUG: self.qfx.eval() should be used before this line due to Dropout()
                    qf2_pi          = self.qf2(b_obs, pi)
                    min_qf_pi       = torch.min(qf1_pi, qf2_pi).view(-1)
                    self.actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

                    self.actor_optim.zero_grad()
                    self.actor_loss.backward()
                    nn.utils.clip_grad_norm_(list(self.actor.parameters()), self.h.a_max_grad_norm) # almost certainly unecessary
                    self.actor_optim.step()

                    # Autotune alpha 
                    with torch.no_grad():
                        _, log_pi, _ = self.actor.get_action(b_obs)
                    self.alpha_loss = (-self.log_alpha.exp() * (log_pi + self.target_entropy)).mean()

                    self.alpha_optim.zero_grad()
                    self.alpha_loss.backward()
                    nn.utils.clip_grad_norm_([self.log_alpha], self.h.a_max_grad_norm) # almost certainly unecessary
                    self.alpha_optim.step()
                    self.alpha = self.log_alpha.exp().detach().clone()
            
            # Plasticity metrics occasionally
            if self.global_step % 2048 == 0 or self.global_step == self.h.learn_start:
                self.actor_avg_wgt_mag  = avg_weight_magnitude(self.actor)
                self.qf1_avg_wgt_mag    = avg_weight_magnitude(self.qf1)
                self.qf2_avg_wgt_mag    = avg_weight_magnitude(self.qf2)

                b_obs, b_actions = self.rb.plasticity_data(2048) # a representative sample 
                _, _, self.actor_dead_pct   = count_dead_units(self.actor, in1=b_obs, threshold=self.h.dead_hurdle)
                _, _, self.qf1_dead_pct     = count_dead_units(self.qf1, in1=b_obs, in2=b_actions, threshold=self.h.dead_hurdle)
                _, _, self.qf2_dead_pct     = count_dead_units(self.qf2, in1=b_obs, in2=b_actions, threshold=self.h.dead_hurdle)

            chronos_total = time.time() - chronos_start
        
        return updated, chronos_total
    