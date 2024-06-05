import time, math
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace
from .utils import avg_weight_magnitude, count_dead_units, dormant_ratio
from .buffers import ReplayBufferSAC
from torchinfo import summary

'''
    Based on https://github.com/vwxyzjn/cleanrl
    SAC agent https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py

    CrossQ extensions to SAC influenced by:
    - CrossQ: Batch Normalization in Deep Reinforcement Learning for Greater Sample Efficiency and Simplicity https://arxiv.org/abs/1902.05605
    - v3 of the arxiv paper (2023)

    Plasticity metrics and regularisation influnced by the following papers:
    - Maintaining Plasticity in Continual Learning via Regenerative Regularization https://arxiv.org/abs/2308.11958
    - Loss of Plasticity in Deep Continual Learning https://arxiv.org/abs/2306.13812
    - Sample-Efficient Reinforcement Learning by Breaking the Replay Ratio Barrier https://openreview.net/forum?id=OpC-9aBBVJe
    – Bigger, Better, Faster: Human-level Atari with human-level efficiency https://arxiv.org/abs/2305.19452

    Performance optimisations inspired by BRO and friends:
    - BRO: Bigger, Regularized, Optimistic: scaling for compute and sample-efficient continuous control https://arxiv.org/abs/2405.16158
    – SR-SA: Sample-Efficient Reinforcement Learning by Breaking the Replay Ratio Barrier https://openreview.net/forum?id=OpC-9aBBVJe 
    - Overestimation, Overfitting, and Plasticity https://arxiv.org/abs/2403.00514
'''

class CrossQBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.cqb = nn.Sequential(
            nn.Linear(in_dim, out_dim, bias=False), # batchnorm has bias, this one is redundant
            nn.BatchNorm1d(out_dim, momentum=0.01), # CrossQ momentum
            nn.ReLU(),
            nn.Linear(out_dim, out_dim, bias=False), # batchnorm has bias, this one is redundant
            nn.BatchNorm1d(out_dim, momentum=0.01), # CrossQ momentum
            nn.ReLU(),
        )

    def forward(self, x):
        return self.cqb(x)        


class SoftQNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()

        self.b1 = CrossQBlock(obs_dim + action_dim, hidden_dim)
        self.b2 = CrossQBlock(hidden_dim, hidden_dim)
        self.b3 = CrossQBlock(hidden_dim, hidden_dim)
        self.b4 = nn.Linear(hidden_dim, 1)


    def forward(self, o, a=None):
        
        if a is None:
            x = o
        else:
            x = torch.cat([o, a], 1)
        
        x1 = self.b1(x)
        x2 = self.b2(x1) + x1
        x3 = self.b3(x2) + x2
        x4 = self.b4(x3)

        return x4


class Actor(nn.Module):
    def __init__(self, env_spec, hidden_dim=256):
        super().__init__()
        
        obs_dim = env_spec['obs_dim']
        act_dim = env_spec['act_dim']
        
        self.mlp = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim, momentum=0.01), #CrossQ
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim, momentum=0.01), #CrossQ
            nn.ReLU(),
        )
        self.fc_mean   = nn.Linear(hidden_dim, act_dim)
        self.fc_logstd = nn.Linear(hidden_dim, act_dim)

        action_high       = nn.Parameter(torch.tensor(env_spec['act_max']), requires_grad=False)
        action_low        = nn.Parameter(torch.tensor(env_spec['act_min']), requires_grad=False)
        self.action_scale = nn.Parameter((action_high - action_low) * 0.5, requires_grad=False)
        self.action_bias  = nn.Parameter((action_high + action_low) * 0.5, requires_grad=False)
        self.log_std_max  = 2
        self.log_std_min  = -5

    def forward(self, x):
        x       = self.mlp(x)
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
                 buffer_size = int(1e6),
                 num_env     = 1,
                 device      = 'cpu',
                 seed        = 42,
                 rr          = 2,        # RR = 1 for CrossQ
                 q_lr        = 1e-3,     # CrossQ learning rates
                 actor_lr    = 1e-3,
                 alpha_lr    = 1e-3,
                 ):
        
        # Make global
        self.name           = "sac_crossq_bro"           # name for logging
        self.obs_dim        = env_spec['obs_dim']    # environment inputs for agent
        self.action_dim     = env_spec['act_dim']    # agent outputs to environment
        self.act_max        = env_spec['act_max']    # action range, scalar or vector
        self.act_min        = env_spec['act_min']    # action range, scalar or vector
        self.device         = device                 # gpu or cpu

        # All seeds default to 42
        torch.manual_seed(torch.tensor(seed))
        torch.backends.cudnn.deterministic = True
        # torch.cuda.set_sync_debug_mode(1)            # Set to 1 to receive warnings
        # torch.set_float32_matmul_precision("high")   # "high" is 11% faster, but can reduce learning performance in certain envs

        # Hyperparameters
        hyperparameters = {
        "gamma"             : 0.99,     # (def: 0.99) Discount factor
        "q_lr"              : q_lr,     # (def: 1e-3) Q learning rate 
        "a_lr"              : actor_lr, # (def: 1e-3) Policy learning rate
        "alpha_lr"          : alpha_lr, # (def: 1e-3) alpha auto entropoty tuning learning rate
        "learn_start"       : int(5e3), # (def: 5e3) Start updating policies after this many global steps
        "batch_size"        : 256,      # (def: 256) Batch size of sample from replay buffer
        "policy_freq"       : 3,        # (def: 3) CrossQ
        "dead_hurdle"       : 0.001,    # (def: 0.001) units with greater variation in output over one batch of data than this are not dead in plasticity terms
        "a_hidden_dim"      : 256,      # (def: 256) size of actor's hidden layer(s)
        "q_hidden_dim"      : 1024,     # (def: 2048) CrossQ with 512 wide Qf did just as well, but with a little more variance
        "replay_ratio"      : round(rr),
        "adam_betas"        : (0.5, 0.999), # CrossQ
        }
        self.h = SimpleNamespace(**hyperparameters)

        # Loggin & debugging
        self.qf1_a_values        = torch.tensor([0.0])
        self.qf2_a_values        = torch.tensor([0.0])
        self.qf1_loss            = 0
        self.qf2_loss            = 0
        self.qf_loss             = 0
        self.actor_loss          = 0
        self.alpha_loss          = 0
        self.actor_avg_wgt_mag   = 0 # average weight magnitude as per https://arxiv.org/abs/2306.13812
        self.qf1_avg_wgt_mag     = 0     
        self.qf2_avg_wgt_mag     = 0     
        self.actor_dead_pct      = 0 # dead units as per https://arxiv.org/abs/2306.13812
        self.qf1_dead_pct        = 0     
        self.qf2_dead_pct        = 0     
        self.qf1_dormant_ratio   = 0 # DrM: Dormant Ratio Minimisation https://arxiv.org/abs/2310.19668
        self.qf2_dormant_ratio   = 0
        self.actor_dormant_ratio = 0


        # Instantiate actor and Q networks, optimisers
        # CrossQ uses Adam but experience with AdamW is better
        self.qf1        = SoftQNetwork(self.obs_dim, self.action_dim, self.h.q_hidden_dim).to(device)
        self.qf2        = SoftQNetwork(self.obs_dim, self.action_dim, self.h.q_hidden_dim).to(device)
        self.q_optim    = torch.optim.AdamW(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=self.h.q_lr, betas=self.h.adam_betas)
        
        # summary(self.qf1, input_size=(1, self.obs_dim+self.action_dim))       
        # exit()

        self.actor       = Actor(env_spec, self.h.a_hidden_dim).to(device)
        self.actor_optim = torch.optim.AdamW(list(self.actor.parameters()), lr=self.h.a_lr)
        
        # Use automatic entropy tuning
        self.target_entropy = -(torch.prod(torch.Tensor((self.action_dim,))).to(device)).item()
        self.log_alpha      = torch.tensor((math.log(0.1)), requires_grad=True, device=device)
        self.alpha          = self.log_alpha.exp().item()
        self.alpha_optim    = torch.optim.AdamW([self.log_alpha], lr=self.h.alpha_lr)

        # Storage setup
        self.rb          = ReplayBufferSAC(self.obs_dim, self.action_dim, buffer_size, num_env, device=self.device)
        self.global_step = 0

        # CUDA timers for the update process
        self.chronos_start = torch.cuda.Event(enable_timing=True)
        self.chronos_end   = torch.cuda.Event(enable_timing=True)


    def choose_action(self, obs):
        # Random uniform actions before learn_start can speed up training over using the agent's inital randomness.
        if self.global_step < self.h.learn_start:
            # actions are rand_uniform of shape (obs_batch_size, action_dim)
            action = (torch.rand((obs.size(0), self.action_dim), device=self.device) - 0.5) * 2.0  # rand_uniform -1..+1
            action = action * self.actor.action_scale + self.actor.action_bias # apply scale and bias
        else:
            with torch.no_grad():
                self.actor.eval() # prevent changes to batchnorm layers
                action, _, _ = self.actor.get_action(obs)
                self.actor.train()
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
            self.chronos_start.record()

            for replay in range(0, self.h.replay_ratio):
    
                b_obs, b_actions, b_obs_next, b_rewards, b_dones = self.rb.sample(self.h.batch_size)

                with torch.no_grad():
                    self.actor.eval()
                    next_state_actions, next_state_log_pi, _ = self.actor.get_action(b_obs_next)
                    self.actor.train()

                bb_obs  = torch.cat((b_obs, b_obs_next), dim=0)
                bb_acts = torch.cat((b_actions, next_state_actions), dim=0)

                bb_q1 = self.qf1(bb_obs, bb_acts)
                bb_q2 = self.qf2(bb_obs, bb_acts)

                b_q1, b_q1_next = torch.chunk(bb_q1, chunks=2, dim=0)
                b_q2, b_q2_next = torch.chunk(bb_q2, chunks=2, dim=0)
                self.qf1_a_values = b_q1 # mean of this is used in logging
                self.qf2_a_values = b_q2 # mean of this is used in logging

                # BRO uses mean() instead of min() for "exploration optimism"
                # min_q_next   = torch.min(b_q1_next, b_q2_next) - self.alpha * next_state_log_pi
                min_q_next = ((b_q1_next + b_q2_next) * 0.5) - self.alpha * next_state_log_pi
                
                next_q_value = b_rewards.flatten() + (1 - b_dones.flatten()) * self.h.gamma * (min_q_next).view(-1)
                torch.detach_(next_q_value) # no gradients through here

                self.qf1_loss     = F.mse_loss(b_q1.flatten(), next_q_value)
                self.qf2_loss     = F.mse_loss(b_q2.flatten(), next_q_value)
                self.qf_loss      = self.qf1_loss + self.qf2_loss

                self.q_optim.zero_grad()
                self.qf_loss.backward()
                self.q_optim.step()

                # BRO and SR-SAC seem to include actor in RR (unclear: alpha?)
                
                # Update actor network and alpha parameter
                if self.global_step % self.h.policy_freq == 0:  # TD 3 Delayed update support
                    pi, log_pi, _   = self.actor.get_action(b_obs)
                    
                    self.qf1.eval()
                    self.qf2.eval()
                    qf1_pi          = self.qf1(b_obs, pi)
                    qf2_pi          = self.qf2(b_obs, pi)
                    self.qf1.train()
                    self.qf2.train()

                    min_qf_pi       = torch.min(qf1_pi, qf2_pi).view(-1)
                    self.actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

                    self.actor_optim.zero_grad()
                    self.actor_loss.backward()
                    self.actor_optim.step()

                    # Autotune alpha 
                    with torch.no_grad():
                        self.actor.eval()
                        _, log_pi, _ = self.actor.get_action(b_obs)
                        self.actor.train()
                    self.alpha_loss = (-self.log_alpha.exp() * (log_pi + self.target_entropy)).mean()

                    self.alpha_optim.zero_grad()
                    self.alpha_loss.backward()
                    self.alpha_optim.step()
                    self.alpha = self.log_alpha.exp().detach().clone()
            
            # Plasticity metrics occasionally
            if self.global_step % 2048 == 0 or self.global_step == self.h.learn_start:
                self.actor_avg_wgt_mag  = avg_weight_magnitude(self.actor)
                self.qf1_avg_wgt_mag    = avg_weight_magnitude(self.qf1)
                self.qf2_avg_wgt_mag    = avg_weight_magnitude(self.qf2)

                b_obs, b_actions          = self.rb.plasticity_data(2048) # a representative sample 
                _, _, self.qf1_dead_pct   = count_dead_units(self.qf1, in1=b_obs, in2=b_actions, threshold=self.h.dead_hurdle)
                _, _, self.qf2_dead_pct   = count_dead_units(self.qf2, in1=b_obs, in2=b_actions, threshold=self.h.dead_hurdle)
                _, _, self.actor_dead_pct = count_dead_units(self.actor, in1=b_obs, threshold=self.h.dead_hurdle)

                self.qf1_dormant_ratio   = dormant_ratio(self.qf1, in1=b_obs, in2=b_actions)
                self.qf2_dormant_ratio   = dormant_ratio(self.qf2, in1=b_obs, in2=b_actions)
                self.actor_dormant_ratio = dormant_ratio(self.actor, in1=b_obs)

            # Record end time, wait for all cuda threads to sync and calc time in seconds
            self.chronos_end.record()
            torch.cuda.synchronize()
            chronos_total = (self.chronos_start.elapsed_time(self.chronos_end) * 0.001)   
        
        return updated, chronos_total
    

    def save(self, path='./checkpoints/'):
        
        path   = path + self.name + '.pt'
        models = {}

        models['Q1']    = self.qf1.state_dict()
        models['Q2']    = self.qf2.state_dict()
        models['Actor'] = self.actor.state_dict()

        torch.save(models, path)        

    def load(self, path='./checkpoints/'):
        path = path + self.name + '.pt'
        models_file = torch.load(path)

        self.qf1.load_state_dict(models_file['Q1'])
        self.qf2.load_state_dict(models_file['Q2'])
        self.actor.load_state_dict(models_file['Actor'])