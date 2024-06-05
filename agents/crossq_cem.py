import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace
from .utils import avg_weight_magnitude, count_dead_units, dormant_ratio
from .buffers import ReplayBufferSAC

'''
    Based on https://github.com/vwxyzjn/cleanrl
    - SAC agent https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py
    - CrossQ extensiopns to SAC, replacing actor with cross entropy method action selection

    CrossQ extensions to SAC influenced by:
    - CrossQ: Batch Normalization in Deep Reinforcement Learning for Greater Sample Efficiency and Simplicity https://arxiv.org/abs/1902.05605
    - v3 of the arxiv paper (2023)

    Cross Entropy Method Action Optimisation influnced by 
    - QT-Opt: Scalable Deep Reinforcement Learning for Vision-Based Robotic Manipulation https://arxiv.org/abs/1806.10293
    – TD-MPC2: Scalable, Robust World Models for Continuous Control https://arxiv.org/abs/2310.16828
    - iCEM: https://arxiv.org/abs/2008.06389

    Plasticity metrics and regularisation influenced by the following papers:
    - Maintaining Plasticity in Continual Learning via Regenerative Regularization https://arxiv.org/abs/2308.11958
    - Loss of Plasticity in Deep Continual Learning https://arxiv.org/abs/2306.13812
    - Sample-Efficient Reinforcement Learning by Breaking the Replay Ratio Barrier https://openreview.net/forum?id=OpC-9aBBVJe
    – Bigger, Better, Faster: Human-level Atari with human-level efficiency https://arxiv.org/abs/2305.19452
'''

class SoftQNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim, bias=False), # batchnorm has bias, this one is redundant
            nn.BatchNorm1d(hidden_dim, momentum=0.01), #CrossQ
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim, momentum=0.01), #CrossQ
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # init_model(self.mlp, init_method='xavier_uniform_') # CrossQ no mention of init

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = self.mlp(x)
        return x


class Agent:
    def __init__(self, 
                 env_spec, 
                 buffer_size = int(1e6),
                 num_env     = 1,
                 device      = 'cpu',
                 seed        = 42,
                 rr          = 1,        # RR = 1 for CrossQ
                 q_lr        = 1e-3,     # CrossQ learning rates
                 ):
        
        # Make global
        self.name           = "crossq_cem"           # name for logging
        self.obs_dim        = env_spec['obs_dim']    # environment inputs for agent
        self.action_dim     = env_spec['act_dim']    # agent outputs to environment
        self.act_max        = env_spec['act_max']    # action range, scalar or vector
        self.act_min        = env_spec['act_min']    # action range, scalar or vector
        self.device         = device                 # gpu or cpu

        self.action_scale = torch.tensor(((self.act_max - self.act_min) * 0.5), device=device)
        self.action_bias  = torch.tensor(((self.act_max + self.act_min) * 0.5), device=device)
        self.act_min = torch.tensor(self.act_min, device=device)
        self.act_max = torch.tensor(self.act_max, device=device)

        # All seeds default to 42
        torch.manual_seed(torch.tensor(seed))
        torch.backends.cudnn.deterministic = True
        # torch.cuda.set_sync_debug_mode(1)            # Set to 1 to receive warnings
        # torch.set_float32_matmul_precision("high")   # "high" is 11% faster, but can reduce learning performance in certain envs

        # Hyperparameters
        hyperparameters = {
        "gamma"             : 0.99,     # (def: 0.99) Discount factor
        "q_lr"              : q_lr,     # (def: 1e-3) Q learning rate 
        "learn_start"       : int(5e3), # (def: 5e3) Start updating policies after this many global steps
        "batch_size"        : 256,      # (def: 256) Batch size of sample from replay buffer
        "dead_hurdle"       : 0.001,    # (def: 0.001) units with greater variation in output over one batch of data than this are not dead in plasticity terms
        "q_hidden_dim"      : 512,     # (def: 2048) CrossQ with 512 wide Qf did just as well, but with a little more variance
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
        
        # Storage setup
        self.rb          = ReplayBufferSAC(self.obs_dim, self.action_dim, buffer_size, num_env, device=self.device)
        self.global_step = 0
        self.action_prev = None

        # CUDA timers for the update process
        self.chronos_start = torch.cuda.Event(enable_timing=True)
        self.chronos_end   = torch.cuda.Event(enable_timing=True)
    

    def ce_action_solver(self, obs, explore=False):
        with torch.no_grad():

            # Hyperparameters for CEM
            samples        = 32 * self.action_dim # TODO: Starting wider and reducing sample numbers likely more efficient and effective
            topk           = (samples // 8)
            iterations     = 6
            converged_stdd = 0.025
            explore_std    = 0.5                  # TODO: Should reduce on schedule or as learning improves
            stdd_min       = torch.tensor(0.01, device=self.device)

            batch_size = obs.shape[0] # input shape assumes (batch, channels)
            self.qf1.eval()           # don't mess with crossq batchnorm layers during inference
            self.qf2.eval()

            
            # large stdd on first action otherwise smaller stdd to converge faster (assuming solution will be nearby)
            if self.action_prev is None:
                iter1_stdd = 1.0
                top_mean = torch.tile(torch.zeros(self.action_dim, device=self.device) + self.action_bias, (batch_size, 1))
            else:
                iter1_stdd = 0.25
                top_mean = torch.tile(self.action_prev, (batch_size, 1))
            
            # Important that we don't start optimising outside action range
            top_mean.clamp_(self.act_min, self.act_max)
            
            # action scaling is applied to stdd so it covers the action range correctly
            top_stdd = torch.ones((batch_size, self.action_dim), device=self.device) * iter1_stdd * self.action_scale # (batch, action_dim)

            for i in range(iterations):
                    
                    # Expand obs batch by tile of size samples
                    obs_tiled = torch.tile(obs,(samples,1)) # (samples*batch_dim, channels)
                    
                    # Sample actions from normal distribution in as many batches as there are in obs
                    #TODO: actions range should be clamped, make sure applying clamp does not break things
                    actions_tiled = torch.normal(mean=torch.tile(top_mean, (samples,1)), std=torch.tile(top_stdd, (samples,1))) # (samples*batch_dim, action_dim)
                    actions_tiled.clamp_(self.act_min, self.act_max)

                    # Value each sampled action and get top (elite) indices  
                    v1   = self.qf1(obs_tiled, actions_tiled)         # (samples*batch_size)
                    v2   = self.qf2(obs_tiled, actions_tiled)         # (samples*batch_size)
                    v    = torch.min(v1, v2).view(samples,batch_size) # (samples, batch_size)
                    _, k = torch.topk(v, k=topk, dim=0)               # (topk, batch_size)

                    # Get mean and standard deviation of the top (elite) actions
                    actions_view = actions_tiled.view(samples,batch_size, -1)      # (samples, batch_size, action_dim)
                    top_actions  = actions_view[k,range(actions_view.shape[1]),:]  # (topk, batch_size, action_dim)

                    top_stdd, top_mean = torch.std_mean(top_actions, dim=0)         # (batch_size, action_dim)
                    top_stdd.clamp_min_(stdd_min)                                   # clamp stdd to a min (also normal(std≤0) crashes
                    
                    # Break when cpnmverged
                    if (torch.mean(top_stdd) < converged_stdd):
                        break

            # Store it for next time if acting. Will be reset on new episode
            if explore:
                self.action_prev = top_mean # important to store before applying noise
                top_mean += torch.normal(mean=top_mean, std=explore_std)
            
            self.qf1.train()
            self.qf2.train()

        return top_mean


    def choose_action(self, obs):
        # Random uniform actions before learn_start can speed up training over using the agent's inital randomness.
        if self.global_step < self.h.learn_start:
            # actions are rand_uniform of shape (obs_batch_size, action_dim)
            action = (torch.rand((obs.size(0), self.action_dim), device=self.device) - 0.5) * 2.0  # rand_uniform -1..+1
            action = action * self.action_scale + self.action_bias # apply scale and bias
        else:
            action = self.ce_action_solver(obs, explore=True) # output is scaled and biased
        
        self.rb.store_choice(obs, action)
        
        return action

    def store_transition(self, reward, done):
        self.rb.store_transition(reward, done)
        self.global_step += 1

        if done:
            self.action_prev = None


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
                    next_state_actions = self.ce_action_solver(b_obs_next)

                bb_obs  = torch.cat((b_obs, b_obs_next), dim=0)
                bb_acts = torch.cat((b_actions, next_state_actions), dim=0)

                bb_q1 = self.qf1(bb_obs, bb_acts)
                bb_q2 = self.qf2(bb_obs, bb_acts)

                b_q1, b_q1_next = torch.chunk(bb_q1, chunks=2, dim=0)
                b_q2, b_q2_next = torch.chunk(bb_q2, chunks=2, dim=0)
                self.qf1_a_values = b_q1 # mean of this is used in logging
                self.qf2_a_values = b_q2 # mean of this is used in logging

                min_q_next   = torch.min(b_q1_next, b_q2_next)
                next_q_value = b_rewards.flatten() + (1 - b_dones.flatten()) * self.h.gamma * (min_q_next).view(-1)
                torch.detach_(next_q_value) # no gradients through here

                self.qf1_loss     = F.mse_loss(b_q1.flatten(), next_q_value)
                self.qf2_loss     = F.mse_loss(b_q2.flatten(), next_q_value)
                self.qf_loss      = self.qf1_loss + self.qf2_loss

                self.q_optim.zero_grad()
                self.qf_loss.backward()
                self.q_optim.step()

            # Plasticity metrics occasionally
            if self.global_step % 2048 == 0 or self.global_step == self.h.learn_start:
                self.qf1_avg_wgt_mag    = avg_weight_magnitude(self.qf1)
                self.qf2_avg_wgt_mag    = avg_weight_magnitude(self.qf2)

                b_obs, b_actions          = self.rb.plasticity_data(2048) # a representative sample 
                _, _, self.qf1_dead_pct   = count_dead_units(self.qf1, in1=b_obs, in2=b_actions, threshold=self.h.dead_hurdle)
                _, _, self.qf2_dead_pct   = count_dead_units(self.qf2, in1=b_obs, in2=b_actions, threshold=self.h.dead_hurdle)

                self.qf1_dormant_ratio   = dormant_ratio(self.qf1, in1=b_obs, in2=b_actions)
                self.qf2_dormant_ratio   = dormant_ratio(self.qf2, in1=b_obs, in2=b_actions)

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

        torch.save(models, path)        

    def load(self, path='./checkpoints/'):
        path = path + self.name + '.pt'
        models_file = torch.load(path)

        self.qf1.load_state_dict(models_file['Q1'])
        self.qf2.load_state_dict(models_file['Q2'])