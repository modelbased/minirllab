import torch

''' 
    Replay buffers for agents
'''

class ReplayBufferPPO():
    """
        Call reset or flush at the end of a update
        Must call .store_transition() to advance pointer
    """
    def __init__(self, obs_dim, action_dim, num_steps, num_env, device):
        
        self.obs_dim    = obs_dim
        self.action_dim = action_dim

        # Storage setup
        self.obs        = torch.zeros((num_steps, num_env) + (obs_dim,), device=device)
        self.actions    = torch.zeros((num_steps, num_env) + (action_dim,), device=device)
        self.logprobs   = torch.zeros((num_steps, num_env), device=device)
        self.rewards    = torch.zeros((num_steps, num_env), device=device)
        self.dones      = torch.zeros((num_steps, num_env), device=device)

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

    def get_obs(self):
        s = int(self.size)
        return self.obs[0:s].reshape((-1,) + (self.obs_dim,))

    def reset(self):
        self.size = 0

    def flush(self):
        self.obs.zero_()
        self.actions.zero_()
        self.logprobs.zero_()
        self.rewards.zero_()
        self.dones.zero_()
        self.size   = 0


class ReplayBufferSAC():
    """
        Circular buffer with reset() and flush()
        Must call .store_transition() to advance pointer
    """
    def __init__(self, obs_dim, action_dim, max_size, num_env, device):
        
        self.obs_dim    = obs_dim
        self.action_dim = action_dim
        self.num_env    = num_env
        self.device     = device

        # Storage setup
        self.obs        = torch.zeros((max_size, num_env) + (obs_dim,), device=device)
        self.actions    = torch.zeros((max_size, num_env) + (action_dim,), device=device)
        self.rewards    = torch.zeros((max_size, num_env), device=device)
        self.dones      = torch.zeros((max_size, num_env), device=device)

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
        s    = int(self.size) - 1                 # allow for obs_next

        # Guarantees uniqueness, but slow and gets slower with time
        # inds = torch.randperm(s)                  # shuffled index over entire buffer
        # inds = inds[:batch_size // self.num_env]  # limit sample to batch size when vectorised
        
        # Faster, but no uniqueness guarantee
        # Set correct device to avoid cuda-cpu synchronisation
        inds = torch.randint(0, s, (batch_size // self.num_env,), device=self.device)

        # Flatten the batch (global_step, env_num, channels) -> (b_step, channels)
        b_obs       = self.obs[inds].reshape((-1,) + (self.obs_dim,))
        b_actions   = self.actions[inds].reshape((-1,) + (self.action_dim,))
        b_obs_next  = self.obs[inds + 1].reshape((-1,) + (self.obs_dim,)) 
        b_rewards   = self.rewards[inds].reshape(-1) 
        b_dones     = self.dones[inds].reshape(-1)
        return b_obs, b_actions, b_obs_next, b_rewards, b_dones

    def plasticity_data(self, batch_size):
        inds        = torch.randint(0, self.size, (batch_size // self.num_env,), device=self.device)
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