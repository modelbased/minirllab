import torch

''' 
    Replay buffers for agents
    – On specific device, so no transfers or syncs when on GPU
    – pytorch only, no other dependencies
    – easily extensible by adding functions
   
    Additionally ReplayBufferSAC features
    – shared memory for use with multiprocessing
    – ability to return a trace (trace is created, no duplicate data is stored in the buffer)

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
        self.obs        = torch.zeros((num_steps, num_env) + (obs_dim,), device=device, requires_grad=False)
        self.actions    = torch.zeros((num_steps, num_env) + (action_dim,), device=device, requires_grad=False)
        self.logprobs   = torch.zeros((num_steps, num_env), device=device, requires_grad=False)
        self.rewards    = torch.zeros((num_steps, num_env), device=device)
        self.dones      = torch.zeros((num_steps, num_env), device=device, dtype=torch.int8, requires_grad=False)

        # Size in steps 
        self.max_size = num_steps
        self.ptr      = 0

    def store_choice(self, obs, action, logprob):
        self.obs[self.ptr]      = obs
        self.actions[self.ptr]  = action
        self.logprobs[self.ptr] = logprob
            
    def store_transition(self, reward, done):
        self.rewards[self.ptr]  = reward
        self.dones[self.ptr]    = done
        self.ptr               += 1

    def get_ppo_update(self):
        s          = int(self.ptr)
        
        # Flatten the batch
        b_obs      = self.obs[0:s].reshape((-1,) + (self.obs_dim,))
        b_actions  = self.actions[0:s].reshape((-1,) + (self.action_dim,))
        b_logprobs = self.logprobs[0:s].reshape(-1)
        return b_obs, b_actions, b_logprobs

    def get_gae(self):
        s           = int(self.ptr)
        
        # Don't flatten for GAE
        b_obs       = self.obs[0:s]
        b_rewards   = self.rewards[0:s]
        b_dones     = self.dones[0:s]
        return b_obs, b_rewards, b_dones

    def get_obs(self):
        s = int(self.ptr)
        return self.obs[0:s].reshape((-1,) + (self.obs_dim,))

    def reset(self):
        self.ptr = 0

    def flush(self):
        self.obs.zero_()
        self.actions.zero_()
        self.logprobs.zero_()
        self.rewards.zero_()
        self.dones.zero_()
        self.ptr   = 0


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

        # Use .share_memory_() to allow multiprocessing processes access to the same buffer data
        self.obs        = torch.zeros((max_size, num_env) + (obs_dim,), device=device, requires_grad=False)
        self.actions    = torch.zeros((max_size, num_env) + (action_dim,), device=device, requires_grad=False)
        self.rewards    = torch.zeros((max_size, num_env), device=device, requires_grad=False)
        self.dones      = torch.zeros((max_size, num_env), device=device, dtype=torch.int8, requires_grad=False)
        self.ep_num     = torch.zeros((max_size, num_env), device=device, dtype=torch.int32, requires_grad=False)

        # Counters and bookeeping. Tensors so that adding .share_memory() enables multiprocessing shared memory support
        self.max_size   = torch.tensor(min(int(1e6),max_size), dtype=torch.int32, device=device, requires_grad=False)        
        self.ptr        = torch.tensor(0, dtype=torch.int32, device=device, requires_grad=False)
        self.size       = torch.tensor(0, dtype=torch.int32, device=device, requires_grad=False)
        self.ep_count   = torch.zeros((num_env), dtype=torch.int32, device=device, requires_grad=False)


    def store_choice(self, obs, action):
        self.obs[self.ptr]     = obs        # o0 –> B
        self.actions[self.ptr] = action     # a0 –> B
        return


    def store_transition(self, reward, done):
        self.rewards[self.ptr]  = reward    # r0 -> B
        self.dones[self.ptr]    = done      # d0 -> B
        self.ep_count          += done      # episode count increments when an episode finishes

        # In-place operations maintains .shared_memory() if that's in use
        self.ptr.add_(1)                    # t0 -> t1
        self.ptr %= self.max_size           
        self.size.add_(1)
        self.size.clamp_(min=torch.zeros_like(self.size, device=self.device), max=self.max_size)
        self.ep_num[self.ptr] = self.ep_count # store episode number
        return
        

    def get_trace(self, idx, length):
        # Create a trace: start at idx, of length = length
        trace         = torch.arange(idx, idx-length, step=-1, device=self.device) # start at idx and count backward
        window        = self.ep_num[trace, :]     # section of buffer we will get and mask
        idx_ep        = self.ep_num[idx]          # episode number at idx requested
        mask          = (window == idx_ep)        # mask out episodes != ep_num at idx
        obs           = self.obs[trace, :, :]     # get the trace we want
        obs[~mask, :] = 0.0                       # apply mask and zero out any data from different episodes

        # Now a trace of actions_prev, the actions that resulted in these obs
        trace                 -= 1                         # actions leading to obs
        window                 = self.ep_num[trace, :]     # section of buffer we will get and mask
        idx_ep                 = self.ep_num[idx]          # episode number at idx requested
        mask                   = (window == idx_ep)        # mask out episodes != ep_num at idx
        actions_prev           = self.actions[trace, :, :] # get the trace we want
        actions_prev[~mask, :] = 0.0                       # apply mask and zero out any data from different episodes

        # return (batch, channels, length) used by convnets.
        return obs.permute(1, 2, 0), actions_prev.permute(1, 2, 0) # newest data in trace at (:, :, 0)


    def sample_trace(self, batch_size, length):
        ''' 
        Returns a batch for obs and actions of trace lenght with zeros where data is not from the same episode 
        b_actions are the actions that caused the b_obs, for critic training
        b_obs_next are the next obs also for actor critic training
        b_actions_next are for critic training and provides trace for new next action from actor(b_obs_next)
        b_rewards and b_dones are len=1, not provided as a trace 
        '''
        assert batch_size % self.num_env == 0, 'batch_size must be divisible by num_env'

        #TODO: Can almost certainly remove some ops
        def trace_and_mask(samples):

            # Create traces starting at each index
            inds        = samples.unsqueeze(1).repeat(1,length)                 # make a 2d array of the indices
            count       = torch.arange(0, -length, step=-1, device=self.device) # prepare a trace, same for all inds
            inds       += count.unsqueeze(0)                                    # inds now 2d array with a number of traces
            inds_trace  = inds.view(-1)                                         # reshape back into a 1d array of traces

            # Prepare window, which is series of traces from the buffer
            window  = self.ep_num[inds_trace]                   # get episode number at that buffer position 
            window  = window.transpose(dim0=1,dim1=0).flatten() # we want a 1d vector with envs ordered sequentially, not interleaved

            inds_ep = self.ep_num[samples]                      # get episode number at that buffer position. samples are just the start of the trace, not the whole trace
            inds_ep = inds_ep.transpose(dim0=1, dim1=0)         # correcting so each env's data will be sequential and not interleaved
            inds_ep = inds_ep.reshape(batch_size, -1)           # correcting so each env's data will be sequential and not interleaved
            inds_ep = inds_ep.repeat(1,length).flatten()        # now copy the episode number across the whole trace , flatten into a 1D vector
            
            # Create the mask to remove data from different episodes in the trace
            mask    = (window - inds_ep) == 0                   # data from correct episodes will match                                                                             == 0 #
            mask    = mask.view(batch_size,length)              # we want shape (batch_size, length)
            
            return mask, inds_trace

        # Sample random indices
        end     = self.size - 1 # allow for obs_next
        start   = 1             # allow for actions_prev
        samples = torch.randint(start, end, (batch_size // self.num_env,), device=self.device) #BUG: causes cuda<>cpu sync ?

        # Make a mask and trace
        mask     , inds_trace      = trace_and_mask(samples)    # ordinary samples
        mask_next, inds_trace_next = trace_and_mask(samples+1)  # obs_next samples
        mask_prev, inds_trace_prev = trace_and_mask(samples-1)  # action_prev samples

        # Get the obs and action data, re-arrange and mask
        b_obs        = self.obs[inds_trace]
        b_obs.transpose_(dim0=1, dim1=0)
        b_obs        = b_obs.reshape((batch_size, length) + (self.obs_dim,))
        b_obs[~mask] = 0.0
        b_obs        = b_obs.permute(0, 2, 1) # shape (batch, channels, length)

        b_actions        = self.actions[inds_trace]
        b_actions.transpose_(dim0=1, dim1=0)
        b_actions        = b_actions.reshape((-1,length) + (self.action_dim,))
        b_actions[~mask] = 0.0
        b_actions        = b_actions.permute(0, 2, 1)

        # Now do obs_next
        b_obs_next             = self.obs[inds_trace_next]
        b_obs_next.transpose_(dim0=1, dim1=0)
        b_obs_next             = b_obs_next.reshape((-1,length) + (self.obs_dim,))
        b_obs_next[~mask_next] = 0.0
        b_obs_next             = b_obs_next.permute(0, 2, 1)

        # Now do action_next
        b_actions_next             = self.actions[inds_trace_next]
        b_actions_next.transpose_(dim0=1, dim1=0)
        b_actions_next             = b_actions_next.reshape((-1,length) + (self.action_dim,))
        b_actions_next[~mask_next] = 0.0
        b_actions_next             = b_actions_next.permute(0, 2, 1)

        # Now do action_prev
        b_actions_prev             = self.actions[inds_trace_prev]
        b_actions_prev.transpose_(dim0=1, dim1=0)
        b_actions_prev             = b_actions_prev.reshape((-1,length) + (self.action_dim,))
        b_actions_prev[~mask_prev] = 0.0
        b_actions_prev             = b_actions_prev.permute(0, 2, 1)

        # No trace for these two but re-arranging needed for correct sequencing with obs and action
        b_rewards   = self.rewards[samples].transpose(dim0=1,dim1=0).reshape(-1)
        b_dones     = self.dones[samples].transpose(dim0=1,dim1=0).reshape(-1)

        # return osb and actions with shape (batch, channels, length) or (batch) for rewards and dones
        return b_obs, b_actions, b_obs_next, b_actions_next, b_actions_prev, b_rewards, b_dones # newest data in trace at (:, :, 0)     


    def sample(self, batch_size, ere_bias=False):
        '''' For training the critic. Actions returned are the ones taken to cause the Observation '''
          
        # Faster,  but no uniqueness guarantee like inds = torch.randperm(s)
        end     = self.size - 1 # allow for obs_next

        # BUG: conditionals probably causing cuda<>cpu syncs, optimise at some point
        # Emphasise Recent Experience bias https://arxiv.org/abs/1906.04009
        if ere_bias:
            start   = torch.randint(0, end // 2, (1,), device=self.device)
        else:
            start   = 0 # no bias, uniform sampling of buffer
        
        samples = torch.randint(start, end, (batch_size // self.num_env,), device=self.device) # on correct device to avoid cuda-cpu synchronisation

        # When biased account for circular buffer
        if ere_bias:
            if self.ptr < self.size: samples = (samples + self.ptr) % self.size #  will force a cuda <> cpu sync

        # Flatten the batch (global_step, env_num, channels) -> (b_step, channels)
        b_obs       = self.obs[samples].reshape((-1,) + (self.obs_dim,))
        b_actions   = self.actions[samples].reshape((-1,) + (self.action_dim,))
        b_obs_next  = self.obs[samples + 1].reshape((-1,) + (self.obs_dim,)) # Sometimes obs_next will be step zero of next episode, but ok for SAC
        b_rewards   = self.rewards[samples].reshape(-1) 
        b_dones     = self.dones[samples].reshape(-1)
        return b_obs, b_actions, b_obs_next, b_rewards, b_dones

    def plasticity_data(self, batch_size):
        inds        = torch.randint(0, self.size, (batch_size // self.num_env,), device=self.device)
        b_obs       = self.obs[inds].reshape((-1,) + (self.obs_dim,))
        b_actions   = self.actions[inds].reshape((-1,) + (self.action_dim,))
        return b_obs, b_actions

    def reset(self):
        self.ptr  = 0
        self.size = 0
        return

    def flush(self):
        self.obs.zero_()
        self.actions.zero_()
        self.rewards.zero_()
        self.dones.zero_()
        self.ep_num.zero_()
        self.ptr  = 0
        self.size = 0
        return