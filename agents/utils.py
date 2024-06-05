import torch
import torch.nn as nn
import numpy as np

''' Utility functions for agents '''

# symlog() and symexp() are alternatives to avg/mean running input normalisation 
# Dreamer V3 https://arxiv.org/abs/2301.04104
@torch.jit.script
def symlog(x):
    return torch.sign(x) * torch.log(torch.abs(x)+1) 

@torch.jit.script
def symexp(x):
    return torch.sign(x) * torch.exp(torch.abs(x)-1)


def init_layer(layer, std=torch.sqrt(torch.tensor(2)), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    if bias_const is None:
        layer.bias = None
    else:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def init_model(model, init_method):
    """
    Initialize the weights and biases of a PyTorch model's linear layers using the specified initialization method.

    Args:
        model (nn.Module): The PyTorch model to initialize.
        init_method (str): The initialization method to apply to the model's linear layers.
                           Should be one of 'xavier_uniform_', 'kaiming_uniform_', or 'orthogonal_'.

    Returns:
        None
    """
    # iterate over the model's linear layers and initialize their weights and biases
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            # initialize the weights and biases using the specified method
            if init_method == 'xavier_uniform_':
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None: nn.init.zeros_(layer.bias)
            elif init_method == 'kaiming_uniform_':
                nn.init.kaiming_uniform_(layer.weight, a=0, mode='fan_in', nonlinearity='linear')
                if layer.bias is not None: nn.init.zeros_(layer.bias)
            elif init_method == 'orthogonal_':
                nn.init.orthogonal_(layer.weight)
                if layer.bias is not None: nn.init.zeros_(layer.bias)


# Credit GPT4
def avg_weight_magnitude(model):
    """ Average weight magnitude of a models parameters """
    total_weight_sum = 0.0
    total_weight_count = 0
    
    for param in model.parameters():
        total_weight_sum   += torch.sum(torch.abs(param))
        total_weight_count += param.numel()

    return total_weight_sum / total_weight_count


# GPT4 helpful as ever
def count_dead_units(model, in1, in2=None, threshold=0.01):
    """ Counts dead units based on observed range of activation for various types """
    
    activations = []

    def hook_fn(module, input, output):
        activations.append(output)

    # Register hooks for all layers
    hooks = []
    for layer in model.modules():
        if isinstance(layer, (nn.ReLU, nn.Sigmoid, nn.Tanh, nn.LeakyReLU, nn.ELU, nn.SiLU)):
            hooks.append(layer.register_forward_hook(hook_fn))

    # Initialize max and min values for each layer's activations
    max_values = []
    min_values = []

    model.eval() # dont mess with the model if it has e.g. batchnorm, dropout etc
    if in2 is None:
        with torch.no_grad():
            model(in1)
    else:
        with torch.no_grad():
            model(in1, in2)
    model.train() # put it back how it was

    # Directly compute the max and min values for each activation unit
    max_values = [act.max(dim=0)[0] for act in activations]
    min_values = [act.min(dim=0)[0] for act in activations]

    # Count dead units based on observed activation range
    dead_units_count = 0
    total_units_count = 0

    for max_val, min_val in zip(max_values, min_values):
        dead_range = (max_val - min_val) < threshold
        dead_units_count += dead_range.sum()
        total_units_count += max_val.numel()  # Since max_val and min_val have the same size, we can use either for counting

    # Clean up hooks
    for hook in hooks:
        hook.remove()

    dead_percentage = (dead_units_count / total_units_count) * 100.0
    return dead_units_count, total_units_count, dead_percentage


# Credit GPT4: Class to store initial parameters and compute L2 Init loss
# https://arxiv.org/abs/2308.11958
class L2NormRegularizer():
    def __init__(self, model, lambda_reg, device):
        self.device         = device
        self.lambda_reg     = lambda_reg
        self.initial_params = torch.empty(0, device=device)

        # A tensor of model parameters
        for par in model.parameters(): 
            self.initial_params = torch.cat((self.initial_params, par.view(-1).detach().clone()), dim=0)
    
    # Add this loss to the model's usual loss
    def __call__(self, model):
        current_params = torch.empty(0, device=self.device)

        for par in model.parameters(): 
            current_params = torch.cat((current_params, par.view(-1)), dim=0)

        l2_loss = torch.linalg.vector_norm((current_params - self.initial_params), ord=2)
        return self.lambda_reg * l2_loss
    

# DrM: Dormant Ratio Minimisation https://arxiv.org/abs/2310.19668
# https://github.com/XuGW-Kevin/DrM/blob/main/utils.py
# "a sharp decline in the dormant ratio of an agent’s policy network serves as an intrinsic indicator of the agent executing meaningful actions for exploration"
# see dormant neurons here also: https://arxiv.org/abs/2302.12902
# to get dormant neurons from start of training like paper needed 1. nn.RuLE(inplace=True) & 2. no BatchNorm()
class LinearOutputHook:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)


def dormant_ratio(model, in1, in2=None, percentage=0.10):
    hooks           = []
    hook_handlers   = []
    total_neurons   = 0
    dormant_neurons = 0

    for _, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hook = LinearOutputHook()
            hooks.append(hook)
            hook_handlers.append(module.register_forward_hook(hook))

    model.eval() # dont mess with the model if it has e.g. batchnorm, dropout etc
    if in2 is None:
        with torch.no_grad():
            model(in1)
    else:
        with torch.no_grad():
            model(in1, in2)
    model.train() # put it back how it was

    for module, hook in zip((module for module in model.modules() if isinstance(module, nn.Linear)), hooks):
        with torch.no_grad():
            for output_data in hook.outputs:
                mean_output        = output_data.abs().mean(0)
                avg_neuron_output  = mean_output.mean()
                dormant_indices    = (mean_output < avg_neuron_output * percentage).nonzero(as_tuple=True)[0]
                total_neurons     += module.weight.shape[0]
                dormant_neurons   += len(dormant_indices)

    for hook in hooks:
        hook.outputs.clear()

    for hook_handler in hook_handlers:
        hook_handler.remove()

    return dormant_neurons / total_neurons

# https://sites.google.com/view/rl-vcse
# https://github.com/kingdy2002/VCSE/blob/main/VCSE_SAC/vcse.py
'''
Usage: 
1. Add another pair of critics to SAC
2. One pair trains without receiving vcse intrinsic reward. Use this pair to feed value to vcse
3. Other pair trains with vcse intrinsic reward (add to usual reward). use this pair to train actor.
'''
class VCSE(object):
    """particle-based entropy based on knn normalized by running mean """
    def __init__(self, knn_k=12, beta=0.1, device='cpu'):
        self.knn_k  = knn_k
        self.device = device
        self.beta   = beta # Tuning factor from paper

    def __call__(self, state, value):
        # value => [b1 , 1]
        # state => [b1 , c]
        # z => [b1, c+1]
        # [b1] => [b1,b1]
        ds = state.size(1)
        source = target = state
        b1, b2 = source.size(0), target.size(0)
        # (b1, 1, c+1) - (1, b2, c+1) -> (b1, 1, c+1) - (1, b2, c+1) -> (b1, b2, c+1) -> (b1, b2)
        sim_matrix_s = torch.norm(source[:, None, :].view(b1, 1, -1) -  target[None, :, :].view(1, b2, -1), dim=-1, p=2)

        source = target = value
        # (b1, 1, 1) - (1, b2, 1) -> (b1, 1, 1) - (1, b2, 1) -> (b1, b2, 1) -> (b1, b2)
        sim_matrix_v = torch.norm(source[:, None, :].view(b1, 1, -1) - target[None, :, :].view(1, b2, -1), dim=-1, p=2)
        
        sim_matrix = torch.max(torch.cat((sim_matrix_s.unsqueeze(-1),sim_matrix_v.unsqueeze(-1)),dim=-1),dim=-1)[0]
        eps, index = sim_matrix.topk(self.knn_k, dim=1, largest=False, sorted=True)  # (b1, k)
        
        state_norm, index = sim_matrix_s.topk(self.knn_k, dim=1, largest=False, sorted=True)  # (b1, k)
        value_norm, index = sim_matrix_v.topk(self.knn_k, dim=1, largest=False, sorted=True)  # (b1, k)
        
        eps = eps[:, -1] #k-th nearest distance
        eps = eps.reshape(-1, 1) # (b1, 1)
        
        state_norm = state_norm[:, -1] #k-th nearest distance
        state_norm = state_norm.reshape(-1, 1) # (b1, 1)

        value_norm = value_norm[:, -1] #k-th nearest distance
        value_norm = value_norm.reshape(-1, 1) # (b1, 1)
        
        sim_matrix_v = sim_matrix_v < eps
        n_v          = torch.sum(sim_matrix_v,dim=1,keepdim = True) # (b1,1)
        
        sim_matrix_s = sim_matrix_s < eps
        n_s          = torch.sum(sim_matrix_s,dim=1,keepdim = True) # (b1,1)
        reward       = torch.digamma((n_v+1).to(torch.float)) / ds + torch.log(eps * 2 + 0.00001)
        return reward * self.beta, n_v,n_s, eps, state_norm, value_norm



class LowPassSinglePole():
    ''' Classic simple filter '''
    def __init__(self, decay=0.6, vec_dim=None):
        self.b = 1.0 - decay

        if vec_dim is None:
            self.y = 0.0
        else:
            self.y = np.zeros(vec_dim)
    
    def filter(self, x):
        ''' filter a scalar or a vector of independent channels '''
        self.y += self.b * (x - self.y)
        return self.y