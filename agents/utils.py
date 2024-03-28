import torch
import torch.nn as nn

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
                nn.init.zeros_(layer.bias)
            elif init_method == 'kaiming_uniform_':
                nn.init.kaiming_uniform_(layer.weight, a=0, mode='fan_in', nonlinearity='linear')
                nn.init.zeros_(layer.bias)
            elif init_method == 'orthogonal_':
                nn.init.orthogonal_(layer.weight)
                nn.init.zeros_(layer.bias)


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