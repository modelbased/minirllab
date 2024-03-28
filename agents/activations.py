import torch
import torch.nn as nn

''' 
    Non-linear activations 
    Example usage: self.activation = torch.jit.script(LiSHT())
'''

# Used for normalisation in Dreamer v3 https://arxiv.org/abs/2301.04104
# Added scale and bias learnable parameters like layer_norm
# A useful stat.stackexchnage comment on the derivative of symlog https://stats.stackexchange.com/questions/605641/why-isnt-symmetric-log1x-used-as-neural-network-activation-function
class SymLog(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Learnable scale and bias like layer_norm
        self.scale = nn.Parameter(torch.ones(dim))
        self.bias  = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        x = (torch.sign(x) * torch.log(torch.abs(x) + 1.0)) * self.scale + self.bias
        return x


# LiSHT activation
# https://arxiv.org/abs/1901.05894
class LiSHT(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return x * torch.tanh(x)


# OFN activation https://arxiv.org/abs/2403.05996
# Unit ball normalisation to prevent gradient explosion
# "all values are strictly between 0 and 1 and the gradients will be tangent to the unit sphere"
class OFN(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return x / torch.linalg.vector_norm(x)