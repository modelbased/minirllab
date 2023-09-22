import torch
import torch.nn as nn

''' 
    Non-linear activations 
    Example usage: self.activation = torch.jit.script(LiSHT())
'''

# Used for normalisation in Dreamer v3
# https://arxiv.org/abs/2301.04104
class SymLog(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return torch.sign(x) * torch.log(torch.abs(x) + 1.0 )


# LiSHT activation
# https://arxiv.org/abs/1901.05894
class LiSHT(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return x * torch.tanh(x) 