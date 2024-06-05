import torch
import torch.nn as nn

''' 
    Non-linear activations 
    Example usage: self.activation = torch.jit.script(LiSHT())
'''

# Used for normalisation in Dreamer v3 https://arxiv.org/abs/2301.04104
# Added scale and bias learnable parameters like layer_norm
# A useful stat.stackexchnage comment on the derivative of symlog https://stats.stackexchange.com/questions/605641/why-isnt-symmetric-log1x-used-as-neural-network-activation-function
# TODO: receive shape like Rational to allow for (b,c,l) inputs; optional learnable like layer_norm
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
    
# From Adaptive Rational Activations to Boost Deep Reinforcement Learning (v5, 2024)
# rational function f(x) = Q(x) / P(x) where P and Q are polynomials and 
# (some) polynomial parameters are learnable
# https://arxiv.org/abs/2102.09407v5
class Rational(nn.Module):
    def __init__(self, num_dims=2, preset='lrelu') -> None:
        super().__init__()
        # m > n allows rationals to implicitly make use of residual connections
        m = 6
        n = 4

        # f(x) = Sum(P(x)) / 1 + Sum(Q(x)) where P and Q are polynomials
        # P(x) = ax^j , Q(x) = bx^k where a and b are learnable
        # whilst j and k are fixed integer power coefficients
        a, b = self._presets(preset)
        self.a = a
        self.b = b
        self.j = torch.arange(m)
        self.k = torch.arange(n) + 1

        # input could be dim = 2 (batch, channels) or 
        # dim = 3 (batch, channels, length) for example
        # num_dims = x.dim() # if doing this dynamically
        p_shape = self.a.shape + (1,) * (num_dims)
        q_shape = self.b.shape + (1,) * (num_dims)

        # Reshape to broadcast with x
        self.abz = self.a.view(p_shape)
        self.jbz = self.j.view(p_shape)
        self.bbz = self.b.view(q_shape)
        self.kbz = self.k.view(q_shape)

        # Ensure they are learnable and/or moves to correct device
        self.ab = nn.Parameter(self.abz) 
        self.bb = nn.Parameter(self.bbz)
        self.register_buffer('jb', self.jbz) 
        self.register_buffer('kb', self.kbz) 


    # 45% faster or more than reference implementation depending on sizes
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Element-wise operation with broadcasting
        p = torch.sum(self.ab * x.pow(self.jb), dim=0)
        q = torch.sum((self.bb * x.pow(self.kb)).abs(), dim=0) + 1.0
        return p/q


    # From paper's own code base: https://github.com/ml-research/rational_activations/blob/master/rational/torch/rationals.py
    # See also here for eq. details: https://arxiv.org/abs/1907.06732
    # https://rational-activations.readthedocs.io/en/latest/index.html
    def reference(self, x):
        # Rational_PYTORCH_A_F
        # P(X) / Q(X) = a_0 + a_1 * X + ... + a_n * X^n / 1 + | b_1 * X | + | b_2 * X^2| + ... + | b_m * X ^m|

        weight_numerator = self.a
        weight_denominator = self.b

        z = x.view(-1)
        len_num, len_deno = len(weight_numerator), len(weight_denominator)
        # xps = torch.vander(z, max(len_num, len_deno), increasing=True)
        xps = self._get_xps(z, len_num, len_deno)
        numerator = xps.mul(weight_numerator).sum(1)
        expanded_dw = torch.cat([torch.tensor([1.]), weight_denominator, torch.zeros(len_num - len_deno - 1)])
        denominator = xps.mul(expanded_dw).abs().sum(1)
        return numerator.div(denominator).view(x.shape)
    

    def _get_xps(self, z, len_numerator, len_denominator):
        xps = list()
        xps.append(z)
        for _ in range(max(len_numerator, len_denominator) - 2):
            xps.append(xps[-1].mul(z))
        xps.insert(0, torch.ones_like(z))
        return torch.stack(xps, 1)


    # https://github.com/ml-research/rational_activations/blob/master/rational/rationals_config.json
    def _presets(self, name):
        # presets assume m=6 and n=4
        
        if name == 'lrelu':
            # leaky_relu upperbound=3, lowerbound=-3
            num = torch.tensor([0.029792778657264946, 0.6183735264987601, 2.323309062531321, 3.051936237265109, 1.4854203263828845, 0.2510244961111299])
            den = torch.tensor([-1.1419548357285474,4.393159974992486,0.8714712309957245, 0.34719662339598834])
        elif name == 'tanh':
            # tanh, ub=3, lb=-3
            num = torch.tensor([-1.0804622559204184e-08,1.0003008043819048,-2.5878199375289335e-08,0.09632129918392647,3.4775841628196104e-09,0.0004255709234726337])
            den = torch.tensor([-0.0013027181209176277,0.428349017422072,1.4524304083061898e-09,0.010796648111337176])
        elif name == 'sigmoid':
            # sigmoid,  ub=3, lb=-3
            num = torch.tensor([0.4999992534599381,0.25002157564685185,0.14061924500301096,0.049420492431596394,0.00876714851885483,0.0006442412789159799])
            den = torch.tensor([2.1694506382753683e-09,0.28122766100417684,1.0123620714203357e-05,0.017531988049946])
        elif name == 'gelu':
            # gelu, ub=3, lb=-3
            num = torch.tensor([-0.0012423594497499122,0.5080497063245629,0.41586363182937475,0.13022718688035761,0.024355900098993424,0.00290283948155535])
            den = torch.tensor([-0.06675015696494944,0.17927646217001553,0.03746682605496631,1.6561610853276082e-10])
        elif name == 'swish':
            # swish, ub=3, lb=-3
            num = torch.tensor([3.054879741161051e-07,0.5000007853744493,0.24999783422824703,0.05326628273219478,0.005803034571292244,0.0002751961022402342])
            den = torch.tensor([-4.111554955950634e-06,0.10652899335007572,-1.2690007399796238e-06,0.0005502331264140556])
        else:
            print("Error: No such preset for rational activations")
            exit()

        return num, den