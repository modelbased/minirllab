import numpy as np
import torch as th
import time

""" 
    Utils for normalizing observations
    based on https://github.com/openai/gym/blob/master/gym/wrappers/normalize.py 
"""

# From gymnasium normalise wrapper
class Normalise():
    """This wrapper will normalize observations s.t. each coordinate is centered with unit variance.
    Note:
        The normalization depends on past trajectories
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    """

    def __init__(self, input_dim, epsilon=1e-8):
        """ epsilon: A stability parameter that is used when scaling the observations """
        self.epsilon = epsilon

        """Tracks the mean, variance and count of values."""
        self.mean  = np.zeros(input_dim, "float64")
        self.var   = np.ones(input_dim, "float64")
        self.count = 1e-4

    def update(self, x):
        """Updates the mean, var and count using the previous mean, var, count and batch values."""

        batch_mean  = np.mean(x, axis=0)
        batch_var   = np.var(x, axis=0) 
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count
        
        self.mean   = new_mean
        self.var    = new_var
        self.count  = new_count

    def new(self, input):
        """Normalises new data"""
        input = np.array(input)
        input = np.expand_dims(input, axis=0) # new fix
        self.update(input)
        norm_obs = (input - self.mean) / np.sqrt(self.var + self.epsilon)
        norm_obs = np.squeeze(norm_obs) # new fix
        return np.float32(norm_obs)


# Same as Normalise but slightly fewer calls, improving samples per second performance
class NormaliseFast():
    def __init__(self, input_dim, epsilon=1e-8):
        self.epsilon = epsilon

        self.mean  = np.zeros(input_dim, "float64")
        self.var   = np.ones(input_dim, "float64")
        self.count = 1e-4

    def new(self, input):
        input = np.array(input)
        
        delta = input - self.mean
        tot_count = self.count + 1

        M2 = (self.var * self.count) + np.square(delta) * self.count / tot_count
        
        self.mean   = self.mean + delta / tot_count
        self.var    = M2 / tot_count
        self.count  += 1

        norm_obs = (input - self.mean) / np.sqrt(self.var + self.epsilon)
        return np.float32(norm_obs)


# Same as Normalise, replacing numpy with torch
class NormaliseTorch():
    def __init__(self, input_dim, epsilon=1e-8, device='cpu'):
        self.device = device
        
        self.epsilon = th.tensor(epsilon, device=device)

        self.mean  = th.zeros(1, input_dim, dtype=th.float64, device=device)
        self.var   = th.ones(1, input_dim, dtype=th.float64, device=device)
        self.count = th.tensor(1e-4, device=device)

    def new(self, input):
        delta = input - self.mean
        tot_count = self.count + th.ones(1, device=self.device)

        M2 = (self.var * self.count) + th.square(delta) * self.count / tot_count
        
        self.mean   = self.mean + delta / tot_count
        self.var    = M2 / tot_count
        self.count  = self.count + th.ones(1, device=self.device)

        norm_obs = (input - self.mean) / th.sqrt(self.var + self.epsilon)
        return norm_obs.to(dtype=th.float32)
    

# GPT4 torchscript re-write of NormaliseTorch for better performance
class NormaliseTorchScript(th.nn.Module):
    def __init__(self, input_dim, epsilon=1e-8, device='cpu'):
        super(NormaliseTorchScript, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.epsilon = epsilon
        
        self.mean = th.zeros(1, input_dim, dtype=th.float64, device=device)
        self.var = th.ones(1, input_dim, dtype=th.float64, device=device)
        self.count = th.ones(1, device=device) * 1e-4  # ensure count has shape [1]
        self.one = th.ones(1, device=self.device)

    @th.jit.export
    def new(self, input):
        delta = input - self.mean
        
        tot_count = self.count + self.one
        M2 = (self.var * self.count) + (delta ** 2) * self.count / tot_count
        
        self.mean = self.mean + delta / tot_count
        self.var = M2 / tot_count
        self.count = self.count + self.one
        
        norm_obs = (input - self.mean) / th.sqrt(self.var + self.epsilon)
        return norm_obs.to(dtype=th.float32)



# For performance comparison
@th.jit.script
def symlog(x):
    # Element-wise symlog mapping
    return th.sign(x) * th.log(th.abs(x) + 1.0) 


def test_normalisations():
    DEVICE = 'cpu'
    old = Normalise(2)
    new = NormaliseFast(2)
    tor = th.jit.script(NormaliseTorchScript(2, device=DEVICE))

    # are they identical results?
    for i in range(100):
        o = old.new((1, i**2))
        n = new.new((1, i**2))
        data = th.tensor((1, i**2), device=DEVICE)
        t = tor.new(data)
        print("Deltas: ",o - n,"    ",o - t.cpu().detach().numpy())

    # Compare performance
    cycles = int(100e3)
    data = np.ndarray((1,2))

    start_time = time.time()
    for i in range(cycles):
        x = old.new(data)
    norm1_time = (time.time() - start_time)
    print("done 1")

    start_time = time.time()
    for i in range(cycles):
        x = new.new(data)
    norm2_time = (time.time() - start_time)
    print("done 2")

    data = th.tensor((1, 2), device=DEVICE)
    
    start_time = time.time()
    for i in range(cycles):
        x = tor.new(data)
    norm3_time = (time.time() - start_time)
    print("done 3")

    start_time = time.time()
    for i in range(cycles):
        x = symlog(data)
    norm4_time = (time.time() - start_time)
    print("done 4")

    print("Seconds per (100k) ops for old: ", norm1_time, "new: ", norm2_time, "torch: ", norm3_time, "symlog: ", norm4_time)

##########################
if __name__ == '__main__':
    test_normalisations()