# Mini RL Lab

### Easy Agent Experiments for Beginners

I wrote this set of scripts to help me research and experiement with the latest papers in RL, as well as a way to learn Python and PyTorch.

The focus of Mini RL Lab is on continous control gym-like environments aimed at physical systems or expensive-to-sample simulations - my personal research interest, and it makes the problem and architecture space tractable on modest hardware.

The basis is CleanRL's PPO agent [https://github.com/vwxyzjn/cleanrl] which I modified to:

1. Separate the environment rollout and logging from the agent code. CleanRL's single file approach is great but I find this arrangement easier to use for experiments
2. Simplify the code
3. Use different specialised training scripts

### Prerequisites

* Pytorch 2 (though 1.x will work with small changes)
* Numpy (1.25 though older should work)
* Tensorboard
* Gymnasium[box2d] (https://gymnasium.farama.org) or your preferred env
* Bayesian Optimisation (https://github.com/bayesian-optimization/BayesianOptimization)


### Quickstart

Test a change quickly for major errors:

`Python learn_simple.py`

Training run with multiple random seeds logging to tensorboard:

`Python learn_simple.py --log --seed 8 --name "testing X new feature"`

Run a vectorised environment with cuda and log:

`Python learn_vectorised.py --log --cuda`

Use bayesian optimisation to optimise hyperparameter(s):

`Python hyperturne.py`

### Usage Notes

* acw_v01
  * "actor critic world", based on CleanRL's continuous PPO agent 
  * simplified for easy modifictions
  * uses pytorch only (no numpy) for running on GPU (with e.g. brax gym)
  * improved samples per second performance using torch.jit, torch.compile and other small optimisations
* learn_simple.py
  * Multiple training runs in parallel using multiprocessing (the processes have independent agents and environments)
  * few assumptions about environments, more easily compatible with rl envs approximating the open ai gym api
  * easy to edit and modify
* learn_vectorised.py
  * no multiprocessing, runs a single process
  * ppo seems to really need different hyperparameters in this case
* hypertune.py
  * uses bayesian optimisation to tune selected hyperparameters
  * uses multiprocessing to run multiple evaluations in parallel


### Useful Links

I found these particularly useful early on:

* https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
* https://stable-baselines.readthedocs.io/en/master/guide/rl_tips.html
* https://github.com/seungeunrho/minimalRL
* https://andyljones.com/posts/rl-debugging.html
* https://github.com/aowen87/ppo_and_friends
  
