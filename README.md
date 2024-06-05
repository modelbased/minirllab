# Mini RL Lab

### Easy Agent Experiments for Beginners

I wrote this set of scripts to help me research and experiement with the latest concepts in RL, as well as a way to learn Python and PyTorch. 

It is a setup and workflow that works well for me to debug and experiment with concepts like agent algorithms, world models, planning, plasticity, transformers etc, and other beginners might find it a useful starting point for their own experiments.

The focus of Mini RL Lab is on continous control gym-like environments aimed at physical systems or expensive-to-sample simulations - my personal research interest, and it makes the problem and architecture space tractable.

The basis are CleanRL's PPO and SAC agents [https://github.com/vwxyzjn/cleanrl] which I modified to:

1. Separate the environment rollout and logging from the agent code. CleanRL's single file approach is great but I find this arrangement easier for experiments
2. Simplify the code, improve performance where possible
3. Use different specialised training scripts
4. Include algorithms + variants as baselines with which to compare

### Benefits

1. Agents based on established, tried and tested baselines from CleanRL
2. Agents are structured for easy experimentation, whilst staying ~"one file"
3. Various performance considerations such as minimising cpu<>gpu syncs, data transfers from buffer etc
   1. Helpful for those of us limited to one workstation and a midrange GPU 
4. Inline comments document design choices and links to source papers
5. Learn scripts implement a lot best practices I discovered as I went, minor (data logging structure) to major (multiprocessing allows running a number of parallel agents with different seeds, essential in RL)  

### Prerequisites

* Pytorch 2 (though 1.x will work with small changes)
* Numpy (1.25 though older should work)
* Tensorboard
* Gymnasium[Box2D] and/or [Mujoco] (https://gymnasium.farama.org)
  * Or other gym compatible environment of choice
* Bayesian Optimisation (https://github.com/bayesian-optimization/BayesianOptimization)

### Quickstart

Test a change quickly for major errors:

`Python learn_simple.py`

Training run with multiple random seeds logging to tensorboard:

`Python learn_simple.py --log --seed 8 --name "testing X new feature"`

Run a vectorised environment with cuda and log:

`Python learn_vectorised.py --log --cuda`

Use bayesian optimisation to optimise hyperparameter(s):

`Python hypertune.py`

### Usage Notes

* ppo_baseline
  * Based on CleanRL's continuous PPO agent 
  * Simplified for easy modifictions
  * Improved samples per second through small optimisations

* sac_baseline
  * Based on CleanRL's continuous SAC agent
  * Simplified for easy modification
  * Removed CUDA <> CPU synchronisations for better performance
  * Variants: DroQ and CrossQ
    * CrossQ in particular is great, see the paper: https://arxiv.org/abs/1902.05605

* Novel agents
  * crossq_cem
    * Based on sac_crossq with actor replaced by a cross entropy method optimiser
    * Inspired by QT-Opt https://arxiv.org/abs/1806.10293 and TD-MPC2 https://arxiv.org/abs/2310.16828
    * Question: Can QT-Opt's performance improve to match TD-MPC2 using CrossQ's improvements to the Q functions?
    * Result: maybe, but CEM actor is so compute intensive it is not clear this is a direction worth pursuing
    * WIP, could be improved
  
  * **sac_crossq_bro**
    * Inspired by various papers showing that SAC can be improved with (a) more compute (b) regularisation (c) simple design changes
    * **Promising results**, first agent in Mini RL Lab that seems to reliably solve WalkerHardcore in <500k steps
    * WIP, not tuned or optimised yet
     
* learn_simple.py
  * Multiple training runs in parallel using multiprocessing (the processes have independent agents and environments)
  * Few assumptions about environments, more easily compatible with rl envs approximating the open ai gym api
  * Easy to edit and modify, design choices in comments
  * Use case: test performance of new feature on multiple environments with many random seeds in parallel

* learn_vectorised.py
  * No multiprocessing, runs a single process
  * PPO seems to really need different hyperparameters when vectorised
  * Use case: check performance when vectorised

* hypertune.py
  * uses bayesian optimisation to tune selected hyperparameters
  * uses multiprocessing to run multiple evaluations in parallel
  * implements a median pruner to stop badly performing runs early
  * Use case: optimise a new hyperparameter
  * Hyperparameters in RL https://arxiv.org/abs/2306.01324 is a good reference for this  