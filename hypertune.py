import gymnasium as gym
import numpy as np
import os, time
import torch as th
from torch.utils.tensorboard import SummaryWriter
from multiprocessing import Process, Queue
from bayes_opt import BayesianOptimization, UtilityFunction
from utils import log_scalars
from agents.acw_v01_baseline1 import Agent

'''
    Hyperparameter tuning using bayesian optimisation
    https://github.com/fmfn/BayesianOptimization
    – Simultenous multiprocessing with max_processes
'''

def run_env(cmd_queue, res_queue, current_time, environment, env_name, random_seed, device, process_num):

        print("\nSTARTED WORKER PROCESS: ", process_num)
        print("TIME: ", current_time, "ENV: ", env_name, "DEVICE: ", device)

        # Receive and run through sample points
        while True:

            # Wait here for next point to sample
            sample_point    = cmd_queue.get(block=True)
            next_point      = sample_point["next_sample"]
            sample_num      = sample_point["sample_num"]

            # Unknown reason cannot use sub-dictionary directly, have to put in new variable
            update_steps    = environment['update_steps']
            run_steps       = environment['run_steps']

            # Create environment
            env         = gym.make(env_name) 
            act_dim     = env.action_space.shape[0]
            obs_dim     = env.observation_space.shape[0]
            obs, info   = env.reset(seed=random_seed)
            score       = np.zeros(1)
            scoreboard  = np.zeros(1)

            # Hyparameters pre agent initialisation, passed on instantiation
            # weight_decay = next_point['weight_decay']
            
            # Create agent
            agent = Agent(obs_dim, act_dim, num_steps=update_steps, device=device, seed=random_seed)

            # Hyperparameters post agent initialisation, modified after instantiation
            agent.h.gamma                   = next_point['gamma']
            # agent.h.gae_lambda              = next_point['gae_lambda']
            # agent.h.clip_coef               = next_point['clip_coef']
            # agent.h.ent_coef                = next_point['ent_coef']
            # agent.h.vf_coef                 = next_point['vf_coef']
            # agent.h.max_grad_norm           = next_point['max_grad_norm']
            # agent.h.max_kl                  = next_point['max_kl']       
            # agent.h.adam_lr                 = next_point['adam_lr']
            # agent.h.update_epochs           = int(next_point['update_epochs']) 
            # agent.h.mb_size                 = int(next_point['mb_size'])    
            
            # New log for this environment
            writer = SummaryWriter(f"hypertune/{current_time}/{env_name}/{agent.name}/{str(sample_num)}")

            # Main loop: run environment for run_steps steps
            sps_timer = time.time()
            for step in range(run_steps):

                # Step the environment and collect observations, shape: (batch, channels)
                obs_tensor = th.tensor(obs, device=device).unsqueeze(0)
                action = agent.choose_action(obs_tensor)
                action_numpy = action.cpu().squeeze().numpy()
                obs, reward, terminated, truncated, info = env.step(action_numpy)
                done = (terminated or truncated)
                done_tensor = th.tensor(done, device=device).unsqueeze(0)
                reward_tensor = th.tensor(reward, device=device).unsqueeze(0).unsqueeze(0)
                agent.store_transition(reward_tensor, done_tensor)

                score += reward

                if done:
                    scoreboard = np.append(scoreboard, score)
                    obs, info  = env.reset(seed=random_seed)
                    writer.add_scalar(f"score/{env_name}", score, step)
                    score = np.zeros(1)
                
                if (((step + 1) % update_steps) == 0) and (step > 128):
                    sps = update_steps / (time.time() - sps_timer)
                    
                    update_timer = time.time()
                    agent.update()
                    training_time = (time.time() - update_timer)
                    
                    writer.add_scalar("perf/Update", training_time, step)
                    writer.add_scalar("perf/SPS", sps, step)
                    log_scalars(writer, agent, step)
                    print("Process: ", process_num, "SPS: %.0f" % sps, "Update time: %.1f" % training_time)

                    # Start clock once update done
                    sps_timer = time.time()
            
            target_metric = np.sum(scoreboard)
            writer.add_hparams(hparam_dict=next_point, metric_dict={'score/target metric': target_metric}, run_name=str(sample_num))

            # Return score to optimiser here
            result = {'target_metric': target_metric, 'sampled_point': next_point}
            res_queue.put(result)

            # Tidy up
            writer.close()
            env.close()
            del agent
        

def main():
    
    # System parameters
    random_seed         = 42        # default: 42 
    num_workers         = 8         # def: 8 (achieves 8k total SPS); actually, num_workers * num_envs ...!
    agent_uses_tensors  = True      # should ENV send and receive Tensors or Numpy ?
    device              = 'cpu'     # cpu is very usually faster 
    os.nice(10)                     # don't hog the system
    th.set_num_threads(1)           # no faster with more, and can run more sims in parallel this way
    np.random.seed(random_seed)
    np.set_printoptions(precision=3)
    th.set_printoptions(precision=3)

    # https://gymnasium.farama.org
    # comment out undesired environments
    environments = {}
    # environments['LunarLanderContinuous-v2']    = {'run_steps': int(120e3), 'update_steps': 1024 * 1}
    environments['BipedalWalker-v3']            = {'run_steps': int(400e3), 'update_steps': 1024 * 2}
    # environments['BipedalWalkerHardcore-v3']    = {'run_steps': int(  2e6), 'update_steps': 1024 * 2}

    # start time of all runs
    current_time = time.strftime('%j_%H:%M')

    # register agent hyperparameters to optimise
    hyperparameter_bounds = {
                        'gamma'         : (0.8, 0.9997),
                        # 'gae_lambda'    : (0.9, 1.0),
                        # 'clip_coef'     : (0.1, 0.3),
                        # 'ent_coef'      : (0.0, 0.01),
                        # 'vf_coef'       : (0.5, 1.0),
                        # 'max_grad_norm' : (0.1, 1.0),
                        # 'max_kl'        : (0.003, 0.03),
                        # 'adam_lr'       : (0.000005, 0.003),
                        # 'update_epochs' : (1, 10), # integer
                        # 'mb_size'       : (8, 128), # integer
                        # 'trace_length'  : (4, 512), # integer
                        # 'latent_dim'    : (4, 128) # integer 
                        # 'topk_pct'        : (0.001, 0.05),
                        # 'min_similarity'  : (0.5, 1.0),
                        # 'pivotal_reward'  : (0.01, 2.0),
                        # 'weight_decay'    : (0.0, 0.1),
                        }

    # Setup bayesian optmiser. RL is noisy, probing duplicate points is valid. Can crash if not allowed.
    optimiser = BayesianOptimization(f=None, pbounds=hyperparameter_bounds, allow_duplicate_points=True)
    # utility   = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)
    utility   = UtilityFunction(kind="ei", kappa=2.5, xi=1.0)

    # Process and queues bookeeping
    processes = []
    cmd_queue = Queue() # Send sampling points to workers
    res_queue = Queue() # Receive results from workers
    sample_count = 0 # bayes sample point number

    # Start processes for listed environments 
    # Assumes if different envs that metric is normalised/scaled appropriately 
    for _ in range(num_workers):
        for env_name in environments.keys():
            p = Process(target=run_env,
            kwargs={
                'cmd_queue': cmd_queue,
                'res_queue': res_queue,
                'current_time': current_time,
                'environment': environments[env_name],
                'env_name': env_name,
                'random_seed': random_seed,
                'device': device,
                'process_num': len(processes)
            })
            p.start()
            processes.append(p)

            # Fetching multiple suggestions before any results 
            # are registered generates different suggestions to start with
            next_point = optimiser.suggest(utility)
            sample_point = {"next_sample":next_point, "sample_num":sample_count}
            cmd_queue.put(sample_point)
            sample_count += 1

    # Optimise forevermore (or ctl-c)
    while True:
        result = res_queue.get(block=True) # wait until a result is in
        optimiser.register(params=result['sampled_point'], target=result['target_metric'])
        
        print("\nOPTIMISER SAMPLE: ", result['sampled_point'], "TARGET METRIC: ", result['target_metric'], "\n")
        
        next_point = optimiser.suggest(utility)
        sample_point = {"next_sample":next_point, "sample_num":sample_count}
        cmd_queue.put(sample_point)
        sample_count += 1


##########################
if __name__ == '__main__':
    main()