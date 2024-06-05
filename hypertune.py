import gymnasium as gym
import numpy as np
import os, time, math
import torch as th
from torch.utils.tensorboard import SummaryWriter
from multiprocessing import Process, Queue
from bayes_opt import BayesianOptimization, UtilityFunction
from utils import log_scalars
from agents.sac_crossq_trace import Agent

'''
    Hyperparameter tuning using bayesian optimisation
    - Bayes Opt from https://github.com/fmfn/BayesianOptimization
    – Simultenous multiprocessing with max_workers and configurable cpu and cuda workers
    - Median pruner, based on https://araffin.github.io/post/hyperparam-tuning/ 
'''


class Pruner():
    ''' Assumes target_metric is reward related therefore maximise to positive is better '''
    def __init__(self):
        self.window = 0.1 # allows this far below the median before pruning
        self.target_metrics_history = {}

    def decide(self, target_metric, prune_chkpoint_idx):

        # Ensure there is a history for the current checkpoint
        if prune_chkpoint_idx not in self.target_metrics_history:
            self.target_metrics_history[prune_chkpoint_idx] = []

        # Only add this target_metric to the list if it is above the existing median, or the median will decline with time
        metrics = self.target_metrics_history[prune_chkpoint_idx]
        if len(metrics) == 0 or target_metric > np.median(metrics):
            self.target_metrics_history[prune_chkpoint_idx].append(target_metric)

        # Prune if below (median - x%) for this checkpoint - less aggressive pruning, since RL is noisy
        prune = len(metrics) > 0 and target_metric < (np.median(metrics) - np.median(np.abs(metrics)) * self.window)

        # return prune  # returns true if process should be pruned, false if process keeps going
        return False  # returns true if process should be pruned, false if process keeps going


def run_env(cmd_queue, sample_queue, res_queue, current_time, environment, env_name, random_seed, device, process_num, prune_chkpoints):

        # Consider target metric carefully, depends on env and impacts pruning considerations
        # Scoreboard is cleared between pruning checkpoints
        # target_metric = lambda: np.median(scoreboard) 
        # target_metric = lambda: np.mean(scoreboard)
        target_metric = lambda: np.mean(np.sort(scoreboard)[int(len(scoreboard)*0.25):int(len(scoreboard)*0.75)]) # interquartile mean (iqm)

        print("\nSTARTED WORKER PROCESS: ", process_num)
        print("TIME: ", current_time, "ENV: ", env_name, "DEVICE: ", device)


        # Receive and run through sample points
        while True:

            # Wait here for next point to sample
            sample_point    = sample_queue.get(block=True)
            next_point      = sample_point["next_sample"]
            sample_num      = sample_point["sample_num"]
            r_seed_mixed    = random_seed + sample_num # because bayes-optim can pick duplicate sample points, ensure each is actually different

            # Unknown reason cannot use sub-dictionary directly, have to put in new variable
            run_steps       = environment['run_steps']
            assert run_steps % prune_chkpoints == 0, "prune_chkpoints must be a multiple of run_steps"

            # Create environment
            env         = gym.make(env_name) 
            env_spec    = {
                'act_dim' : env.action_space.shape[0],
                'obs_dim' : env.observation_space.shape[0],
                'act_max' : env.action_space.high,
                'act_min' : env.action_space.low,
            }
            obs, info   = env.reset(seed=r_seed_mixed)
            score       = np.zeros(1) # total rewards for episode
            scoreboard  = np.zeros(1) # array of scores

            # Create agent with pre-initialisation hyperparameters
            # Remeber math.pow(10, p) for parameters using log ranges in hyperparameter_bounds
            agent = Agent(env_spec, 
                          buffer_size       = run_steps, 
                          device            = device, 
                          seed              = random_seed, 
                        #   rr                = next_point['replay_ratio'], 
                        #   q_lr              = math.pow(10, next_point['q_lr']), 
                        #   actor_lr          = math.pow(10, next_point['actor_lr']),
                        #   alpha_lr          = math.pow(10, next_point['alpha_lr']),
                          )

            # Hyperparameters post agent initialisation, modified after instantiation
            # agent.h.gamma                   = next_point['gamma']
            # agent.h.gae_lambda              = next_point['gae_lambda']
            # agent.h.clip_coef               = next_point['clip_coef']
            # agent.h.ent_coef                = next_point['ent_coef']
            # agent.h.vf_coef                 = next_point['vf_coef']
            # agent.h.max_grad_norm           = next_point['max_grad_norm']
            # agent.h.max_kl                  = next_point['max_kl']       
            # agent.h.update_epochs           = int(next_point['update_epochs']) 
            # agent.h.mb_size                 = int(next_point['mb_size'])    
            
            # New log for this environment
            writer = SummaryWriter(f"runs/hypertune/{current_time}/{env_name}/{agent.name}/{str(sample_num)}")
            writer.add_text("hyperparameters","seed: " + str(r_seed_mixed) + "\n\n" + "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(agent.h).items()])),)

            # Main loop: run environment for run_steps steps
            sps_timer = time.time()
            update_step_counter = 0
            prune_chkpoint = 0
            for step in range(run_steps):

                # Step the environment and collect observations, shape: (batch, channels)
                obs_th    = th.tensor(obs, device=device, dtype=th.float32).unsqueeze(0)
                action_th = agent.choose_action(obs_th)
                action_np = action_th.cpu().squeeze().numpy()
                obs, reward,   terminated, truncated, info = env.step(action_np)
                done_np   = (terminated or truncated)
                done_th   = th.tensor(done_np, device=device, dtype=th.bool).unsqueeze(0)
                reward_th = th.tensor(reward, device=device, dtype=th.float32).unsqueeze(0).unsqueeze(0)
                agent.store_transition(reward_th, done_th)

                # Episodic score
                score += reward

                # Track episodic score
                if done_np:
                    scoreboard = np.append(scoreboard, score)
                    obs, info  = env.reset(seed=r_seed_mixed)
                    writer.add_scalar(f"score/{env_name}", score, step)
                    writer.add_scalar(f"score/target_metric", target_metric(), step)
                    score = np.zeros(1)

                # Call at every step, agent decides if an update if due
                updated, update_time = agent.update()

                # Track samples per second
                if step % 2048 == 0 and step != 0:
                    sps = 2048 / (time.time() - sps_timer)
                    sps_timer = time.time()
                    writer.add_scalar("perf/SPS", sps, step)
                    print(env_name,'\t',sample_num, '\tStep: ',step, '\tSPS: %0.1f' % sps, '\tTarget Metric: %0.1f' % target_metric())
                
                # Log agent update metrics, but not too often
                update_step_counter += 1
                if updated and update_step_counter >= 2048:
                    writer.add_scalar("perf/Update", update_time, step)
                    log_scalars(writer, agent, step)
                    update_step_counter = 0
                
                # send interim result back for pruning control except at 0 and final steps
                if (step % (run_steps // prune_chkpoints) == 0 and step != 0 and step != (run_steps - 1)):
                    
                    result = {'process_num': process_num, 'target_metric': target_metric(), 'sampled_point': next_point, 'prune_chkpoint':prune_chkpoint}
                    res_queue.put(result)

                    # Block until a cmd addressed to this process has been received
                    while True:
                        cmd = cmd_queue.get()
                        if cmd['process_num'] == process_num:
                            break

                    prune_chkpoint += 1

                    # Break from running steps if we're pruned, log and go restart with new sample point
                    if (cmd['process_num'] == process_num) and (cmd['break'] == True):
                        writer.add_hparams(hparam_dict=next_point, metric_dict={'score/target metric': target_metric()}, run_name=str(sample_num))
                        print(env_name,'\t',sample_num, '\tStep: ',step, '\tSPS: %0.1f' % sps, '\tTarget Metric: %0.1f' % target_metric(), '\tPRUNED at ', prune_chkpoint)
                        break

                    # reset scoreboard at checkpoint, after pruner has decided
                    scoreboard  = np.zeros(1)

            # Got to the end without being pruned, log hyperparameters
            if step == (run_steps - 1):
                writer.add_hparams(hparam_dict=next_point, metric_dict={'score/target metric': target_metric()}, run_name=str(sample_num))

                # Return target metric to optimiser, indicate this is not a pruning chkpoint
                result = {'process_num': process_num, 'target_metric': target_metric(), 'sampled_point': next_point, 'prune_chkpoint':prune_chkpoints + 1}
                res_queue.put(result)

            # Tidy up
            writer.close()
            env.close()
            del agent
        

def main():
    
    # System parameters
    random_seed         = 42                                    # default: 42
    num_workers_cpu     = 0                                     # number of workers in cpu, cpu can be faster sometimes (e.g. smaller model sizes & cpu environment)
    num_workers_cuda    = 8                                     # number of workers in cuda 
    max_workers         = num_workers_cpu + num_workers_cuda    # Max number of workers
    prune_chkpoints     = 4                                     # should be divisor of run_steps; this many pruning section per run_steps
    os.nice(10)                                                 # don't hog the system
    th.set_num_threads(2)                                       # usually no faster with more, but parallel runs are more efficient when th.threads=1 (depending on hardware!)
    np.random.seed(random_seed)                                 # also given to agent

    # Select only one environment unless reward/target_metric has been scaled
    # https://gymnasium.farama.org
    environments = {}

    # Box 2D (action range -1..+1)
    # environments['LunarLanderContinuous-v2']    = {'run_steps': int(120e3)}
    # environments['BipedalWalker-v3']            = {'run_steps': int(400e3)}
    environments['BipedalWalkerHardcore-v3']    = {'run_steps': int(  1e6)}
    
    # Mujoco (action range -1..+1 except Humanoid which is -0.4..+0.4)
    # environments['Ant-v4']                      = {'run_steps': int(20e3)}
    # environments['HalfCheetah-v4']              = {'run_steps': int(400e3)}
    # environments['Walker2d-v4']                 = {'run_steps': int(400e3)}
    # environments['Humanoid-v4']                 = {'run_steps': int(200e3)}

    # Start time of all runs
    current_time = time.strftime('%j_%H:%M')

    # Register agent hyperparameters to optimise
    # Consider math.log10(x) & math.pow(10, x) for large OOM ranges
    hyperparameter_bounds = {
                        # 'gamma'         : (0.8, 0.9997),
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
                        # 'l2init_lambda_q'   : (math.log10(1e-5), math.log10(1e-2)),
                        # 'l2init_lambda_a'   : (math.log10(1e-5), math.log10(1e-2)),
                        'replay_ratio'      : (1, 8),
                        # 'q_lr'              : (math.log10(5e-4), math.log10(3e-3)),
                        # 'alpha_lr'          : (math.log10(5e-4), math.log10(3e-3)),
                        # 'actor_lr'          : (math.log10(6e-5), math.log10(9e-4)),
                        }

    # Setup bayesian optmiser. RL is noisy, probing duplicate points is valid. Can crash if not allowed.
    # Utility function EI is preferred for robustness with noisy objective functions whilst exploring landscape
    optimiser = BayesianOptimization(f=None, pbounds=hyperparameter_bounds, allow_duplicate_points=True, random_state=random_seed)
    utility   = UtilityFunction(kind="ucb")
    # utility   = UtilityFunction(kind="ei") # hypertune the optim-hyper-parameters!

    # Process and queues bookeeping
    processes = []
    sample_queue = Queue() # Send sampling points to workers
    res_queue = Queue() # Receive results from workers
    cmd_queue = Queue()
    sample_count = 0 # bayes-optim sample point number
    pruner = Pruner()

    # Start processes for listed environments 
    # Assumes if different envs that metric is normalised/scaled appropriately 
    for _ in range(max_workers):
        for env_name in environments.keys():
            
            # Allocate workers to device
            if len(processes) < num_workers_cpu:
                device = 'cpu'
            else:
                device = 'cuda'
            
            # Stop creating workers when we reach max_workers
            if len(processes) >= max_workers:
                break

            # Each worker is a multiprocessing process connected back here with queues
            p = Process(target=run_env,
            kwargs={
                'cmd_queue': cmd_queue,
                'sample_queue': sample_queue,
                'res_queue': res_queue,
                'current_time': current_time,
                'environment': environments[env_name],
                'env_name': env_name,
                'random_seed': random_seed,
                'device': device,
                'process_num': len(processes),
                'prune_chkpoints': prune_chkpoints,
            })
            p.start()
            processes.append(p)

            # Fetching multiple suggestions before any results 
            # are registered generates different suggestions to start with
            next_point = optimiser.suggest(utility)
            sample_point = {"next_sample":next_point, "sample_num":sample_count}
            sample_queue.put(sample_point)
            sample_count += 1

    # Optimise forevermore (or ctl-c)
    while True:
        try:
            result = res_queue.get(block=True) # wait until a result is in
            
            if result['prune_chkpoint'] <= prune_chkpoints:

                prune_this_run = pruner.decide(result['target_metric'], result['prune_chkpoint'])
                
                if prune_this_run == True:
                    cmd_queue.put({"process_num":result['process_num'], "break":True})
                    
                    optimiser.register(params=result['sampled_point'], target=result['target_metric'])
                    
                    next_point = optimiser.suggest(utility)
                    sample_point = {"next_sample":next_point, "sample_num":sample_count}
                    sample_queue.put(sample_point)
                    sample_count += 1
                else:
                    cmd_queue.put({"process_num":result['process_num'], "break":False})
            
            else:
                optimiser.register(params=result['sampled_point'], target=result['target_metric'])
                
                print("\nOPTIMISER SAMPLE: ", sample_count,' ',result['sampled_point'], "TARGET METRIC: ", result['target_metric'], "\n")
                
                next_point = optimiser.suggest(utility)
                sample_point = {"next_sample":next_point, "sample_num":sample_count}
                sample_queue.put(sample_point)
                sample_count += 1
        
        except KeyboardInterrupt:
            print("Stopping on keyboard interrupt")
            break

    # finished, terminate the processes
    for p in processes:
        p.join()
    exit()

##########################
if __name__ == '__main__':
    main()