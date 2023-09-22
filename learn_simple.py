import gymnasium as gym
import numpy as np
import os, time, argparse
import torch as th
from torch.utils.tensorboard import SummaryWriter
from multiprocessing import Process
from utils import log_scalars
from agents.acw_v01_baseline2 import Agent

'''
    Simple script to test and tune continuous agents
    Makes few assumptions about env api, so should be widely compatible

    Features
    - Simultaneous multiple environments using multiprocessing
    - Multiple random seeds for statistical aggregation
    - Tensorboard logging and progress printing to terminal
    - Select options as cli arguments 
'''

def parse_args():
    parser = argparse.ArgumentParser(description='Train agent',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--log', action='store_true', help=('Enables Tensorboard logging'))
    parser.add_argument('--name', type=str, help=('Name or describe this run in the logs'))
    parser.add_argument('--seeds', type=int, default=1, help=("Number of random seeds per environment"))
    parser.add_argument('--cuda', action='store_true', help=('Use CUDA'))
    
    # TODO: As needed
    # parser.add_argument('--gui', action='store_true', help=('Enables visualisation'))
    # parser.add_argument('--save', action='store_true', help=('Enables saving checkpoints'))
    # parser.add_argument('--load', action='store_true', help=('Loads previously saved agent'))

    return parser.parse_args()

def run_env(current_time, env, env_name, random_seed, device, log, run_name, log_dir):

        # Unknown reason cannot use sub-dictionary directly, have to put in new variable
        update_steps    = env['update_steps']
        run_steps       = env['run_steps']

        # Create environment
        env         = gym.make(env_name)
        act_dim     = env.action_space.shape[0]      # different envs might provide this info differently
        obs_dim     = env.observation_space.shape[0] # different envs might provide this info differently
        obs, info   = env.reset(seed=random_seed)    # using the gym/gymnasium convention
        score       = np.zeros(1)                    # episodic score
        scoreboard  = np.zeros(1)                    # a list of episodic scores
        
        # Create agent
        agent = Agent(obs_dim, act_dim, num_steps=update_steps, device=device, seed=random_seed)

        print('\n>>> RUNNING ENV: ', env_name, "WITH AGENT: ", agent.name)
        print('ACTIONS DIM: ', act_dim, ' OBS DIM: ', obs_dim, "\n")
        print(agent.h, '\n')
        
        # New log for this environment
        if log:
            writer = SummaryWriter(f"{log_dir}/{current_time}/{env_name}/{agent.name}_seed:{random_seed}")
            writer.add_text(tag=f"score/{run_name}", text_string=str(run_name) + " | "+ str(agent.h), global_step=0)

        sps_timer = time.time()
        for step in (range(run_steps)):

            # Step the environment and collect observations, shape: (batch, channels) as tensors
            obs_th = th.tensor(obs, device=device).unsqueeze(0)
            action_th = agent.choose_action(obs_th)
            action_np = action_th.cpu().squeeze().numpy()
            obs, reward, terminated, truncated, info = env.step(action_np)
            done = (terminated or truncated)
            done_th = th.tensor(done, device=device).unsqueeze(0)
            reward_th = th.tensor(reward, device=device).unsqueeze(0).unsqueeze(0)
            agent.store_transition(reward_th, done_th)

            score += reward

            if done:
                scoreboard = np.append(scoreboard, score)
                obs, info  = env.reset(seed=random_seed)
                if log: 
                    writer.add_scalar(f"score/{env_name}", score, step)
                else:
                    print(env_name,'-',random_seed, 'Step: ', step, 'Score: %0.3f' % score[0])
                score = np.zeros(1)
            
            if (((step + 1) % update_steps) == 0) and (step > 128):
                sps = update_steps / (time.time() - sps_timer)
                
                update_timer = time.time()
                agent.update()
                training_time = (time.time() - update_timer)
                print(env_name,'-',random_seed, ' Step: ',step, 'Updated in: %0.3f' % training_time, 'secs', 'SPS: %0.1f' % sps)
                if log:
                    writer.add_scalar("perf/Update", training_time, step)
                    writer.add_scalar("perf/SPS", sps, step)
                    log_scalars(writer, agent, step)

                # start clock once update done
                sps_timer = time.time()
        
        # Tidy up when done
        print(env_name,'-',random_seed, 'FINISHED. SUM OF SCORES: %0.1f' % np.sum(scoreboard))        
        if log: writer.close()
        env.close()
        del agent
        

def main():
    
    # System parameters
    args                = parse_args()
    random_seed         = 42    # default: 42 
    max_processes       = 32    # processes will be queued
    os.nice(10)                 # don't hog the system
    th.set_num_threads(1)       # threads per process, often 1 is most efficient when using cpu
    np.random.seed(random_seed)
    np.set_printoptions(precision=3)
    if args.cuda: 
        device          = 'cuda' # cuda can be faster when training is costly (large model or dataset)
    else:
        device          = 'cpu'  # cpu is very often fastest 
    log_dir             = 'runs' # tensorboard logging dir

    # https://gymnasium.farama.org
    # comment out undesired environments
    environments = {}
    environments['LunarLanderContinuous-v2']    = {'run_steps': int(120e3), 'update_steps': 1024 * 1}
    environments['BipedalWalker-v3']            = {'run_steps': int(400e3), 'update_steps': 1024 * 2}
    # environments['BipedalWalkerHardcore-v3']    = {'run_steps': int(  2e6), 'update_steps': 1024 * 2}

    # start time of all runs
    current_time = time.strftime('%j_%H:%M')
    start_runs = time.time()

    processes = []
    total_sleep = 0.0

    print("\nEXPERIMENT NAME: ", args.name)
    if args.log:
        print(f'>>> TENSORBOARD -> ENABLED IN /{log_dir}/')

    # Run a process for each seed of each env
    for seed in range(args.seeds):
        for env_name in environments.keys():
            
            p = Process(target=run_env,
            kwargs={
                'current_time': current_time,
                'env': environments[env_name],
                'env_name': env_name,
                'random_seed': random_seed + seed,
                'device': device,
                'log': args.log,
                'run_name': args.name,
                'log_dir': log_dir,
            })
            p.start()
            processes.append(p)

            # Check if the maximum number of processes is reached
            if len(processes) >= max_processes:
                # Wait for any of the processes to finish
                print(f"\n Running {len(processes)} environments")
                print("Waiting for a process to complete before starting next process")
                while len(processes) >= max_processes:
                    for proc in processes:
                        if not proc.is_alive():
                            processes.remove(proc)
                    time.sleep(0.1)    
                    total_sleep += 0.1

    # Wait for all processes to finish before continuing
    for p in processes:
        p.join()

    print("\nCompleted runs in %0.3f" % (time.time() - start_runs), "secs")
    print("--name:", args.name)
    print("Slept this long waiting for max_processes: %.03f" % total_sleep, ' secs')
        

##########################
if __name__ == '__main__':
    main()
