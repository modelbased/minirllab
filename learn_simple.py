import gymnasium as gym
import numpy as np
import os, time, argparse, random
import torch as th
from torch.utils.tensorboard import SummaryWriter
from multiprocessing import Process
import multiprocessing as mp
from utils import log_scalars, Colour
from agents.cross_opt import Agent

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
    parser.add_argument('--acc', action='store_true', help=('Use an accelerator'))
    parser.add_argument('--baseline', action='store_true', help=('Saves logs in baseline sub-folder for archiving'))
    parser.add_argument('--save', action='store_true', help=('Enables saving checkpoints'))
    parser.add_argument('--load', action='store_true', help=('Loads previously saved agent'))
    parser.add_argument('--quiet', action='store_true', help=('Loads previously saved agent'))

    
    # TODO: As needed
    # parser.add_argument('--gui', action='store_true', help=('Enables visualisation'))

    return parser.parse_args()

def run_env(current_time, env, env_name, random_seed, device, args, log_dir):

        # Configure torch in spwaned child process
        th.set_num_threads(2)       
        th.manual_seed(random_seed)
        th.backends.cudnn.deterministic = True

        # Unknown reason cannot use sub-dictionary directly, have to put in new variable
        run_steps       = env['run_steps']

        # Create environment
        env         = gym.make(env_name)
        env_spec    = {
            'act_dim' : env.action_space.shape[0],
            'obs_dim' : env.observation_space.shape[0],
            'act_max' : env.action_space.high,
            'act_min' : env.action_space.low,
        }
        obs, info   = env.reset(seed=random_seed)    # using the gym/gymnasium convention
        score       = np.zeros(1)                    # episodic score
        scoreboard  = np.zeros(1)                    # a list of episodic scores
        
        # Create agent
        agent = Agent(env_spec, buffer_size=run_steps, device=device, seed=random_seed)
        if args.load: 
            agent.load()

        print('\n>>> RUNNING ENV: ', env_name, "WITH AGENT: ", agent.name)
        print('ACTIONS DIM: ', env_spec['act_dim'], ' OBS DIM: ', env_spec['obs_dim'], "\n")
        print(agent.h, '\n')
        
        # New log for this environment
        if args.log:
            writer = SummaryWriter(f"{log_dir}/{current_time}/{env_name}/{agent.name}_seed:{random_seed}")
            writer.add_text("hyperparameters",str(args.name) + "\n\n" + "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(agent.h).items()])),)

        sps_timer           = time.time()
        update_log_counter = 0 # avoid GB size logs when agent updates frequently
        updated             = False
        
        for step in range(run_steps):

            # Step the environment and collect observations, shape: (batch, channels) as tensors
            obs_th    = th.tensor(obs, device=device, dtype=th.float32).unsqueeze(0)
            action_th = agent.choose_action(obs_th)
            action_np = action_th.cpu().squeeze().to(dtype=th.float32).numpy()
            obs, reward, terminated, truncated, info = env.step(action_np)
            done      = (terminated or truncated)
            done_th   = th.tensor(done, device=device, dtype=th.int8).unsqueeze(0)
            reward_th = th.tensor(reward, device=device, dtype=th.float32).unsqueeze(0).unsqueeze(0)
            agent.store_transition(reward_th, done_th)
            
            # Episodic score
            score += reward

            # Track episodic score
            if done:
                scoreboard = np.append(scoreboard, score)
                obs, info  = env.reset(seed=random_seed)
                if args.log: 
                    writer.add_scalar(f"score/{env_name}", score, step)
                if (not args.quiet) and (not args.log):
                    print(env_name,'-',random_seed, 'Step: ', step, 'Score: %0.1f' % score[0])
                score = np.zeros(1)
            
            # Call at every step, agent decides if an update if due
            updated, update_time = agent.update()

            # If saving checkpoints do it at x% total steps intervals
            if args.save and step % int(run_steps * 0.25) == 0:
                agent.save()

            # Track samples per second
            if step % 1024 == 0 and step != 0:
                sps = 1024 / (time.time() - sps_timer)
                sps_timer = time.time()

                # ups is indicative, since it's just 1/time_taken, if updates are coupled serially to env interaction
                ups = update_time and 1.0 / update_time or 0.0 # return zeros when dividing by zero

                if args.log:
                    writer.add_scalar("perf/SPS", sps, step)
                    # Median of latest 25% of scores
                    print(env_name,'-',random_seed, 'Step: ',step, 'SPS: %0.1f' % sps, 'UPS: %0.1f' % ups, 'Scoreboard Median: %0.1f' % np.median(scoreboard[int(len(scoreboard) * 0.75): len(scoreboard)]))
                else:
                    print(Colour.BLUE,env_name,'-',random_seed, 'Step: ',step, 'SPS: %0.1f' % sps, 'UPS: %0.1f' % ups, Colour.END)

            # Log agent update metrics, but not too often
            update_log_counter += 1
            if updated and args.log and update_log_counter >= 1024:
                writer.add_scalar("perf/Update", update_time, step)
                log_scalars(writer, agent, step)
                update_log_counter = 0

        # Tidy up when done
        print(env_name,'-',random_seed, 'FINISHED. MEDIAN OF FINAL 25pct of SCORES: %0.1f' % np.median(scoreboard[int(len(scoreboard) * 0.75): len(scoreboard)]))        
        if args.log: writer.close()
        env.close()
        del agent
        

def main():
    
    # System parameters
    args                = parse_args()
    random_seed         = 42    # default: 42 
    max_processes       = 2     # small models (e.g. ppo) can have lots in cpu for greater total SPS
    th.set_num_threads(2)       # threads per process, often 1 is most efficient when using cpu with seed > 1
    os.nice(10)                 # don't hog the system
    np.random.seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    th.manual_seed(random_seed)
    th.backends.cudnn.deterministic = True
    np.set_printoptions(precision=3)
    mp.set_start_method('spawn') 
    
    if args.acc:
        if th.cuda.is_available(): 
            device = 'cuda'  # cuda is faster when training is costly (large model or dataset)
        else:
            if th.backends.mps.is_available():
                device = 'mps' # Recent apple macs
            else:
                print("No accelerator available, exiting.")
                exit()
    else:
        device = 'cpu'   # cpu is often faster for PPO (small models)
    
    if not args.baseline:
        log_dir = 'runs'          # tensorboard logging dir
    else:
        log_dir = 'runs/baseline' # baselined agents archive for future comparisons

    # Comment out undesired environments. Simultaneous processes: (num_envs * num_seeds) ≤ max_processes
    # https://gymnasium.farama.org
    environments = {}
    
    # Box 2D (action range -1..+1)
    environments['LunarLanderContinuous-v2']    = {'run_steps': int(120e3)}
    environments['BipedalWalker-v3']            = {'run_steps': int(400e3)}
    # environments['BipedalWalkerHardcore-v3']    = {'run_steps': int(400e3)}
    
    # Mujoco (action range -1..+1 except Humanoid which is -0.4..+0.4)
    environments['Ant-v4']                      = {'run_steps': int(400e3)}
    environments['HalfCheetah-v4']              = {'run_steps': int(400e3)}
    environments['Walker2d-v4']                 = {'run_steps': int(400e3)}
    environments['Humanoid-v4']                 = {'run_steps': int(400e3)}
    environments['HumanoidStandup-v4']          = {'run_steps': int(400e3)}

    # start time of all runs
    current_time = time.strftime('%j_%H:%M')
    start_runs = time.time()

    processes = []
    total_sleep = 0.0

    print("\nEXPERIMENT NAME: ", args.name)
    if args.log:
        print(f'>>> TENSORBOARD -> ENABLED IN /{log_dir}/{current_time}')

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
                'args': args,
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

    print("\n")
    print("Completed runs in %0.3f" % (time.time() - start_runs), "secs")
    print("Completed runs in %0.3f" % ((time.time() - start_runs) / 3600), "hours")
    print("Log dir: /",log_dir,'/',current_time,'/')
    print("\n--name:", args.name)
    print("Slept this long waiting for max_processes: %.03f" % total_sleep, ' secs')
        

##########################
if __name__ == '__main__':
    main()
