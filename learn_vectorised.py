import gymnasium as gym
import torch as th
import numpy as np
import os, time, argparse
from torch.utils.tensorboard import SummaryWriter
from utils import log_scalars
from agents.acw_v01_baseline1 import Agent

'''
Vectorised gym script to test and tune continous agents
- Uses gym.vector.AsyncVectorEnv
'''

def parse_args():
    parser = argparse.ArgumentParser(description='Train agent',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--log', action='store_true', help=('Enables Tensorboard logging'))
    parser.add_argument('--name', type=str, help=('Name or describe this run in the logs'))
    parser.add_argument('--cuda', action='store_true', help=('Use CUDA'))
    parser.add_argument('--vecs', type=int, default=1, help=("Number of vectorised environments"))

    # These features are not implemented yet
    # parser.add_argument('--gui', action='store_true', help=('Enables visualisation'))
    # parser.add_argument('--save', action='store_true', help=('Enables saving checkpoints'))
    # parser.add_argument('--load', action='store_true', help=('Loads previously saved agent'))

    return parser.parse_args()

def main():
    args        = parse_args()
    random_seed = 42
    num_vecs    = args.vecs
    os.nice(10)
    np.random.seed(random_seed)
    th.set_num_threads(1)
    np.set_printoptions(precision=0)
    if args.cuda: 
        device          = 'cuda' # cuda can be faster when training is costly (large model or dataset)
    else:
        device          = 'cpu'  # cpu is very often fastest 
    log_dir             = 'runs_vec' # tensorboard logging dir

    # https://gymnasium.farama.org
    # comment out undesired environments
    environments = {}
    environments['LunarLanderContinuous-v2']    = {'run_steps': int(120e3), 'update_steps': 1024}
    # environments['BipedalWalker-v3']            = {'run_steps': int(400e3), 'update_steps': 1024 * 2}
    # environments['BipedalWalkerHardcore-v3']    = {'run_steps': int(1e6), 'update_steps': 1024 * 2}

    # start time of all runs
    current_time = time.strftime('%j_%H:%M')

    # Run though all environments in list sequentially, each one vectortised num_vecs times
    for env_name in environments.keys():

        # Create environment
        envs        = [lambda: gym.make(env_name) for i in range(num_vecs)]
        vec_env     = gym.vector.AsyncVectorEnv(envs)
        act_dim     = vec_env.single_action_space.shape[0]
        obs_dim     = vec_env.single_observation_space.shape[0]
        obs, info   = vec_env.reset(seed=random_seed)
        score       = np.zeros(num_vecs)
        score_sum   = 0
        done_sum    = 0
        global_step = 0

        print('>>> RUNNING ENV   -> ', env_name, " NUM VECS: ",num_vecs)
        print('ACTIONS DIM: ', act_dim, ' OBS DIM: ', obs_dim)
        
        # Create agent
        agent = Agent(obs_dim, act_dim, num_steps=environments[env_name]['update_steps'], num_env=num_vecs, device=device)
        print('>>> RUNNING AGENT -> ', agent.name)

        # New log for this environment
        if args.log:
            print('>>> TENSORBOARD -> ENABLED')
            writer = SummaryWriter(f"{log_dir}/{current_time}/{env_name}/{agent.name}")
            # if args.name != None:
                # writer.add_text(tag=f"score/{env_name}", text_string=args.name, global_step=0)
            writer.add_text(tag=f"score/{env_name}", text_string=str(args.name) + " | "+ str(agent.h), global_step=0)


        sps_timer = time.time()
        for step in (range(1, environments[env_name]['run_steps'])):

            # Agent receives tensor and sends back tensor
            action = agent.choose_action(th.tensor(obs).to(device))
            obs, reward, terminated, truncated, info = vec_env.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            agent.store_transition(th.tensor(reward).to(device), th.tensor(done).to(device))
            global_step += 1 * num_vecs
            score += reward

            if done.any():
                score_sum += np.sum(score * done)
                done_sum  += np.sum(done)
                # print(f"Steps: {step}, Score: {score * done}")
                if args.log:
                    score_log = score * done # zero out non-completed episodes
                    # print("logging score: ", score_log)
                    for i in score_log:
                        if np.abs(i) > 0.0:
                            writer.add_scalar(f"score/{env_name}", i, step)
                score = score * (1 - done) # reset to zero done episodes

            if (((step) % (environments[env_name]['update_steps'])) == 0) and (step > 0):
                sps = (environments[env_name]['update_steps'] * num_vecs) / (time.time() - sps_timer)
                update_timer = time.time()
                agent.update()
                training_time = (time.time() - update_timer)
                
                # Update on progress
                print('\033[4m','On global step: ',global_step, 'Updated in: %0.3f' % training_time, 'secs', 'SPS: %0.1f' % sps,'\033[0m')
                if done_sum > 0:
                    print(f"Done episodes: {done_sum}, Mean Score: %0.0f" % (score_sum / done_sum), "\n")
                score_sum, done_sum = 0, 0
                
                if args.log:
                    writer.add_scalar("perf/Update", training_time, step)
                    writer.add_scalar("perf/SPS", sps, step)
                    log_scalars(writer, agent, step)

                # start clock once update done
                sps_timer = time.time()
        
        # Tidy up for running next environment
        if args.log: writer.close()
        vec_env.close()
        del agent
        print("[--name] was set to: ", args.name)
        

##########################
if __name__ == '__main__':
    main()