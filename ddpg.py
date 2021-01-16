import sys
import time
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
import torch

from agent import Agent
from hyperparameters import Hyperparameters
from replay_buffer import ReplayBuffer


def start_actor_critic_algorithm_unity():
    param=Hyperparameters()
    from unityagents import UnityEnvironment
    env = UnityEnvironment(file_name='./Reacher_20_Windows_x86_64/Reacher.exe')
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=param.TRAINMODE)[brain_name]
    # number of agents
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)
    # size of each action
    action_dim = brain.vector_action_space_size
    print('# dimensions of each action:', action_dim)
    # examine the state space
    states = env_info.vector_observations
    state_dim = states.shape[1]
    print('# dimensions of each state:', state_dim)
    device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ddpg(env,brain_name,num_agents,state_dim,action_dim,0,device,param)




def ddpg(env,brain_name,nr_agents,state_dimension,action_dimension,seed,device,param):
    filename_scores="scores_garbage.txt"
    all_scores=[]
    all_average_scores=[]
    scores_window = deque(maxlen=100)
    print_to_file("",filename_scores,True)
    agents=[]
    agent=Agent(state_dimension, action_dimension, seed,device,param)
    for i in range(nr_agents):
        # agents.append(Agent(state_dimension, action_dimension, seed,device,param))
        agents.append(agent)
    shared_replay_buffer=ReplayBuffer(action_dimension, param.BUFFER_SIZE, param.BATCH_SIZE, seed,device)
    max_score=0
    for i_episode in range(param.MAX_NR_EPISODES):
        start_time=time.time()
        for agent in agents:
            agent.reset()
        env_info = env.reset(train_mode=param.TRAINMODE)[brain_name]     # reset the environment
        states = env_info.vector_observations                  # get the current state (for each agent)
        # initialize the score (for each agent)
        scores = np.zeros(nr_agents)
        step=0
        while True:
            actions = np.array([agent.act(state) for agent,state in zip(agents,states)]).squeeze(1)
            # collect data from environment
            env_info = env.step(actions)[brain_name]       # send all actions to the environment
            next_states = env_info.vector_observations         # get next state (for each agent)
            rewards = env_info.rewards                         # get reward (for each agent)
            dones = env_info.local_done                        # see if episode finished
            rewards = [0.1 if rew > 0 else 0 for rew in rewards] # make sure rewards are really 0.1
            scores += rewards

            # save experience in shared replay buffer
            for state, action,reward,next_state,done in zip(states, actions, rewards, next_states, dones):
                shared_replay_buffer.add(state, action,reward,next_state,done)

            # if enough samples are available in memory, every agent learns separately!
            if step%param.UPDATE_EVERY==0 and len(shared_replay_buffer) > agent.param.BATCH_SIZE and param.TRAINMODE:
                for i in range(param.NR_UPDATES):
                    agent.learn(param.GAMMA,shared_replay_buffer)

            states = next_states


            print('\r\t Total score (averaged over agents) episode {} step {}: {}'.format(i_episode,step,np.mean(scores)), end="")
            if any(dones):
                break
            step+=1
        max_score=max(max_score,scores.mean())
        scores_window.append(scores.mean())
        all_scores.append(scores.mean())
        all_average_scores.append(np.mean(scores_window))
        time_for_episode=time.time()-start_time
        print_to_file("Episode "+str(i_episode)+" Time "+str(time_for_episode)+" Score: "+str(scores.mean())+"("+str(scores.mean()/max_score)+" of optimum) max Score: "+str(max_score),filename_scores,False)
        if(param.SAVE):
            agent.save_networks()
        print("")
        print('Episode {}\tTime {}\tScore this episode (averaged over agents): {:.2f}\tAverage Score last 100 episodes (averaged over agents): {:.2f}'.format(i_episode,time_for_episode,scores.mean(), np.mean(scores_window)))


        if np.mean(scores_window)>=param.VAL_ENV_SOLVED:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            if param.PLOT:
                # plot the scores
                fig = plt.figure()
                ax = fig.add_subplot(111)
                plt.plot(np.arange(len(all_scores)), all_scores,label ="score")
                plt.plot(np.arange(len(all_scores)), all_average_scores,label ="average score over last 100 episodes")
                plt.ylabel('Score')
                plt.xlabel('Episode #')
                plt.legend()
                plt.savefig(param.PLOTNAME)
                plt.show()
            break



def print_to_file(text,filename,overwrite):
    original_stdout = sys.stdout
    par='a'
    if overwrite:
        par='w'
    with open(filename, par) as f:
        sys.stdout = f
        print(text)
    sys.stdout = original_stdout
