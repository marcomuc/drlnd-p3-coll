PATH_TO_ENV = "/home/marco/Dropbox/Data_Science/Udacity/DRLND/deep-reinforcement-learning/p3_collab-compet/Tennis_Linux/Tennis.x86_64"
save_name = "tennis_agent"
from unityagents import UnityEnvironment
import numpy as np
import torch
from train_agent import train_agent, test_agent
from CollaborativeDDPGAgent import CollaborativeDDPGAgent as Agent
import sys

if __name__=="__main__":
    # Initiate env
    env = UnityEnvironment(file_name=PATH_TO_ENV)
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]
    num_agents = len(env_info.agents)
    action_size = brain.vector_action_space_size*2
    states = env_info.vector_observations
    state_size = states.shape[1]*2
    
    # denoise factor
    dnf = 1
    
    agent = Agent(state_size=state_size, action_size=action_size,
                  random_seed=4, actor_units=(200,150), critic_units=(200,150), sigma=.1, batch_size=256, buffer_size=100000*dnf,
                 update_rate = dnf, iters_per_update = 1*dnf)
    
    if len(sys.argv)>=2 and sys.argv[1]=="retrain":
        train_scores = train_agent(agent, env, brain_name, save_name, n_episodes = 3000, max_avg_score = 0.66)
    elif len(sys.argv)>=2 and sys.argv[1]=="continue":
        agent.actor_local.load_state_dict(torch.load(save_name+"_actor.pth"))
        agent.actor_target.load_state_dict(torch.load(save_name+"_actor_target.pth"))
        agent.critic_local.load_state_dict(torch.load(save_name+"_critic.pth"))
        agent.critic_target.load_state_dict(torch.load(save_name+"_critic_target.pth"))
        train_scores = train_agent(agent, env, brain_name, save_name, n_episodes = 1000, max_avg_score = 0.66)
    else:
        agent.actor_local.load_state_dict(torch.load(save_name+"_actor.pth"))
           
    test_scores = test_agent(agent, env, brain_name, save_name, n_episodes = 100, train_mode = True)
    print("Average Test scores: {:.2f}".format(np.mean(test_scores)))

    env.close()

