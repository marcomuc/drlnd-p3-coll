import numpy as np
from collections import deque
import torch
from datetime import datetime

def train_agent(agent, env, brain_name, save_name = 'checkpoint', n_episodes=400, max_t=1001, scores_intervall= 100, max_avg_score = 0.66):
    """Trains a DDPGagent within an Unity Environment.
    Finishes when a specified score is achieved or when the numter of training episodes is exceeded
        
    Params
    =====
        agent (DDPGAgent): agent to be trained
        env (): environment
        brain_name (): agent to be trained
        save_name (str): name of the file where the weights are stored after training
        n_episodes (int): Max number of training episodes
        max_t (int): Max number of time steps within each episode before the episode is interrupted
        scores_intervall (int): Number of episodes for which the average score is calculated
        max_avg_score (float): Stops training when the average score over the last scores_intervall episodes is larger than that value.

    Returns
    ======
        scores ([floats]): list containing the score achieved during every episode of the training

    """
    print("Training agent", save_name)
    scores_deque = deque(maxlen=scores_intervall)
    scores = []
    
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations#[0]
        agent.reset()
        score = np.zeros(len(env_info.agents))
        t = 0
        episode_length = 0
        for t in range(max_t):
            t += 1
            episode_length = t
            action = agent.act(state)                           # select an action (for each agent)
            env_info = env.step(action)[brain_name]             # send all actions to tne environment
            next_state = env_info.vector_observations           # get next state (for each agent)
            reward = env_info.rewards                           # get reward (for each agent)
            done = env_info.local_done                          # see if episode finished
            agent.step(state, action, reward, next_state, done) # let the agent learn from experiences
            state = next_state                                  # roll over states to next time step
            score += reward 
            if np.any(done):                                    # exit loop if episode finished
                break 
        scores_deque.append(score.max())
        scores.append(score.max())
        print('Episode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(i_episode, np.mean(scores_deque), np.mean(score)), datetime.now(),episode_length, end="\n")
        
        if np.mean(scores_deque)>=max_avg_score:
            print("Max avg score achieced. Stopping training.")
            break
        
        
        
    torch.save(agent.actor_local.state_dict(), save_name+'_actor.pth')
    torch.save(agent.critic_local.state_dict(), save_name+'_critic.pth')
    torch.save(agent.actor_target.state_dict(), save_name+'_actor_target.pth')
    torch.save(agent.critic_target.state_dict(), save_name+'_critic_target.pth')
    
    np.save(save_name+"_train_results.npy", np.array(scores))
  
    return scores

def test_agent(agent, env, brain_name, save_name = "checkpoint", n_episodes = 100, max_t=1001, train_mode = True):
    """Tests an agent within an Unity Environment.
            
    Params
    =====
        agent (): agent to be tested
        env (): environment
        brain_name (): agent to be tested
        N_episodes (int): Max number of testing episodes
        max_t (int): Max number of time steps within each episode before the episode is interrupted


    Returns
    ======
        scores ([floats]): list containing the score achieved during every episode of the test

    """
    print("Start with testing agent", datetime.now())
    scores = []

    for i_episode in range(n_episodes):
        env_info = env.reset(train_mode=train_mode)[brain_name]  # reset the environment    
        state = env_info.vector_observations                  # get the current state (for each agent)
        score = np.zeros(len(env_info.agents))                # initialize the score (for each agent)
        for t in range(max_t):
            action = agent.act(state, train_mode)             # select an action (for each agent)
            action = np.clip(action, -1, 1)                   # all actions between -1 and 1
            env_info = env.step(action)[brain_name]           # send all actions to tne environment
            next_state = env_info.vector_observations         # get next state (for each agent)
            reward = env_info.rewards                         # get reward (for each agent)
            done = env_info.local_done                        # see if episode finished
            score += env_info.rewards                         # update the score (for each agent)
            state = next_state                                # roll over states to next time step
            if np.any(done):                                  # exit loop if episode finished
                break

        scores.append(score.max())
        print('Episode {}\tScore: {:.2f}'.format(i_episode, np.mean(score)), datetime.now(), end="\n")

    print('Total score (averaged over agents) over 100 episodes: {}'.format(np.mean(scores)))
    if save_name!=None:
        np.save(save_name+"_test_results.npy", np.array(scores))
    return scores
