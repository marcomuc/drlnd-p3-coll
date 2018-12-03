# drlnd-p3-coll
The aim of this repository is to provide a solution to Udacity's [DRLND project 3.](https://github.com/udacity/deep-reinforcement-learning/tree/master/p3_collab-compet) The challenge of this programm is to create an agent which is able to act in the tennis agent. In this environments, two rackets need to bounce a ball over a net. Each of the rackets can be moved toward or away from the net and it can jump, resulting in an action of size 2 for each racket.
Each racket observes the environment through 8 variables representing position and velocity of itself, the other racket, and the ball.

If the ball drops to the floor or leaves the boundary, the responsible racket earns a reward of *-0.01.* If a racket hits a ball over a net, it receives an reward of *+0.1.* The score for a single episode is calculated by taking the max of the sum of the rewards of the individual rackets. The environement is considered solved when the average score of *100* consecutive episodes is larger than *0.5.*

The code of this repository is based on the solution to [Project 2](https://github.com/marcomuc/drlnd-p2-cc) whose code is initially based on the [DDPG pendulum notebook.](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum).

It contains the following files:
* tennis_agent.py - Code to run the agent with the final parameters
* tennis_agent_actor.pth - weights of the actor (online) network (not necessary to run the agent)
* tennis_agent_actor_target.pth - weights of the actor target network
* tennis_agent_critic.pth - weights of the critic (online) network (not necessary to run the agent)
* tennis_agent_critic_target.pth - weights of the critic target network (not necessary to run the agent)
* CollaborativeDDPGAgent.py - Implementation of the DDPG Algorithm
* train_agent.py - Auxillary functions to train and test the agent
* Report.md - Report about the agents performance


# Getting started
The agent itself requires only standard python3, PyTorch and Numpy. 
1. Install [Anaconda](https://www.anaconda.com/download)
2. Clone [Udacitys'DRLND repository(https://github.com/udacity/deep-reinforcement-learning)] and follow the installation instructions.
3. Download the corresponding Unity environment for your system as described [here](https://github.com/udacity/deep-reinforcement-learning/tree/master/p3_collab-compet)
4. Unzip the environment and change the *PATH_TO_ENV* in variable line 1 in tennis_agent.py accordingly.

# Instructions
You have two options:
1. You can run the agent using the provided pre-trained weights and observe how the agents performs over 100 episodes by executing the following command:
*python tennis_agent.py*
2. Alternatively you can retrain the agent yourself. Please note that this overwrites the provided pre-trained weights if you do not modify the code before.
*python tennis_agent.py retrain*
The last option is especially interesting if you want to play with the agent's parameters yourself.
