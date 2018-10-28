# Project 2: Continuous Control - Unity Reacher Environment

## Overview

The goal of this project was to solve Unity Reacher Environment. 
During the project Reacher environment with 20 agents was solved in two different ways using DDPG algorithm. 

## Environment description
In Reacher environment, a double-jointed arm can move to target locations. 
A reward of +0.1 is provided for each step that the agent's hand is in the goal location. 
Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. 
Each action is a vector with four numbers, corresponding to torque applicable to two joints. 
Every entry in the action vector should be a number between -1 and 1.

The barrier for solving the multi-agent version of the environment is slightly different, to take into account the presence of many agents.
In particular, the agents must get an average score of +30 (over 100 consecutive episodes, and over all agents).  
Specifically,
- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent.  This yields 20 (potentially different) scores.  We then take the average of these 20 scores. 
- This yields an **average score** for each episode (where the average is over all 20 agents).

The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30. 

## Deep Deterministic Policy Gradient (DDPG) Algorithm

DDPG uses actor-critic architecture with two elements, actor and critic. 
An actor represents the policy, which decides which action is best to proceed with for a specific state. 
A critic on other hand is used for evaluation of policy function, estimated by the actor using the temporal difference (TD) error.
The pseudocode of the DDPG algorithm is depicted below: 

![image1](https://raw.githubusercontent.com/mshtelma/Udacity-Deep-Reinforcement-Learning-ND-Projects/master/continuous-control/ddpg_algo.png)

Source: https://arxiv.org/pdf/1509.02971.pdf


## Architecture

## DDPG Agent

DDPG agent provided by Udacity was adapted for working with Unity environment with 20 parallel agents. 
Both actor and critic were implemented using deep neural networks. 
Actor network has four fully-connected layers with 33, 128, 48 and 24 neurons respectively. 
The architecture of actor network is depicted on the figure below:


![image2](https://raw.githubusercontent.com/mshtelma/Udacity-Deep-Reinforcement-Learning-ND-Projects/master/continuous-control/actor.png)


Critic network has three fully-connected layers with 33 + 4, 128 and 64 neurons respectively. 
The architecture of critic network is depicted on the figure below:


![image3](https://raw.githubusercontent.com/mshtelma/Udacity-Deep-Reinforcement-Learning-ND-Projects/master/continuous-control/critic.png)


These changes in model architecture produced superior performance compared to Udacity's baseline attempt.

## Hypeparameter

Hyperparameters:

```
BUFFER_SIZE = int(1e5)  
BATCH_SIZE = 128        
GAMMA = 0.99            
TAU = 1e-3            
LR_ACTOR = 1e-3       
LR_CRITIC = 1e-3      
WEIGHT_DECAY = 0      
```

## Results

The Reacher Environment was solved in 107 episodes using DDPG algorithm and hyperparameters mentioned in the previous chapter. 
Below is the chart of score and mean score over 100 last episodes for run of Unity Reacher Environment:


![image1](https://raw.githubusercontent.com/mshtelma/Udacity-Deep-Reinforcement-Learning-ND-Projects/master/navigation/scores.png)



## Future Work
