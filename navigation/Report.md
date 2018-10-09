# Project 1: Navigation - Unity Banana Environment

## Overview

The goal of this project was to solve Unity ML Banana Environment. 
During the project this environment was solved in two different ways: once the agent has used  vector observations, the second agent has used visual observations.

## Environment description
A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  
Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding purple bananas.  

The vectorstate space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

The second approach has used visual observations, which were represented as images (84x84).

## Architecture

The main class that can be used for running algorithm is called RLAlgorithm and is stored in navigation.py. 
This class implements simple Q-Learning algorithm applied to sensor data and is compatible with sensor Unity ML environments. 
There is also a class VisualRLAlgorithm, which extends RLAlgorithm with posibility of using visual Unity ML environments. 
It uses four latest frames as a state. 
DQN Agnet code is stored in dqn_agent.py and is represented by DQNAgent class. Most of the abstract logic of DQN agent is developed in abstract Agent class. 
This class serves as a basis for implementation of Double DQN algorithm. 

## DQN Agent

The agent was implemented in both cases (sensor and visual) as deep neural network. 
For vector observations I have used networks with different number of layers.
One of the best was the network with the two hidden layers, 32 and 24 neurons. 
It was possible to use smaller network with 24 and 6 neurons respectively, but the training time was longer for such network. 
The basic DNN is implemented in class QNetwork. There is also abstract implementation, that reused amoing different versions of QNetworks.
Dueling Network algorithm is implemented in DuelingQNetwork. Both classes allow defining number of hidden layers and number of neurons in them. 
For visual agent, convolutional version of QNetwork was implemented. This network has used dueling architecture. 

## Hypeparameter optimisation

During the project a couple of hyperparameter searches runs were tried. One of the best hyperparameters constallation was the follwoing one:
`eps_start=0.2, eps_end=0.0001, eps_decay=0.99, lr=0.0008, buffer_size=100000, batch_size=64, gamma=0.99,
  tau=0.0075, update_rate=4, agent_type='DQN', layers=[32, 24], network_type='Simple'`
Below is the chart of mean score over 100 last episodes for run with these parameters:


![image1] https://raw.githubusercontent.com/mshtelma/Udacity-Deep-Reinforcement-Learning-ND-Projects/master/navigation/best_vector_banana.png "Best Vector Banana"

There is also a comparison between combinations of Q-Learning  extensions: 


![image2] https://raw.githubusercontent.com/mshtelma/Udacity-Deep-Reinforcement-Learning-ND-Projects/master/navigation/algo_comparison_scores.png "Comparison"


