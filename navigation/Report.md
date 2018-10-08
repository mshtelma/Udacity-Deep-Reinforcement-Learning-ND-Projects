# Project 1: Navigation - Unity Banana Environment

##Overview

The goal of this project was to solve Unity ML Banana Environment. 
During the project this environment was solved in two different ways: once the agent has used  vector observations, the second agent has used visual observations.

##Environment description
A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  
Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding purple bananas.  

The vectorstate space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

The second approach has used visual observations, which were represented as images (84x84).


## DQN Agent

The agent was implemented in both cases as deep neural network. 
For vector observations I have used networks with different number of layers.
One of the best was the network with the two hidden layers, 32 and 24 neurons. 
It was possible to use smaller network with 24 and 6 neurons respectively, but the traning time was longer for such network. 