# Project 1: Navigation - Unity Banana Collector Environment

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
The basic DNN is implemented in class QNetwork. There is also abstract implementation, that reused among different versions of QNetworks.
Here is the architecture of simple QNetwork:
```
QNetwork(
  (seq): Sequential(
    (l0): Linear(in_features=37, out_features=32, bias=True)
    (r0): ReLU()
    (l1): Linear(in_features=32, out_features=24, bias=True)
    (r1): ReLU()
    (ll): Linear(in_features=24, out_features=4, bias=True)
  )
)
```
Dueling Network algorithm is implemented in DuelingQNetwork. Both classes allow defining number of hidden layers and number of neurons in them. 
Dueling QNetwork for vector environment has the following architecture:
```
DuelingQNetwork(
  (seq): Sequential(
    (l0): Linear(in_features=37, out_features=32, bias=True)
    (r0): ReLU()
    (l1): Linear(in_features=32, out_features=24, bias=True)
    (r1): ReLU()
    (ll): Linear(in_features=24, out_features=4, bias=True)
  )
  (adv1): Linear(in_features=4, out_features=8, bias=True)
  (adv2): Linear(in_features=8, out_features=4, bias=True)
  (val1): Linear(in_features=4, out_features=8, bias=True)
  (val2): Linear(in_features=8, out_features=1, bias=True)
)
```
For visual agent, convolutional version of QNetwork was implemented. 
This network has used dueling architecture:
```
ConvQNetwork(
  (conv_a1): Conv2d(12, 12, kernel_size=(4, 4), stride=(1, 1), padding=(2, 2))
  (bn_a1): BatchNorm2d(12, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (conv_a2): Conv2d(12, 24, kernel_size=(4, 4), stride=(1, 1), padding=(2, 2))
  (bn_a2): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (conv_a3): Conv2d(24, 48, kernel_size=(4, 4), stride=(1, 1), padding=(2, 2))
  (bn_a3): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (conv_a4): Conv2d(48, 96, kernel_size=(4, 4), stride=(1, 1), padding=(2, 2))
  (bn_a4): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (conv_b1): Conv2d(12, 24, kernel_size=(3, 3), stride=(3, 3))
  (bn_b1): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (conv_b2): Conv2d(24, 48, kernel_size=(3, 3), stride=(3, 3))
  (bn_b2): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (conv_b3): Conv2d(48, 96, kernel_size=(3, 3), stride=(3, 3))
  (bn_b3): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (conv_b4): Conv2d(96, 24, kernel_size=(3, 3), stride=(1, 1))
  (bn_b4): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (conv_c1): Conv2d(12, 24, kernel_size=(7, 2), stride=(2, 2))
  (bn_c1): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (conv_c2): Conv2d(24, 48, kernel_size=(7, 2), stride=(2, 2))
  (bn_c2): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (conv_c3): Conv2d(48, 96, kernel_size=(7, 2), stride=(2, 2))
  (bn_c3): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (conv_c4): Conv2d(96, 24, kernel_size=(4, 4), stride=(1, 1))
  (bn_c4): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (conv_d1): Conv2d(12, 24, kernel_size=(2, 7), stride=(2, 2))
  (bn_d1): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (conv_d2): Conv2d(24, 48, kernel_size=(2, 7), stride=(2, 2))
  (bn_d2): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (conv_d3): Conv2d(48, 96, kernel_size=(2, 7), stride=(2, 2))
  (bn_d3): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (conv_d4): Conv2d(96, 24, kernel_size=(4, 4), stride=(1, 1))
  (bn_d4): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (head1): Linear(in_features=1896, out_features=1024, bias=True)
  (head2): Linear(in_features=1024, out_features=24, bias=True)
  (adv1): Linear(in_features=24, out_features=12, bias=True)
  (adv2): Linear(in_features=12, out_features=4, bias=True)
  (val1): Linear(in_features=24, out_features=12, bias=True)
  (val2): Linear(in_features=12, out_features=1, bias=True)
)
```


## Hypeparameter optimisation

During the project a couple of hyperparameter searches were tried. 
The code for the searches is available in python files (test01.py - test05.py)
One of the best hyperparameters constellations was the following one:
`eps_start=0.2, eps_end=0.0001, eps_decay=0.99, lr=0.0008, buffer_size=100000, batch_size=64, gamma=0.99,
  tau=0.0075, update_rate=4, agent_type='DQN', layers=[32, 24], network_type='Simple'`

## Results

### Vector Banana Environment

The vector environment can be solved in 137 episodes: 
```
LearningTask(eps_start=0.2, eps_end=0.0001, eps_decay=0.99, lr=0.0008, buffer_size=100000, batch_size=64, gamma=0.99,
 tau=0.0075, update_rate=4, agent_type='DQN', layers=[32, 24], network_type='Simple')
Episode 100     Average Score: 8.273
Episode 200     Average Score: 12.46
Episode 237     Average Score: 13.00
Environment solved in 137 episodes!     Average Score: 13.00
```
Below is the chart of mean score over 100 last episodes for run of Unity Banana Environment with vector state (using hyperparameters mentioned above):


![image1](https://raw.githubusercontent.com/mshtelma/Udacity-Deep-Reinforcement-Learning-ND-Projects/master/navigation/best_vector_banana.png)

The comparison between different flavors of DQN algorithm was also conducted:

![image2](https://raw.githubusercontent.com/mshtelma/Udacity-Deep-Reinforcement-Learning-ND-Projects/master/navigation/algo_comparison_scores.png)

It has shown, that for so small and easy vector state environment, it does not bring any real benefit to use Double DQN or Dueling networks, but these both technics played crucial role while training visual Banana environment. 



### Visual Banana Environment

Visual environment was solved in 1400 episodes. 
(Detailed history of scores duing the training is available in file navigation/visual_banana_nohup.out)
Below is the chart of mean score over 100 last episodes for run of Unity Banana Environment with visual state :


![image2](https://raw.githubusercontent.com/mshtelma/Udacity-Deep-Reinforcement-Learning-ND-Projects/master/navigation/visual_banana_scores.png)


## Future Work
Rainbow paper (https://arxiv.org/pdf/1710.02298.pdf) suggests combining many of already known improvements to Q-Learning into one reinforcement learning algorithm.
This approach is known as Rainbow algorithm, which was already implemented as part of another project released by Google, which is called Dopamine. 
The next steps should include using this framework and Rainbow algorithm!