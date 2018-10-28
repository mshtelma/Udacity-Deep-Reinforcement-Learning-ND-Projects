# Project 1: Navigation - Unity Banana Collector Environment

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

During this project I have trained an agent to navigate though two types of bananas in square world, collects yellow bananas and omits the purple ones.  

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

## Getting started

### Download the environments
This repository already includes environment  for Mac. Environments for other operating systems are available using the following links: 
  - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
  - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
  - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
  - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
  
  
### Install the Anaconda distribution of Python 3
Anaconda for Mac OS is available here: https://www.anaconda.com/download/#macos

### Install the dependencies

Install Unity ML-Agents (available here: https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md)
Install all packages from requiremnets.txt
  
## Run instructions

Run training for vector environment: 

    python train_vector_ind.py

Run training for visual environment: 

    python train_pixels_ind.py
    
Run trained vector agent: 

     python run_ind_test_vector_env.py
     
## Report
[Report](https://github.com/mshtelma/Udacity-Deep-Reinforcement-Learning-ND-Projects/blob/master/navigation/Report.md)