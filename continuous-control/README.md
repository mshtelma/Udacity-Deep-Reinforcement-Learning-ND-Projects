[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/43851646-d899bf20-9b00-11e8-858c-29b5c2c94ccc.png "Crawler"


# Project 2: Continuous Control - Reacher Environment

## Environment

During this project the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment was solved 


![Trained Agent][image1]

In this environment, a double-jointed arm can move to target locations. 
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

## Getting Started

### Download the environments
This repository already includes environments  for Mac. 
Environments for other operating systems are available using the following links: 
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

Then, place the file in the `Environments/` folder in the this repository, and unzip (or decompress) the file.  

### Install the Anaconda distribution of Python 3
Anaconda for Mac OS is available here: https://www.anaconda.com/download/#macos

### Install the dependencies

- Install Unity ML-Agents (available here: https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md)
- Install all packages from requiremnets.txt
  
## Run instructions

Run training for Reacher environment: 

    python DDPG.py
    
Run trained vector agent: 

     python run_reacher.py
     
## Report
[Report](https://github.com/mshtelma/Udacity-Deep-Reinforcement-Learning-ND-Projects/blob/master/continuous-control/report.md)