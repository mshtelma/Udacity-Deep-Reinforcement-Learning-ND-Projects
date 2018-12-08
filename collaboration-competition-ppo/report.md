# Project 3: Collaboration and Competition - Tennis Environment

## Overview

The goal of this project was to solve Unity Tennis Environment. 
During the project Tennis environment with two agents was solved using PPO algorithm. 

## Environment description
In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

## Proximal Policy Optimization (PPO) Algorithm

I have used Proximal Policy Optimization (PPO) algorithm during this project, which was first introduced by Schulman et al in 
 [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
 It has a good balance between sample complexity and implementation simplicity.

The pseudocode of the PPO algorithm is depicted below: 

![image1](https://raw.githubusercontent.com/mshtelma/Udacity-Deep-Reinforcement-Learning-ND-Projects/master/collaboration-competition-ppo/ppo.png)


## Implementation

During this project, the **DeepRL** Deep Reinforcement Learning Framework, developed by [Shangtong Zhang](https://shangtongzhang.github.io/) was used. 
More information about this framework is available on its GitHub page:  [github](https://github.com/ShangtongZhang/DeepRL)

**DeepRL** allows us to define a Task, which can be afterwards optimised using different algorithms. 
It is also possible to configure various optimisation hyperparameters. 


## Model

The following model was used : 
    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
             DummyBody-1                [-1, 1, 48]               0
                Linear-2                [-1, 1, 32]           1,568
                Linear-3                [-1, 1, 32]           1,056
                FCBody-4                [-1, 1, 32]               0
                Linear-5                [-1, 1, 32]           1,568
                Linear-6                [-1, 1, 32]           1,056
                FCBody-7                [-1, 1, 32]               0
                Linear-8                 [-1, 1, 4]             132
                Linear-9                 [-1, 1, 1]              33
    ================================================================
    Total params: 5,413
    Trainable params: 5,413
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.00
    Forward/backward pass size (MB): 0.00
    Params size (MB): 0.02
    Estimated Total Size (MB): 0.02
    ----------------------------------------------------------------



## Hypeparameter

The following hyperparameters were used:

```python
learning_rate = 1e-3

ppoCfg.discount = 0.99

ppoCfg.ppo_ratio_clip = 0.5
ppoCfg.gradient_clip = 3

ppoCfg.num_mini_batches = 128
ppoCfg.rollout_length = 4096
ppoCfg.max_steps = 1e9

ppoCfg.gae_tau = 0.9
ppoCfg.use_gae = True     
```

## Results

The Tennis Environment was solved in 5426 episodes using PPO algorithm and hyperparameters mentioned in the previous chapter. 
Below is the chart of  mean score over 100 last episodes for the training of Unity Tennis Environment:


![image2](https://raw.githubusercontent.com/mshtelma/Udacity-Deep-Reinforcement-Learning-ND-Projects/master/collaboration-competition-ppo/mean_scores.png)

Below is the chart of score for the training of Unity Tennis Environment:

![image3](https://raw.githubusercontent.com/mshtelma/Udacity-Deep-Reinforcement-Learning-ND-Projects/master/collaboration-competition-ppo/scores.png)



## Future Work
I think, that MADDPG algorithm can show even better results then PPO