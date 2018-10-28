
import numpy as np
import torch
from collections import deque
import matplotlib.pyplot as plt
from unityagents import UnityEnvironment

from ddpg_agent import Agent

env = UnityEnvironment(file_name='Environments/Reacher20.app')
# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=False)[brain_name]
num_agents = len(env_info.agents)
print('number of agents: ',num_agents)
action_size = brain.vector_action_space_size
state_size = env_info.vector_observations.shape[1]
state = env_info.vector_observations
score = np.zeros(num_agents)

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR_ACTOR = 1e-3  # learning rate of the actor
LR_CRITIC = 1e-3  # learning rate of the critic

agent = Agent(state_size=state_size, action_size=action_size, num_agents=num_agents,
              buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE, gamma=GAMMA, tau=TAU, lr_actor=LR_ACTOR,
              lr_critic=LR_CRITIC,
              random_seed=0)
agent.actor_local.load_state_dict(torch.load('reacher_checkpoint_actor_final.pth'))
agent.critic_local.load_state_dict(torch.load('reacher_checkpoint_critic_final.pth'))


while True:
  # select an action using trained policy (epsilon equals 0 now)
  action = agent.act(state)
  # Act!
  env_info = env.step(action)[brain_name]
  next_state = env_info.vector_observations
  reward = env_info.rewards
  done = env_info.local_done
  # update the score
  score += reward
  state = next_state
  if np.any(done):
      break


print("Final Score: {}".format(score))