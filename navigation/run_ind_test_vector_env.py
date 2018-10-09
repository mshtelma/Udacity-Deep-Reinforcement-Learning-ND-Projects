import pickle
import itertools
import pandas as pd
import torch
from unityagents import UnityEnvironment

from navigation import Result, LearningTask
from dqn_agent import DQNAgent

task =  LearningTask(eps_start=0.2, eps_end=0.0001, eps_decay=0.99, lr=0.0008, buffer_size=100000, batch_size=64, gamma=0.99,
  tau=0.0075, update_rate=4, agent_type='DQN', layers=[32, 24], network_type='Simple')

env = UnityEnvironment(file_name="Environments/Banana.app", no_graphics=False)

brain_name = env.brain_names[0]
brain = env.brains[brain_name]
# reset the environment
env_info = env.reset(train_mode=False)[brain_name]
# get the current state
state = env_info.vector_observations[0]
score = 0

agent = DDQNAgent(state_size=len(env_info.vector_observations[0]),
                                   action_size=brain.vector_action_space_size,
                                   buffer_size=task.buffer_size, batch_size=task.batch_size, gamma=task.gamma,
                                   tau=task.tau,
                                   lr=task.lr, update_rate=task.update_rate, layers=task.layers,
                                   network_type=task.network_type, seed=0)
agent.qnetwork_local.load_state_dict(torch.load('vector_banana_checkpoint.pth'))


while True:
  # select an action using trained policy (epsilon equals 0 now)
  action = agent.act(state, 0)
  # Act!
  env_info = env.step(action)[brain_name]
  next_state = env_info.vector_observations[0]
  reward = env_info.rewards[0]
  done = env_info.local_done[0]
  # update the score
  score += reward
  state = next_state
  if done:
    break

print("Final Score: {}".format(score))