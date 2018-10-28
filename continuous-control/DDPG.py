import sys

import numpy as np
import torch
from collections import deque
import matplotlib.pyplot as plt
import pickle
from unityagents import UnityEnvironment

from ddpg_agent import Agent

env = UnityEnvironment(file_name='Environments/Reacher20.app')
# env = UnityEnvironment(file_name='Environments/Reacher_Linux_NoVis/Reacher.x86_64')

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

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


def ddpg(n_episodes=50000, print_every=100):
    scores_window = deque(maxlen=print_every)

    scores = []
    mean_scores = []

    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations
        agent.reset()

        current_scores = np.zeros(num_agents)

        while True:
            action = agent.act(state)

            # next_state, reward, done, _ = env.step(action)
            env_info = env.step(action)[brain_name]  # send the action to the environment
            reward = env_info.rewards  # get the reward
            done = env_info.local_done
            next_state = env_info.vector_observations

            agent.step(state, action, reward, next_state, done)
            state = next_state
            current_scores += reward
            if np.any(done):  # exit loop if episode finished
                break

        current_score = np.mean(current_scores)
        scores_window.append(current_score)
        mean = np.mean(scores_window)
        scores.append(current_score)
        mean_scores.append(mean)
        if mean >= 30.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, mean))
            torch.save(agent.actor_local.state_dict(), 'reacher_checkpoint_actor_final.pth')
            torch.save(agent.critic_local.state_dict(), 'reacher_checkpoint_critic_final.pth')
            break
        print('\rEpisode {}\tAverage Score: {:.2f}\tCurrent Score: {:.2f}'.format(i_episode, mean, current_score))
        sys.stdout.flush()

        # if i_episode % print_every == 0:
        #    print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, mean))

    torch.save(agent.actor_local.state_dict(), 'reacher_checkpoint_actor.pth')
    torch.save(agent.critic_local.state_dict(), 'reacher_checkpoint_critic.pth')

    return scores, mean_scores


scores, mean_scores = ddpg()

f = open('scores.pckl', 'wb')
pickle.dump((scores, mean_scores), f)
f.flush()
f.close()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores) + 1), scores)
plt.plot(np.arange(1, len(mean_scores) + 1), mean_scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig('scores.png', bbox_inches='tight')
