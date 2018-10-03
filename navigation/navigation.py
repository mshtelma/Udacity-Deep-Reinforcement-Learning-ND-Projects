import sys

import torch
import torchvision.transforms as T
from PIL import Image

import numpy as np
from collections import namedtuple, deque

from dqn_agent import DQNAgent
from dqn_agent import DDQNAgent

LearningTask = namedtuple('LearningTask', ['eps_start', 'eps_end', 'eps_decay', 'lr', 'buffer_size', 'batch_size',
                                           'gamma', 'tau', 'update_rate', 'agent_type', 'layers', 'network_type'])

Result = namedtuple('Result', ['mean_score', 'episode', 'task', 'scores'])

RunningLearningTask = namedtuple('RunningLearningTask',
                                 ['memory', 'last_episode', 'scores', 'scores_window', 'epsilon', 'local_weights',
                                  'target_weights'])


class RLAlgorithm():
    def __init__(self, env, max_episodes=300):
        self.scores = []  # list containing scores from each episode
        self.scores_window = deque(maxlen=100)  # last 100 scores

        self.env = env

        self.max_episodes = max_episodes

        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]

        # reset the environment
        self.env_info = self.env.reset(train_mode=True)[self.brain_name]

        # number of agents in the environment
        print('Number of agents:', len(self.env_info.agents))

        # number of actions
        self.action_size = self.brain.vector_action_space_size
        print('Number of actions:', self.action_size)

        # examine the state space
        self.state = self.env_info.vector_observations[0]
        print('States look like:', self.state)
        self.state_size = len(self.state)
        print('States have length:', self.state_size)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def run_dqn(self, task):
        self.env_info = self.env.reset(train_mode=True)[self.brain_name]

        agent = None
        if task.agent_type == 'DQN':
            self.agent = DQNAgent(state_size=len(self.env_info.vector_observations[0]),
                                  action_size=self.brain.vector_action_space_size,
                                  buffer_size=task.buffer_size, batch_size=task.batch_size, gamma=task.gamma,
                                  tau=task.tau,
                                  lr=task.lr, update_rate=task.update_rate, layers=task.layers,
                                  network_type=task.network_type, seed=0)
        elif task.agent_type == 'DDQN':
            self.agent = DDQNAgent(state_size=len(self.env_info.vector_observations[0]),
                                   action_size=self.brain.vector_action_space_size,
                                   buffer_size=task.buffer_size, batch_size=task.batch_size, gamma=task.gamma,
                                   tau=task.tau,
                                   lr=task.lr, update_rate=task.update_rate, layers=task.layers,
                                   network_type=task.network_type, seed=0)
        mean_score, episode, scores, _, _ = self.dqn(task, self.agent, n_episodes=self.max_episodes,
                                                     eps_start=task.eps_start,
                                                     eps_end=task.eps_end,
                                                     eps_decay=task.eps_decay)
        return Result(task=task, mean_score=mean_score, episode=episode, scores=scores)

    def dqn(self, task, agent, start_episode=1, n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.03,
            eps_decay=0.85):
        """Deep Q-Learning.

        Params
        ======
            n_episodes (int): maximum number of training episodes
            max_t (int): maximum number of timesteps per episode
            eps_start (float): starting value of epsilon, for epsilon-greedy action selection
            eps_end (float): minimum value of epsilon
            eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
        """

        eps = eps_start  # initialize epsilon

        for i_episode in range(start_episode, n_episodes + 1):
            env_info = self.env.reset(train_mode=True)[self.brain_name]
            state = deque(maxlen=4)
            state.append(self.decode_state(env_info))
            state.append(self.decode_state(env_info))
            state.append(self.decode_state(env_info))
            state.append(self.decode_state(env_info))
            score = 0
            for t in range(max_t):
                action = agent.act(state, eps)

                # next_state, reward, done, _ = env.step(action)

                env_info = self.env.step(action)[self.brain_name]  # send the action to the environment
                reward = env_info.rewards[0]  # get the reward
                done = env_info.local_done[0]

                next_state = state.copy()  # get the next state
                next_state.append(self.decode_state(env_info))  # get the next state

                # if not done:
                #    next_state = next_state - state
                # else:
                #    next_state = None

                agent.step(state, action, reward, next_state, done)

                state = next_state
                score += reward

                if done:
                    break
            self.scores_window.append(score)  # save most recent score
            self.scores.append(score)  # save most recent score
            eps = max(eps_end, eps_decay * eps)  # decrease epsilon

            mean = np.mean(self.scores_window)

            # if mean > 11:
            #     agent.set_lr(task.lr * 0.8)
            # elif mean > 7:
            #     agent.set_lr(task.lr * 0.99)

            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, mean), end="")
            #print('\rEpisode {}\tAverage Score: {:.4f}'.format(i_episode, mean))
            if i_episode % 10 == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, mean))
                sys.stdout.flush()
            if i_episode % 100 == 0:
                torch.save(agent.qnetwork_local.state_dict(), 'checkpoint_'+str(i_episode)+'.pth')
            if mean >= 13.0:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                             mean))
                torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
                break
        return mean, i_episode - 100, self.scores, self.scores_window, eps

    def decode_state(self, env_info):
        return env_info.vector_observations[0]


class VisualRLAlgorithm(RLAlgorithm):
    def __init__(self, env, rtask, max_episodes=300):
        self.rtask = rtask
        super(VisualRLAlgorithm, self).__init__(env, max_episodes)

        self.scores = rtask.scores
        self.scores_window = rtask.scores_window

        self.resize = T.Compose([T.ToPILImage(),
                                 #T.Resize(40, interpolation=Image.CUBIC),
                                 T.ToTensor()])

        self.state = self.decode_state(self.env_info)
        print('Pixel states have length:', self.state.shape)

    def run_dqn(self, task):
        self.env_info = self.env.reset(train_mode=True)[self.brain_name]

        agent = None
        if task.agent_type == 'DQN':
            self.agent = DQNAgent(state_size=len(self.env_info.vector_observations[0]),
                                  action_size=self.brain.vector_action_space_size,
                                  buffer_size=task.buffer_size, batch_size=task.batch_size, gamma=task.gamma,
                                  tau=task.tau,
                                  lr=task.lr, update_rate=task.update_rate, layers=task.layers,
                                  network_type=task.network_type, seed=0)
        elif task.agent_type == 'DDQN':
            self.agent = DDQNAgent(state_size=len(self.env_info.vector_observations[0]),
                                   action_size=self.brain.vector_action_space_size,
                                   buffer_size=task.buffer_size, batch_size=task.batch_size, gamma=task.gamma,
                                   tau=task.tau,
                                   lr=task.lr, update_rate=task.update_rate, layers=task.layers,
                                   network_type=task.network_type, seed=0)

        if self.rtask.local_weights is not None:
            self.agent.qnetwork_local.load_state_dict(self.rtask.local_weights)
        if self.rtask.target_weights is not None:
            self.agent.qnetwork_target.load_state_dict(self.rtask.target_weights)

        #self.agent.memory = self.rtask.memory
        mean, i_episode, scores, scores_window, eps = super(VisualRLAlgorithm, self).dqn(task, self.agent,
                                                                                         start_episode=0,
                                                                                         n_episodes=self.max_episodes,
                                                                                         eps_start=self.rtask.epsilon,
                                                                                         eps_end=task.eps_end,
                                                                                         eps_decay=task.eps_decay
                                                                                         )
        global_episode = self.rtask.last_episode + i_episode + 100

        print('\n')
        print('Finished process with eps=', eps, ' episode=', i_episode+100, ' mean score = ', mean)
        print('Global episode number is ', global_episode)
        new_task = RunningLearningTask(None, global_episode, scores, scores_window, eps,
                                       self.agent.qnetwork_local.state_dict(), self.agent.qnetwork_target.state_dict())
        return new_task

    def decode_state(self, env_info):
        # transpose into torch order (CHW)
        screen = env_info.visual_observations[0].squeeze().transpose((2,0,1))
        screen = np.ascontiguousarray(screen, dtype=np.float32) #/ 255
        screen = torch.from_numpy(screen)
        # Resize, and add a batch dimension (BCHW)
        return self.resize(screen).unsqueeze(0).to(self.device)
