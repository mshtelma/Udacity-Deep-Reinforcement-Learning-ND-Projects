from tensorboard import summary
from unityagents import UnityEnvironment
import numpy as np
import pickle
from DeepRL.deep_rl import *

env = UnityEnvironment(file_name="Environments/Tennis.app")

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

scores_window = deque(maxlen=100)
scores_all = []


class TennisTask(Task):
    def __init__(self):
        self.name = 'Unity-Tennis-Env'
        self.env = env
        self.num_agents = num_agents
        self.state_dim = state_size * self.num_agents
        self.action_dim = action_size * self.num_agents

        self.episode_reward = 0.0
        self.episode_steps = 0

    def step(self, action):
        env_info = self.env.step(np.clip(action, -1, 1))[brain_name]
        state = env_info.vector_observations.reshape(1, -1)
        step_reward = np.sum(env_info.rewards)
        done = np.any(env_info.local_done)

        self.episode_reward += step_reward
        self.episode_steps += 1

        if done:
            scores_window.append(self.episode_reward)
            scores_all.append(self.episode_reward)
            state = self.reset()

        return state, np.array([step_reward]), np.array([done]), None

    def reset(self):
        self.episode_reward = 0.0
        self.episode_steps = 0
        env_info = self.env.reset(train_mode=True)[brain_name]
        return env_info.vector_observations.reshape(1, -1)


ppoCfg = Config()
tennisTask = TennisTask()

ppoCfg.task_fn = lambda: tennisTask
ppoCfg.eval_env = tennisTask

network = GaussianActorCriticNet(state_size * num_agents, action_size * num_agents,
                                 actor_body=FCBody(state_size * num_agents, hidden_units=(32, 32)),
                                 critic_body=FCBody(ppoCfg.state_dim, hidden_units=(32, 32))
                                 )
ppoCfg.network_fn = lambda: network
ppoCfg.optimizer_fn = lambda params: torch.optim.Adam(params, 1e-3)

ppoCfg.discount = 0.99

ppoCfg.ppo_ratio_clip = 0.5
ppoCfg.gradient_clip = 3

ppoCfg.num_mini_batches = 128
ppoCfg.rollout_length = 4096
ppoCfg.max_steps = 1e9

ppoCfg.gae_tau = 0.9
ppoCfg.use_gae = True

agent = PPOAgent(ppoCfg)

from torchsummary import summary

summary(agent.network, input_size=(1, num_agents * state_size))

while True:
    agent.step()
    episode = len(scores_all)
    mean = np.mean(scores_window)
    print('Episode: %d,  mean reward: %.4f ' % (episode, mean))
    if mean > 1:
        print('Environment solved in %d episodes' % episode)
        break

agent.save('tennis_final.pth')
f = open('scores.pckl', 'wb')
pickle.dump(scores_all, f)
f.flush()
f.close()
