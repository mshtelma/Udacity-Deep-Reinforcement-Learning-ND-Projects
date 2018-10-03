import pickle
import itertools
from unityagents import UnityEnvironment

from navigation import Result, LearningTask, RLAlgorithm




#task = LearningTask(eps_start=1.0, eps_end=0.01, eps_decay=0.99, lr=0.001, buffer_size=100000, batch_size=64,
#                    gamma=0.99, tau=0.01, update_rate=4, agent_type='DQN', layers=[32, 24], network_type='Dueling')

# 200 -> 8.22
#LearningTask(eps_start=1.0, eps_end=0.01, eps_decay=0.99, lr=0.0015, buffer_size=100000, batch_size=64, gamma=0.99,
# tau=0.01, update_rate=4, agent_type='DQN', layers=[24, 12, 8], network_type='Simple')


# 100 -> 3
#task = LearningTask(eps_start=1.0, eps_end=0.01, eps_decay=0.99, lr=0.0015, buffer_size=100000, batch_size=64,
#  gamma=0.99, tau=0.009, update_rate=4, agent_type='DDQN', layers=[32, 24], network_type='Simple')

# 200 -> 9.x
#task = LearningTask(eps_start=1.0, eps_end=0.03, eps_decay=0.85, lr=0.002, buffer_size=100000, batch_size=64,
#                    gamma=0.99, tau=0.02, update_rate=4, agent_type='DDQN', layers=[32, 24], network_type='Dueling')

#task = LearningTask(eps_start=1.0, eps_end=0.03, eps_decay=0.9, lr=0.001, buffer_size=100000, batch_size=64, gamma=0.99,
#             tau=0.03, update_rate=4, agent_type='DDQN', layers=[24, 12, 8], network_type='Dueling')

task =  LearningTask(eps_start=0.25, eps_end=0.001, eps_decay=0.99, lr=0.001, buffer_size=100000, batch_size=64, gamma=0.99,
  tau=0.01, update_rate=1, agent_type='DQN', layers=[32, 24], network_type='Simple')


env = UnityEnvironment(file_name="Environments/Banana.app", no_graphics=False)
dqn = RLAlgorithm(env, max_episodes=900)
print(task)
r = dqn.run_dqn(task)
print(r)
