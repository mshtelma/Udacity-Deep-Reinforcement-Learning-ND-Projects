import pickle
import itertools
from unityagents import UnityEnvironment

from navigation import LearningTask, DQN

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 0.0075  # for soft update of target parameters
LR = 0.001  # learning rate
UPDATE_EVERY = 4  # how often to update the network

eps_starts = [1.0]
eps_decays = [0.99]
eps_ends = [0.01]
lrs = [0.001, 0.0015]
buffers = [BUFFER_SIZE]
batches = [64]
gammas = [GAMMA]
taus = [  0.009, 0.01, 0.02]
update_rates = [UPDATE_EVERY]
agent_types = [ 'DQN', 'DDQN']  # Add   ,
layers = [  [24, 12, 8], [32, 24]]
network_types = ['Simple', 'Dueling'] #,


# LearningTask(eps_start=1.0, eps_end=0.01, eps_decay=0.99, lr=0.001, buffer_size=100000, batch_size=64, gamma=0.99, tau=0.01, update_rate=4, agent_type='DQN', layers=[32, 24], network_type='Dueling')
env = UnityEnvironment(file_name="/Users/ms250139/Sources/courses/DRLND/UnityAgents/Banana2.app", no_graphics=True)

tasks = list(itertools.product(
    *[eps_starts, eps_ends, eps_decays, lrs, buffers, batches, gammas, taus, update_rates, agent_types, layers, network_types]))
res = []
dqn = DQN(env, max_episodes=200)
for task_spec in tasks:

    task = LearningTask._make(task_spec)
    print(task)
    r = dqn.run_dqn(task)
    print(r)
    res.append(r)
    f = open('res_03.pckl', 'wb')
    pickle.dump(res, f)
    f.flush()
    f.close()
