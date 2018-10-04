import pickle
import itertools
from unityagents import UnityEnvironment

from navigation import Result, LearningTask, RLAlgorithm

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 0.0075  # for soft update of target parameters
LR = 0.001  # learning rate
UPDATE_EVERY = 4  # how often to update the network

eps_starts = [ 0.25]
eps_decays = [0.99]
eps_ends = [0.001]
lrs = [0.0005 ]
buffers = [BUFFER_SIZE]
batches = [64]
gammas = [GAMMA]
taus = [0.004]
update_rates = [4]
agent_types = ['DQN','DDQN' ]  #
layers = [ [32, 24]] #[16, 12, 6], [24, 12, 8], [32, 8], [24, 6]
network_types = ['Simple','Dueling' ]  #

# LearningTask(eps_start=1.0, eps_end=0.01, eps_decay=0.99, lr=0.001, buffer_size=100000, batch_size=64, gamma=0.99, tau=0.01, update_rate=4, agent_type='DQN', layers=[32, 24], network_type='Dueling')
env = UnityEnvironment(file_name="Environments/Banana.app", no_graphics=False)

tasks = list(itertools.product(
    *[eps_starts, eps_ends, eps_decays, lrs, buffers, batches, gammas, taus, update_rates, agent_types, layers,
      network_types]))
res = []

for task_spec in tasks:
    dqn = RLAlgorithm(env, max_episodes=1200)
    task = LearningTask._make(task_spec)
    print(task)
    r = dqn.run_dqn(task)
    print('episode = ', r.episode, 'mean score = ', r.mean_score)
    res.append(r)
f = open('res_compare.pckl', 'wb')
pickle.dump(res, f)
f.flush()
f.close()
