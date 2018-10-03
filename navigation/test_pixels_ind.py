import pickle
from collections import deque
from pathlib import Path

import itertools
from multiprocessing import Process
import multiprocessing as mp

import torch
from unityagents import UnityEnvironment

from navigation import Result, LearningTask, VisualRLAlgorithm, RunningLearningTask
from dqn_agent import ReplayBuffer


def learn(task, rtask):
    #file = Path("last_task_pixels.pckl")
    #if file.is_file():
    #    rtask = pickle.load(open('last_task_pixels.pckl', 'rb'))
    print('Starting new process with global episode number ', rtask.last_episode)
    env = UnityEnvironment(file_name="VisualBanana_Linux/Banana.x86_64", no_graphics=False)
    #env = UnityEnvironment(file_name="/Users/ms250139/Sources/courses/DRLND/UnityAgents/VisualBanana.app", no_graphics=False)
    algo = VisualRLAlgorithm(env, rtask, max_episodes=2500)
    new_task = algo.run_dqn(task)
    env.close()
    f = open('last_task_pixels.pckl', 'wb')
    pickle.dump(new_task, f)
    f.close()
    print('Closed process')


if __name__ == '__main__':
    mp.set_start_method('spawn')

    task = LearningTask(eps_start=1.0, eps_end=0.01, eps_decay=0.99, lr=0.0001, buffer_size=100000, batch_size=128,
                        gamma=0.99, tau=0.0005, update_rate=4, agent_type='DDQN', layers=[32, 24],
                        network_type='SimpleConvolution')

    #checkpoint = torch.load('run3/checkpoint_2000.pth')
    memory = ReplayBuffer(4, task.buffer_size, task.batch_size, 0)
    rtask = RunningLearningTask(memory, 0, [], deque(maxlen=100), task.eps_start, None, None)
    for i in range(1, 50):
        p = Process(target=learn, args=(task, rtask))
        p.start()
        p.join()

    # print(r)
    #torch.save(algo.agent.qnetwork_local.state_dict(), 'test_pixels_ind.pth')
