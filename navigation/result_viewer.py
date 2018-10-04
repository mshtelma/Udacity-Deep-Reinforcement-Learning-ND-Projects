import pickle
import itertools
from unityagents import UnityEnvironment

from navigation import RunningLearningTask, Result, LearningTask

res = pickle.load(open("res_05.pckl", "rb"))
# print(res)
sorted_res = sorted(res, key=lambda x: x.episode)

r = min(res, key=lambda x:x.episode)
print(r)

for r in sorted_res:
    if r.mean_score > 13:
        print(r.episode, r.task)
