import pickle
import itertools
from unityagents import UnityEnvironment

from navigation import RunningLearningTask, Result, LearningTask

res = pickle.load(open("res_05.pckl", "rb"))
# print(res)
sorted_res = sorted(res, key=lambda x: x.episode)

for r in sorted_res:
    #if r.mean_score > 13:
    print(r.mean_score, r.task)
