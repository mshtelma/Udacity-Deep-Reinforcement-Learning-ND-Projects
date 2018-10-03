import pickle
import numpy as np
from navigation import Result, LearningTask, VisualRLAlgorithm, RunningLearningTask

res = pickle.load(open("data/run5f/scores_final_13.pckl", "rb"))

print(np.mean(res[-100:]))
print(res)