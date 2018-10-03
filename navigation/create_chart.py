import matplotlib
matplotlib.use('Qt5Agg')

from pylab import figure, axes, pie, title, show, plot, xlabel, ylabel, savefig
import pickle
import numpy as np
import pandas as pd

res = pickle.load(open("visual_banana_scores.pckl", "rb"))

print(np.mean(res[-100:]))
print(res)

ts = pd.Series(res)


figure(1, figsize=(12, 6))
plot(ts.rolling(window=100, min_periods=1).mean())
title('Rewards')
xlabel('Episode')
ylabel('Score')

savefig('visual_banana_scores.png', bbox_inches='tight')