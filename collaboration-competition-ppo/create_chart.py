import matplotlib
matplotlib.use('Qt5Agg')

from pylab import figure, legend, title, plot, xlabel, ylabel, savefig
import pickle
import numpy as np
import pandas as pd

scores = pickle.load(open("scores.pckl", "rb"))

print(scores)
print(len(scores))

figure(1, figsize=(12, 6))
plot(pd.Series(scores).rolling(window=100, min_periods=1).mean())
title('Mean score for the last 100 episodes')
xlabel('Episode')
ylabel('Mean score')

savefig('mean_scores.png', bbox_inches='tight')

figure(1, figsize=(12, 6))
plot(pd.Series(scores))
title('Scores')
xlabel('Episode')
ylabel('Score')

savefig('scores.png', bbox_inches='tight')
