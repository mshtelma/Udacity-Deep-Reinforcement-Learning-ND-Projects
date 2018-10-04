import matplotlib
matplotlib.use('Qt5Agg')

from pylab import figure, legend, title, plot, xlabel, ylabel, savefig
import pickle
import numpy as np
import pandas as pd

res = pickle.load(open("res_05.pckl", "rb"))
r = min(res, key=lambda x:x.episode)
print(r.scores)
print(len(r.scores))
figure(1, figsize=(12, 6))
plot(pd.Series(r.scores).rolling(window=100, min_periods=1).mean())
title('Mean score for the last 100 episodes')
xlabel('Episode')
ylabel('Mean score')

savefig('best_vector_banana.png', bbox_inches='tight')

exit(0)

res = pickle.load(open("res_compare.pckl", "rb"))
df = pd.concat([pd.Series(r.scores,
                          name=r.task.network_type + ' ' + r.task.agent_type).rolling(window=100, min_periods=1).mean()
                for r in res], axis=1)

figure(1, figsize=(12, 6))

plot(df)
title('Mean score for the last 100 episodes')
xlabel('Episode')
ylabel('Mean score')
legend(df.columns, loc='lower right')

savefig('algo_comparison_scores.png', bbox_inches='tight')




res = pickle.load(open("visual_banana_scores.pckl", "rb"))

print(np.mean(res[-100:]))
print(res)

figure(1, figsize=(12, 6))
plot(pd.Series(res).rolling(window=100, min_periods=1).mean())
title('Mean score for the last 100 episodes')
xlabel('Episode')
ylabel('Mean score')

savefig('visual_banana_scores.png', bbox_inches='tight')
