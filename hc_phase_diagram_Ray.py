import numpy as np
import pandas as pd
from evaluate_iteration import *
import ray

lr = 17
lb = 19

rr=np.concatenate([np.array([0.0]), np.linspace(0.1, 3, lr)]) 
bb=np.linspace(0.45, 0.99, lb)

aa=[0.8, 1.2]
xx=[0, 0.2, 0.5]
nn = [1e4]

nMonte = 1000

df = pd.DataFrame()

ray.init(address='auto')

n = nn[0]
res = [evaluate_iteration.remote(a, x, b, r, n, nMonte = nMonte,
								 metric = 'power') 
		for a in aa for x in xx for b in bb for r in rr for n in nn
		]
return_value = ray.get(res)
df = pd.DataFrame()
for r in return_value :
    df = df.append(r, ignore_index=True)
df.to_csv('results.csv')
