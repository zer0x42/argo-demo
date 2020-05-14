import pandas as pd
from sklearn.datasets import make_blobs

df = pd.DataFrame.read_csv('argo-demo/lifecycle-management/data/data.csv')
X, y = make_blobs(n_samples=100, centers=3, n_features=2)
new_df = pd.DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
df = pd.concat([df, new_df], axis=0)
df.to_csv('argo-demo/lifecycle-management/data/data.csv')
