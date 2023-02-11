from scipy.stats import multivariate_normal
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits, load_diabetes
from scipy.stats import norm
from collections import Counter
import numpy as np
import pandas as pd
from matplotlib import pylab as plt
import seaborn as sns
sns.set_theme()

X_1 = multivariate_normal(mean=[5, 5], cov=[[4, 0], [0, 2]]).rvs(1000)
X_2 = multivariate_normal(mean=[-5, -10], cov=[[2, 1], [1, 3]]).rvs(1000)
X_3 = multivariate_normal(mean=[15, -6], cov=[[2, 3], [3, 7]]).rvs(1000)

df = pd.DataFrame([dict(x=x, y=y, clase=1) for x, y in X_1] + \
                  [dict(x=x, y=y, clase=2) for x, y in X_2] + \
                  [dict(x=x, y=y, clase=3) for x, y in X_3])

sns.relplot(data=df, kind='scatter',
            x='x', y='y', hue='clase')

X = np.concatenate((X_1, X_2, X_3), axis=0)
y = np.array([1] * 1000 + [2] * 1000 + [3] * 1000)
arbol = tree.DecisionTreeClassifier().fit(X, y)



# from sklearn import datasets
# from sklearn import tree

# X, y = datasets.load_boston(return_X_y=True)
# X, y = datasets.load_diabetes(return_X_y=True)

# arbol = tree.DecisionTreeRegressor(max_depth=2, min_samples_split=10, splitter="random").fit(X, y)

# with open("tree.dot", "w") as fpt:
#     _ = tree.export_graphviz(arbol)
#     fpt.write(_)

# ## !dot -Tpng tree.dot -o tree.png
