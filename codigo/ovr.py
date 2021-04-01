from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
import numpy as np
X, y = load_iris(return_X_y=True)

C = []
for cl in np.unique(y):
    yb = np.empty_like(y)
    m = y == cl
    yb[m] = 1
    yb[~m] = 0
    _ = GaussianNB().fit(X, yb)
    C.append(_)

hy = np.vstack([c.predict_proba(X)[:, 1] for c in C]).T
hy = hy.argmax(axis=1)
