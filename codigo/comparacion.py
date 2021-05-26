from scipy.stats import norm
import numpy as np

alpha = 0.05
z = norm().ppf( 1 - alpha / 2)
p = 0.85
N = 100
Cn = (p - z * np.sqrt(p * (1 - p) / N), p + z * np.sqrt(p * (1 - p) / N))
_ = ["%0.4f" % x for x in Cn]
print(_)

# bootstrap 

X = np.zeros(N)
X[:85] = 1

B = []
for _ in range(500):
    s = np.random.randint(X.shape[0], size=X.shape[0])
    B.append(X[s].mean())

se = np.sqrt(np.var(B))
Cn = (p - z * se, p + z * se)
_ = ["%0.4f" % x for x in Cn]
print(_)

Cn = (np.percentile(B, alpha * 100), np.percentile(B, (1 - alpha) * 100))
_ = ["%0.4f" % x for x in Cn]
print(_)

from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, StratifiedKFold

X, y = load_iris(return_X_y=True)
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3)
model = GaussianNB().fit(Xtrain, ytrain)
hy = model.predict(Xtest)
X = np.where(ytest == hy, 1, 0)
p = X.mean()
N = X.shape[0]
Cn = (p - z * np.sqrt(p * (1 - p) / N), p + z * np.sqrt(p * (1 - p) / N))
_ = ["%0.4f" % x for x in Cn]
print(_)

# B = []
# for _ in range(5000):
#     s = np.random.randint(X.shape[0], size=X.shape[0])
#     B.append(X[s].mean())
# 
# Cn = (np.percentile(B, alpha * 100), np.percentile(B, (1 - alpha) * 100))
# _ = ["%0.4f" % x for x in Cn]
# print(_)
# 
# se = np.sqrt(np.var(B))
# Cn = (p - z * se, p + z * se)
# _ = ["%0.4f" % x for x in Cn]
# print(_)


X, y = load_iris(return_X_y=True)
kf = StratifiedKFold(n_splits=10, shuffle=True)

hy = np.empty_like(y)
for tr, ts in kf.split(X, y):
    model = GaussianNB().fit(X[tr], y[tr])
    hy[ts] = model.predict(X[ts])

X = np.where(y == hy, 1, 0)
p = X.mean()
N = X.shape[0]
Cn = (p - z * np.sqrt(p * (1 - p) / N), p + z * np.sqrt(p * (1 - p) / N))
_ = ["%0.4f" % x for x in Cn]
print(_)

B = []
for _ in range(500):
    s = np.random.randint(X.shape[0], size=X.shape[0])
    B.append(X[s].mean())

se = np.sqrt(np.var(B))
Cn = (p - z * se, p + z * se)
_ = ["%0.4f" % x for x in Cn]
print(_)

Cn = (np.percentile(B, alpha * 100), np.percentile(B, (1 - alpha) * 100))
_ = ["%0.4f" % x for x in Cn]
print(_)


from mlxtend.evaluate import bootstrap_point632_score
X, y = load_iris(return_X_y=True)
cl = GaussianNB()
B = bootstrap_point632_score(cl, X, y, n_splits=500)
Cn = (np.percentile(B, alpha * 100), np.percentile(B, (1 - alpha) * 100))
_ = ["%0.4f" % x for x in Cn]
print(_)


### macro Recall
from scipy.stats import norm
import numpy as np
from sklearn.datasets import load_iris
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import recall_score

alpha = 0.05
z = norm().ppf( 1 - alpha / 2)

X, y = datasets.load_breast_cancer(return_X_y=True)
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

hy = np.empty_like(y)
for tr, ts in kf.split(X, y):
    # model = RandomForestClassifier().fit(X[tr], y[tr])
    model = GaussianNB().fit(X[tr], y[tr])
    hy[ts] = model.predict(X[ts])

B = []
for _ in range(500):
    s = np.random.randint(X.shape[0], size=X.shape[0])
    _ = recall_score(y[s], hy[s], average="macro")
    B.append(_)

p = np.mean(B)
se = np.sqrt(np.var(B))
Cn = (p - z * se, p + z * se)
_ = ["%0.4f" % x for x in Cn]
print(_)

Cn = (np.percentile(B, alpha * 100), np.percentile(B, (1 - alpha) * 100))
_ = ["%0.4f" % x for x in Cn]
print(_)


from mlxtend.evaluate import bootstrap_point632_score
X, y = datasets.load_breast_cancer(return_X_y=True)
cl = GaussianNB()
cl = RandomForestClassifier()
B = bootstrap_point632_score(cl, X, y, n_splits=500,
                             scoring_func=lambda y, hy: recall_score(y, hy, average="macro"))
Cn = (np.percentile(B, alpha * 100), np.percentile(B, (1 - alpha) * 100))
_ = ["%0.4f" % x for x in Cn]
print(_)


## Wilcoxon
import numpy as np
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import recall_score, f1_score
from scipy.stats import wilcoxon

K = 30
kf = StratifiedKFold(n_splits=K, shuffle=True, random_state=0)
X, y = datasets.load_breast_cancer(return_X_y=True)

P = []
for tr, ts in kf.split(X, y):
    forest = RandomForestClassifier().fit(X[tr], y[tr]).predict(X[ts])
    # naive = ExtraTreesClassifier().fit(X[tr], y[tr]).predict(X[ts])
    naive = GaussianNB().fit(X[tr], y[tr]).predict(X[ts])

    P.append([recall_score(y[ts], hy, average="macro") for hy in [forest, naive]])
P = np.array(P)
print(wilcoxon(P[:, 0], P[:, 1]))


p = P[:, 0] - P[:, 1]
_ = np.sqrt(K) * np.mean(p) / np.std(p)
print("%0.4f" % _)


forest = np.empty_like(y)
naive = np.empty_like(y)
alpha=0.05
for tr, ts in kf.split(X, y):
    forest[ts] = RandomForestClassifier().fit(X[tr], y[tr]).predict(X[ts])
    naive[ts] = GaussianNB().fit(X[tr], y[tr]).predict(X[ts])


B = []
for _ in range(500):
    s = np.random.randint(X.shape[0], size=X.shape[0])
    f = recall_score(y[s], forest[s], average="macro")
    n = recall_score(y[s], naive[s], average="macro")
    B.append([f, n])

Cn = (np.percentile(B, alpha * 100, axis=0), np.percentile(B, (1 - alpha) * 100, axis=0))
_ = ["%0.4f" % x for x in Cn[0]]    
print(_)
_ = ["%0.4f" % x for x in Cn[1]]    
print(_)


alpha = 0.0125
z = norm().ppf( 1 - alpha / 2)
p = 0.87
N = 1000
Cn = (p - z * np.sqrt(p * (1 - p) / N), p + z * np.sqrt(p * (1 - p) / N))
_ = ["%0.4f" % x for x in Cn]
print(_)

import numpy as np
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

X, y = datasets.load_boston(return_X_y=True)
K = 10
kf = KFold(n_splits=K, shuffle=True, random_state=0)

hy = np.empty_like(y)
for tr, ts in kf.split(X, y):
    hy[ts] = LinearRegression().fit(X[tr], y[tr]).predict(X[ts])

B = []
for _ in range(500):
    s = np.random.randint(X.shape[0], size=X.shape[0])
    f = r2_score(y[s], hy[s])
    B.append(f)
alpha = 0.05
Cn = (np.percentile(B, alpha * 100, axis=0), np.percentile(B, (1 - alpha) * 100, axis=0))
["%0.2f" % x for x in Cn]