import numpy as np
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
from matplotlib import pylab as plt
from tqdm import tqdm

## El conjunto de datos se descargó de:
## http://www.stat.cmu.edu/~larry/all-of-statistics/=Rprograms/a1882_25.dat

D = [list(map(float, x.strip().split())) for x in open("a1882_25.dat").readlines()]
D = np.array(D)
D = D[:, 2]
D = D[D <= 0.2]
# D = MinMaxScaler().fit_transform(np.atleast_2d(D).T)[:, 0]


def riesgo(D, m=10):
    """Riesgo de validación cruzada de histograma"""
    N = D.shape[0]
    limits = np.linspace(D.min(), D.max(), m + 1)
    h = limits[1] - limits[0]
    _ = np.searchsorted(limits, D, side='right')
    _[_ == 0] = 1
    _[_ == m + 1] = m
    p_j = Counter(_)
    cuadrado = sum([(x / N)**2 for x in p_j.values()])
    return (2 / ((N - 1) * h)) - ((N + 1) * cuadrado / ((N - 1) * h))


m = np.arange(2, 500)
r = [riesgo(D, x) for x in m]
print(np.argmin(r))
plt.plot(m, r)
plt.xlabel("Número de bins (m)")
plt.ylabel("Riesgo")
plt.grid()
plt.savefig("hist-riesgo.png")


# from sklearn import datasets
# X, y = datasets.load_boston(return_X_y=True)
# 
# m = np.arange(2, 300)
# r = [riesgo(y, x) for x in m]
# print(np.argmin(r))
# plt.plot(m, r)
# plt.grid()
# plt.xlabel("Número de bins")
# plt.ylabel("Riesgo")
# plt.savefig("riesgo-boston.png")