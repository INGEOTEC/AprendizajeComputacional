import numpy as np
from matplotlib import pylab as plt
from tqdm import tqdm

def K(x):
    left = 1 / np.sqrt(2 * np.pi)
    right = x * x / 2
    return left * np.exp(- right)


def hat_f(x, D, h):
    N = D.shape[0]
    k = [K((x - w) / h) for w in D]
    return sum(k) / (N * h)


def hat_f_k(x, D, k):
    _ = np.fabs(D - x)
    _.sort()
    h = _[k]
    N = D.shape[0]
    k = [K((x - w) / h) for w in D]
    return 1 / (N * h) * sum(k)


D = [list(map(float, x.strip().split())) for x in open("a1882_25.dat").readlines()]
D = np.array(D)
D = D[:, 2]
D = D[D <= 0.2]

x = np.linspace(D.min(), D.max(), D.shape[0])
plt.plot(x, [hat_f_k(_, D, 100) for _ in tqdm(x)])
plt.xlabel("x")
plt.ylabel("Estimación de f")
plt.savefig("estimador_knn.png")