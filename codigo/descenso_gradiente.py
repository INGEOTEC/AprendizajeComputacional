import numpy as np
from matplotlib import pylab as plt

x = np.linspace(-10, 10, 50)
y = 2.3 * x - 3

a = 5.3
b = -5.1
delta = np.inf
eta = 0.0001
D = [(a, b)]

while delta > 0.0001:
    hy = a * x + b
    e = (y - hy)
    a = a + 2 * eta * (e * x).sum()
    b = b + 2 * eta * e.sum()
    D.append((a, b))
    delta = np.fabs(np.array(D[-1]) - np.array(D[-2])).mean()


D = np.array(D)
plt.plot(D[:, 0], D[:, 1], '.')
plt.plot(2.3, -3, '*')
plt.savefig("descenso.png")
