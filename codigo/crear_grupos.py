import numpy as np
from sklearn import datasets
from matplotlib import pylab as plt
plt.tick_params(axis="both", which="both", bottom=False,
                top=False, left=False, right=False,
                labelbottom=False, labelleft=False)
plt.grid()

X, klass = datasets.make_blobs()
plt.tight_layout()
plt.plot(X[:, 0], X[:, 1], '.')
plt.tight_layout()
plt.savefig("points.png")

plt.clf()
plt.grid()
plt.tick_params(axis="both", which="both", bottom=False,
                top=False, left=False, right=False,
                labelbottom=False, labelleft=False)
plt.tight_layout()
for k in np.unique(klass):
    d = X[klass == k]
    plt.plot(d[:, 0], d[:, 1], '.')
plt.savefig("cluster.png")