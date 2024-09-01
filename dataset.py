import numpy as np
from sklearn.datasets import make_moons
import matplotlib.pylab as plt

X,Y = make_moons(150, noise=0.18, random_state=2)

plt.scatter(X[:,0], X[:,1] ,c=Y)
data = np.hstack((X, Y.reshape(-1, 1)))

fmt = ['%.4f' for i in range(X.shape[1])] + ['%d']

np.savetxt("data.csv", data,  delimiter=',', fmt=fmt)
#plt.show()