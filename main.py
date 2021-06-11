from controller import Controller
from linearbasis import GaussianBasis
import numpy as np
import matplotlib.pyplot as plt

qlis = np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2],
                 [2, 0], [2, 1], [2, 2]], dtype=float)
# mulis = np.array([[2,3],[4,5]], dtype=float)
# sigmalis = np.array([[[5,5], [5,5]], [[5,5], [5,5]]], dtype=float)
mulis = [
    [6, 2],
    [2, 6]
]
sigmalis = [
    [[0.5, 0], [0, 0.5]],
    [[0.5, 0], [0, 0.5]]
]

truephi = GaussianBasis(mulis, sigmalis)
truephi.updateparam(np.array([1.0, 1.0]))
qcoor = np.array([[0,0], [8,8]],dtype=float)
res = (8,8)


c = Controller(qlis, truephi, qcoor, res, mulis, sigmalis)

graphcolors = np.random.rand(9)
for i in range(1000):
    currpos = c.step(0.02)
    plt.clf()
    currpos = np.transpose(currpos)
    plt.scatter(currpos[0], currpos[1],c=graphcolors, alpha=0.5)
    plt.draw()
    plt.pause(0.02)
plt.pause(5)
