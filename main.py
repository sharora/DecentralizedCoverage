from controller import Controller
from linearbasis import GaussianBasis
import numpy as np
import matplotlib.pyplot as plt


qlis = np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2],
                 [2, 0], [2, 1], [2, 2]], dtype=float)
numrobot = qlis.shape[0]
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
amin = np.array([0.1, 0.1])


c = Controller(qlis, truephi, qcoor, res, mulis, sigmalis, amin)
numsteps = 1000

#lists for tracking distance between robot parameters and true parameters
adislist = []
for i in range(numrobot):
    adislist.append([])

graphcolors = np.random.rand(numrobot)
for i in range(numsteps):
    #forward step
    currpos = c.step(0.02)

    #graphing current robot positions
    plt.clf()
    currpos = np.transpose(currpos)
    plt.scatter(currpos[0], currpos[1],c=graphcolors, alpha=0.5)
    plt.draw()
    plt.pause(0.02)

    #adding parameter distances to list
    for j in range(numrobot):
        dist = np.linalg.norm(c._phihatlist[j].getparam() - truephi.getparam())
        adislist[j].append(dist)

#graphing parameter distances with respect to time
plt.clf()
for i in range(numrobot):
    plt.plot(np.array(adislist[i]))
    plt.show()

