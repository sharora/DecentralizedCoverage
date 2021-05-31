from controller import Controller
from linearbasis import GaussianBasis
import numpy as np

qlis = np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2],
                 [2, 0], [2, 1], [2, 2]], dtype=float)
# mulis = np.array([[2,3],[4,5]], dtype=float)
# sigmalis = np.array([[[5,5], [5,5]], [[5,5], [5,5]]], dtype=float)
mulis = [
    [100, 50],
    [200, 150]
]
sigmalis = [
    [[1000, 0], [500, 1000]],
    [[1000, 0], [500, 2000]]
]

truephi = GaussianBasis(mulis, sigmalis)
truephi.updateparam(np.array([1.0, 1.0]))
qcoor = np.array([[0,0], [8,8]],dtype=float)
res = (8,8)


c = Controller(qlis, truephi, qcoor, res, mulis, sigmalis)

for i in range(200):
    currpos = c.step(0.02)
    print(currpos)
