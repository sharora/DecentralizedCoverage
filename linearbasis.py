import numpy as np
from scipy.stats import multivariate_normal

class GaussianBasis(object):
    def __init__(self, mulis, sigmalis):
        super().__init__()
        self._basislen = len(mulis)
        self._a = np.random.rand(self._basislen, 1)
        self._mulis = mulis
        self._sigmalis = sigmalis
    def eval(self, position):
        #TODO make this efficient
        sum = 0
        for i in range(self._basislen):
            sum += self._a[i] * multivariate_normal(self._mulis[i],
                                                    self._sigmalis[i]).pdf(position)
        return sum
    def getparam(self):
        return self._a
    def updateparam(self, newa):
        self._a = newa
