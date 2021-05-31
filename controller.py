import numpy as np
from scipy.spatial import KDTree
from linearbasis import GaussianBasis


class Controller(object):
    '''
    qlis is the list of initial positions q in Q of each robot, we will assume in
    R^2 for now

    phi is the sensing function mapping Q to R^+

    qcoor is a tuple whose first element is the base coordinate,
    and the second is the dimension of the region Q in each direction
    we are assuming that we are dealing with a rectangular region

    res is a tuple telling us the number of regions we want to discretize to in
    each direction

    mulis and sigmalis represent the mean and covariance of each basis function
    that was used in representing the true sensing function, and that will be used for
    representing each robot's estimate of the sensing function
    '''
    def __init__(self, qlis, phi, qcoor, res, mulis, sigmalis):
        super().__init__()
        self._qlis = qlis
        self._numrobot = qlis.shape[0]
        self._phi = phi
        self._qcoor = qcoor
        self._kdtree = KDTree(qlis)
        self._res = res

        self._K = np.eye(2)

        #storing the area of a grid cell in our discretization of Q
        self._dA = (float(qcoor[1][0])/res[0])*(float(qcoor[1][1])/res[1])

        #creating arrays to store intermediate computations needed for update/control
        self._CV = np.zeros(qlis.shape)
        self._LV = np.zeros(qlis.shape)
        self._MV = np.zeros((self._numrobot, 1))

        #creating the basis functions for each robot
        self._phihatlist = []
        for i in range(self._numrobot):
            gb = GaussianBasis(mulis, sigmalis)
            self._phihatlist.append(gb)
            gb.updateparam(np.array([1.0, 1.0]))

    def step(self, dt):
        '''
        This is the method that does everything including applying controls, updating
        internal parameters, and integrating forward. Currently using Euler integration.
        '''
        #updating voronoi regions
        self._kdtree = KDTree(self._qlis)

        #Compute all integrals over voronoi regions
        self.computeVoronoiIntegrals()
#update all parameters

        #apply control input and update state
        for i in range(self._numrobot):
            u_i = self._K @ (self._CV[i]-self._qlis[i])
            self._qlis[i] += u_i*dt

        #returning the current state
        return self._qlis

    def computeVoronoiIntegrals(self):
        '''
        we opt to compute all integrals in one method so we have to sum over
        each square only once.
        '''
        #zeroing all intermediate stores
        self._CV = np.zeros(self._qlis.shape)
        self._LV = np.zeros(self._qlis.shape)
        self._MV = np.zeros((self._numrobot, 1))

        #looping over all squares in Q
        for i in range(self._res[0]):
            for j in range(self._res[1]):
                #converting the grid coordinate to world coordinate
                pos = self.grid2World(i,j)

                #deciding which voronoi region it belongs to
                region = self._kdtree.query(pos)[1]

                #incrementing M and L (recall we don't need to multiply by the determinant of the scaling transformation because it cancel), which in this case would be the unit area of a rectangle
                phihat = self._phihatlist[region].eval(pos)
                self._MV[region] += phihat*self._dA
                self._LV[region] += phihat*pos*self._dA

        #computing all C_V based on M's and L's
        for i in range(self._numrobot):
            self._CV[i] = self._LV[i]/self._MV[i]

    def updateParams(self):
        pass

    def grid2World(self, x, y):
        '''
        we are assuming x and y are not in the image coordinate system, just
        array coordinates with the same standard orientation as R^2
        '''
        newx = self._qcoor[0][0] + (float(x)/self._res[0])*self._qcoor[1][0]
        newy = self._qcoor[0][1] + (float(y)/self._res[1])*self._qcoor[1][1]
        return np.array([newx, newy])








