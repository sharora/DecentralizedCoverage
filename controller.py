import numpy as np
from scipy.spatial import KDTree
from scipy.spatial import Delaunay
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
    def __init__(self, qlis, phi, qcoor, res, mulis, sigmalis, amin, gamma, incl_consensus=False, zeta=0):
        super().__init__()
        self._qlis = qlis
        self._numrobot = qlis.shape[0]
        self._phi = phi
        self._qcoor = qcoor
        self._kdtree = KDTree(qlis)
        self._res = res
        self._amin = amin

        self._K = np.eye(2)

        self._basislen = len(mulis)

        #storing the area of a grid cell in our discretization of Q
        self._dA = (float(qcoor[1][0])/res[0])*(float(qcoor[1][1])/res[1])

        #creating arrays to store intermediate computations needed for update/control
        self._CV = np.zeros(qlis.shape)
        self._LV = np.zeros(qlis.shape)
        self._MV = np.zeros((self._numrobot, 1))
        self._Lambda = np.zeros((self._numrobot, self._basislen, self._basislen))
        self._lambda = np.zeros((self._numrobot, self._basislen))
        self._F = np.zeros((self._numrobot, self._basislen, self._basislen))


        #gamma is the learning rate
        self._gamma = gamma

        #boolean telling us whether we include the consensus term
        self._consensus = incl_consensus
        self._zeta = zeta

        #we will use constant weighting function when computing Lambda and lambda
        #for now, TODO is to see how adding a function changes things

        #creating the basis functions for each robot
        self._phihatlist = []
        for i in range(self._numrobot):
            gb = GaussianBasis(mulis, sigmalis)
            self._phihatlist.append(gb)
            #hardcoding correct params for preliminary testing
            # gb.updateparam(np.array([1.0, 1.0]))

    def step(self, dt):
        '''
        This is the method that does everything including applying controls, updating
        internal parameters, and integrating forward. Currently using Euler integration.
        '''
        #updating voronoi regions
        self._kdtree = KDTree(self._qlis)

        #creating adjaceny matrix for consensus term
        c_terms = None
        if(self._consensus):
            c_terms = self.consensus_terms()

        #Compute all integrals over voronoi regions
        self.computeVoronoiIntegrals()
        #update all parameters
        for i in range(self._numrobot):
            #equation 13, finding adots
            acurr_i = self._phihatlist[i].getparam()
            adot_i = -self._F[i] @ acurr_i - self._gamma * (self._Lambda[i] @ acurr_i - np.reshape(self._lambda[i], (self._basislen, 1)))

            if(c_terms != None):
                adot_i -= self._zeta * c_terms[i]

            #equation 14, projection step
            for j in range(self._basislen):
                if(acurr_i[j] < self._amin[j]):
                    acurr_i[j] = self._amin[j]
                    adot_i[j] = 0
                elif(acurr_i[j] == self._amin[j] and adot_i[j] < 0):
                    adot_i[j] = 0

            #euler integrating parameter derivative forward
            self._phihatlist[i].updateparam(adot_i*dt + acurr_i)

        #equation 11 updating the lambdas
        for i in range(self._numrobot):
            currkap = self._phihatlist[i].evalBasis(self._qlis[i])
            self._Lambda[i] += currkap @ np.transpose(currkap) * dt
            self._lambda[i] += currkap*self._phihatlist[i].eval(self._qlis[i]) * dt

        #apply control input and update state
        for i in range(self._numrobot):
            u_i = self._K @ (self._CV[i]-self._qlis[i])
            self._qlis[i] += u_i*dt

        #returning the current state
        return self._qlis

    def consensus_terms(self):
        #creating the Delaunay triangulation which is the dual graph of the voronoi partition
        tri = Delaunay(self._qlis).simplices
        c_terms = []

        #storing all the parameters of each robot
        a_curr = []
        for i in range(self._numrobot):
            a_curr.append(self._phihatlist[i].getparam())
            c_terms.append(np.zeros((self._basislen, 1)))

        #looping over each triangle and adding each edge to each consensus sum
        for i in range(tri.shape[0]):
            curr = tri[i]
            v1 = curr[0]
            v2 = curr[1]
            v3 = curr[2]

            #we are choosing the weighting in remark 5 because it simpler for now
            #TODO try the weighting given by the edge length shared by the 2 voronoi regions
            c_terms[v1] += (a_curr[v1] - a_curr[v2]) + (a_curr[v1] - a_curr[v3])
            c_terms[v2] += (a_curr[v2] - a_curr[v1]) + (a_curr[v2] - a_curr[v3])
            c_terms[v3] += (a_curr[v3] - a_curr[v1]) + (a_curr[v3] - a_curr[v2])

        return c_terms

    def computeVoronoiIntegrals(self):
        '''
        we opt to compute all integrals in one method so we have to sum over
        each square only once.
        '''
        #zeroing all intermediate stores
        self._CV = np.zeros(self._qlis.shape)
        self._LV = np.zeros(self._qlis.shape)
        self._MV = np.zeros((self._numrobot, 1))

        #F1, F2, are the 2 integrals needed to compute F, the third is just M_V
        F1 = np.zeros((self._numrobot, self._basislen, 2))

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

                F1[region] += self._dA * self._phihatlist[region].evalBasis(pos) @ np.transpose(pos - self._qlis[region])

        #computing all C_V based on M's and L's
        for i in range(self._numrobot):
            self._CV[i] = self._LV[i]/self._MV[i]
            # print(self._CV[i])

        #computing all F_i from F1, M_v. (Equation 12)
        for i in range(self._numrobot):
            self._F[i] = (1.0/self._MV[i])*(F1[i] @ self._K  @ np.transpose(F1[i]))
    def grid2World(self, x, y):
        '''
        we are assuming x and y are not in the image coordinate system, just
        array coordinates with the same standard orientation as R^2
        '''
        newx = self._qcoor[0][0] + (float(x)/self._res[0])*self._qcoor[1][0]
        newy = self._qcoor[0][1] + (float(y)/self._res[1])*self._qcoor[1][1]
        return np.array([newx, newy])








