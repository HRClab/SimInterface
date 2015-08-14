import numpy as np

class MarkovDecisionProcess:
    def __init__(self):
        self.x0 = 0

    def costStep(self,x,u,k):
        return 0

    def step(self,x,u,k):
        return x

    def simulatePolicy(self,policy):
        Horizon = policy.Horizon
        if isinstance(self.x0,np.ndarray):
            n = len(self.x0)
            X = np.zeros((Horizon+1,n))
        else:
            X = np.zeros(Horizon+1)

        x = self.x0
        X[0] = x
        cost = 0.
        for k in range(Horizon):
            u = policy.action(x,k)
            cost = cost + self.costStep(x,u,k)
            x = self.step(x,u,k)
            X[k+1] = x

        return X,cost

class LinearQuadraticSystem(MarkovDecisionProcess):
    """
    A discrete-time linear dynamical system with a quadratic cost.

    """
    def __init__(self,A=0,B=0,g=0,
                 Cxx=0,Cuu=0,C11=0,Cxu=0,Cx1=0,Cu1=0,
                 timeInvariant = True):

        self.timeInvariant = timeInvariant
        
        if timeInvariant:
            self.dynamicsMatrix = self.buildDynamicsMatrix(A,B,g)
            self.costMatrix = self.buildCostMatrix(Cxx,Cuu,C11,
                                                   Cxu,Cx1,Cu1)
        else:
            Horizon = len(A)
            if isintance(A[0],np.ndarray):
                n = A[0].shape[0]
                self.dynamicsMatrix = np.zeros((Horizon,n,n))
                self.costMatrix = np.zeros((Horizon,n,n))
            else:
                self.dynamicsMatrix = np.zeros(Horizon)
                # May need to change this later. . . 
                self.costMatrix = np.zeros((Horizon,n,n))
                
            for k in range(Horizon):
                self.dynamicsMatrix[k] = self.buildDynamicsMatrix(A[k],
                                                                  B[k],
                                                                  g[k])

                self.costMatrix[k] = self.buildCostMatrix(Cxx[k],
                                                          Cuu[k],
                                                          C11[k],
                                                          Cxu[k],
                                                          Cx1[k],
                                                          Cu1[k])
            
    def buildDynamicsMatrix(self,A,B,g):
        """
        Create a matrix, M, such that
        x[k+1] = M * [1; x; u] (in Matlab Notation)
        """
        if isinstance(A,np.ndarray):
            # assume that A is an nxn array
            n = A.shape[0]
            gMat = np.reshape(g,(n,1))
            if len(B.shape) == 1:
                BMat = np.reshape(g,(n,1))
            else:
                BMat = B

            return np.hstack((gMat,A,BMat))
        else:
            # Assume that A, B, and g are scalars
            return np.hstack((g,A,B))


    def buildCostMatrix(self,Cxx,Cuu,C11,Cxu,Cx1,Cu1):
        if isinstance(Cxx,np.ndarray):
            # In this case, Cxx is an nxn array
            n = Cxx.shape[0]
            if isintance(Cuu,np.ndarray):
                # In this case, Cuu is a pxp array
                p = Cuu.shape[1]
            else:
                p = 1

        else:
            n = 1
            p = 1
        CxxMat = np.reshape(Cxx,(n,n))
        CuuMat = np.reshape(Cuu,(p,p))
        C11Mat = np.reshape(C11,(1,1))
        CxuMat = np.reshape(Cxu,(n,p))
        CuxMat = CxuMat.T
        Cx1Mat = np.reshape(Cx1,(n,1))
        C1xMat = Cx1Mat.T
        Cu1Mat = np.reshape(Cu1,(p,1))
        C1uMat = Cu1Mat.T

        C = np.vstack((np.hstack((C11Mat,C1xMat,C1uMat)),
                       np.hstack((Cx1Mat,CxxMat,CxuMat)),
                       np.hstack((Cu1Mat,CuxMat,CuuMat))))

        return C

    def step(self,x,u,k):
        if self.timeInvariant:
            dynMat = self.dynamicsMatrix
        else:
            dynMat = self.dynamicsMatrix[k]

        curVec = np.hstack((1,x,u))
        new_x = np.dot(dynMat,curVec)
        return new_x

    def costStep(self,x,u,k):
        if self.timeInvariant:
            costMat = self.costMatrix
        else:
            costMat = self.costMatrix[k]

        curVec = np.hstack((1,x,u))
        cost = np.dot(curVec,np.dot(costMat,curVec))
        return cost
    
