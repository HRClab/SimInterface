import numpy as np

class MarkovDecisionProcess:
    def __init__(self):
        self.x0 = 0
        self.NumStates = 1
        self.NumInputs = 1

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

def shapeFromB(B):
    if isinstance(B,np.ndarray):
        if len(B.shape) == 2:
            n,p = B.shape
        else:
            n = len(B)
            p = 1
    else:
        n = 1
        p = 1

    return n,p

def buildDynamicsMatrix(A=0,B=0,g=0,timeInvariant=True):
    """
    Create a matrix, M, such that
    x[k+1] = M * [1; x; u] (in Matlab Notation)
    """
    if timeInvariant:
        n,p = shapeFromB(B)
        AMat = np.reshape(A,(n,n))
        BMat = np.reshape(B,(n,p))
        if g == 0:
            gMat = np.zeros((n,1))
        else:
            gMat = np.reshape(g,(n,1))
        H = np.hstack((gMat,AMat,BMat))
    else:
        T = A.shape[0]
        n,p = shapeFromB(B[0])
        AMat = np.reshape(A,(T,n,n))
        BMat = np.reshape(B,(T,n,p))
        if g == 0:
            gMat = np.zeros((T,n,1))
        else:
            gMat = np.reshape(g,(T,n,1))
        H = np.concatenate((gMat,AMat,BMat),axis=2)
    return H

def shapeFromSquareMatrix(M):
    if isinstance(M,np.ndarray):
        n = M.shape[0]
    else:
        n = 1
    return n

def shapeFromQR(Q,R):
    n = shapeFromSquareMatrix(Q)
    p = shapeFromSquareMatrix(R)
    return n,p

def castToShape(M,Shape):
    if (not isinstance(M,np.ndarray)) and (np.prod(Shape)>1):
        MCast = np.zeros(Shape)
    else:
        MCast = np.reshape(M,Shape)
    return MCast

def buildCostMatrix(Cxx=0,Cuu=0,C11=0,Cxu=0,Cx1=0,Cu1=0,timeInvariant=True):
    if timeInvariant:
        n,p = shapeFromQR(Cxx,Cxu)
        CxxMat = castToShape(Cxx,(n,n))
        CuuMat = castToShape(Cuu,(p,p))
        C11Mat = castToShape(C11,(1,1))
        CxuMat = castToShape(Cxu,(n,p))
        CuxMat = CxuMat.T
        Cx1Mat = castToShape(Cx1,(n,1))
        C1xMat = Cx1Mat.T
        Cu1Mat = castToShape(Cu1,(p,1))
        C1uMat = Cu1Mat.T

        C = np.vstack((np.hstack((C11Mat,C1xMat,C1uMat)),
                       np.hstack((Cx1Mat,CxxMat,CxuMat)),
                       np.hstack((Cu1Mat,CuxMat,CuuMat))))

        # deal with time-varying case later.
    return C


class LinearQuadraticSystem(MarkovDecisionProcess):
    """
    A discrete-time linear dynamical system with a quadratic cost.

    """
    def __init__(self,dynamicsMatrix,costMatrix,timeInvariant=True):
        self.dynamicsMatrix = dynamicsMatrix
        self.costMatrix = costMatrix
        self.timeInvariant = timeInvariant
        if timeInvariant:
            shapeOffset = 0
        else:
            shapeOffset = 1
        self.NumStates = dynamicsMatrix.shape[shapeOffset]
        self.NumInputs = dynamicsMatrix.shape[shapeOffset + 1] - \
                         self.NumStates - 1

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

    def getApproximationMatrices(self,x,u,k):
        if self.timeInvariant:
            return self.dynamicsMatrix, self.costMatrix
        else:
            return self.dynamicsMatrix[k:], self.costMatrix[k:]
        

