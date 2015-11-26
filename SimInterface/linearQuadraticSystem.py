import numpy as np
from numpy.random import randn
from scipy.linalg import eigh
import utils.numpy_utils as nu
import MarkovDecisionProcess as MDP

#### Helper functions for Linear Quadratic Systems ####
def sizesFromDynamicsMatrix(dynMat):
    # For time-varying system, we'll have A.shape = T*n*n. 
    # In that case, a negtive index would still work    
    NumStates = dynMat.shape[-2]
    NumInputs = dynMat.shape[-1] - NumStates - 1
    return NumStates, NumInputs


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


def buildDynamicsMatrix(A=0,B=0,g=0):
    """
    Create a matrix, M, such that
    x[k+1] = M * [1; x; u] (in Matlab Notation)

    Currently only dealing with time invariant case
    """
    n,p = shapeFromB(B)
    AMat = np.reshape(A,(n,n))
    BMat = np.reshape(B,(n,p))
    gMat = nu.castToShape(g,(n,1))
    H = np.hstack((gMat,AMat,BMat))
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


def buildCostMatrix(Cxx=0,Cuu=0,C11=0,Cxu=0,Cx1=0,Cu1=0):
    """
    Build a matrix so the step cost can be written as 

    stepCost = [1]'[C11 C1x C1u][1] 
               [x] [Cx1 Cxx Cxu][x]
               [u] [Cu1 Cux Cuu][u]

    Currently, only handling the the time-invariant case
    
    """
    n,p = shapeFromQR(Cxx,Cxu)
    CxxMat = nu.castToShape(Cxx,(n,n))
    CuuMat = nu.castToShape(Cuu,(p,p))
    C11Mat = nu.castToShape(C11,(1,1))
    CxuMat = nu.castToShape(Cxu,(n,p))
    CuxMat = CxuMat.T
    Cx1Mat = nu.castToShape(Cx1,(n,1))
    C1xMat = Cx1Mat.T
    Cu1Mat = nu.castToShape(Cu1,(p,1))
    C1uMat = Cu1Mat.T

    C = np.vstack((np.hstack((C11Mat,C1xMat,C1uMat)),
                   np.hstack((Cx1Mat,CxxMat,CxuMat)),
                   np.hstack((Cu1Mat,CuxMat,CuuMat))))
    
    return C


#### Basic LQ Systems ######
class linearQuadraticSystem(MDP.MarkovDecisionProcess):
    """
    A discrete-time linear dynamical system with a quadratic cost.

    """
    def __init__(self,dynamicsMatrix,costMatrix,timeInvariant=True,x0=None):

        self.dynamicsMatrix = dynamicsMatrix

        self.costMatrix = costMatrix
        self.timeInvariant = timeInvariant

        NumStates,NumInputs = sizesFromDynamicsMatrix(dynamicsMatrix)
        
        MDP.MarkovDecisionProcess.__init__(self,x0,NumStates,NumInputs)

    def step(self,x,u,k=None):
        if self.timeInvariant:
            dynMat = self.dynamicsMatrix
        else:
            dynMat = self.dynamicsMatrix[k]

        curVec = np.hstack((1,x,u))
        new_x = np.dot(dynMat,curVec)
        return new_x

    def costStep(self,x,u,k=None):
        if self.timeInvariant:
            costMat = self.costMatrix
        else:
            costMat = self.costMatrix[k]

        curVec = np.hstack((1,x,u))
        cost = np.dot(curVec,np.dot(costMat,curVec))
        return cost

    def getApproximationMatrices(self,x,u,k=None):
        if self.timeInvariant:
            return self.dynamicsMatrix, self.costMatrix
        else:
            return self.dynamicsMatrix[k:], self.costMatrix[k:]

    def getCorrectionMatrices(self,x,u,k=None):
        if self.timeInvariant:
            # Drop the g vector
            dynMat = np.zeros(self.dynamicsMatrix.shape)
            dynMat[:,1:] = self.dynamicsMatrix[:,1:]

            # Compute the correction terms in the linear quad
            C = self.costMatrix
            C1z = C[0,1:]
            Czz = C[1:,1:]
            z = np.hstack((x,u))
            row = C1z + np.dot(z,Czz)
            row = np.reshape(row,(1,len(row)))
            costMat = np.vstack(np.hstack(([[0]],row)),
                                np.hstack((row.T,Czz)))
            return dynMat, costMat
            
        else:
            print 'Sorry only time invariant at the moment'

####

class linearQuadraticStochasticSystem(linearQuadraticSystem):
    """
    A discrete-time linear system with quadratic cost and Gaussian Process noise
    
    x[k+1] = Ax[k] + Bu[k] + Gw[k]

    where w[k] is identity covariance, zero-mean, Gaussian noise
    """

    def __init__(self,dynamicsMatrix,costMatrix,noiseMatrix,
                 timeInvariant=True,x0=None):

        linearQuadraticSystem.__init__(self,dynamicsMatrix,
                                       costMatrix,timeInvariant,x0)
        
        self.noiseMatrix = noiseMatrix
        self.NumNoiseInputs = noiseMatrix.shape[-1]
    
    def generateNoise(self,Horizon,W):
        if W is None:
            self.W = randn(Horizon,self.NumNoiseInputs)
        else:
            self.W = W

    def step(self,x,u,k=None):
        """
        Stochastic Step
        """

        # The mean is updated like a standard linear system
        new_x_mean = linearQuadraticSystem.step(self,x,u,k)

        # Now compute the noise term
        w = self.W[k]
        if self.timeInvariant:
            G = self.noiseMatrix
        else:
            G = self.noiseMatrix[k]
        
        return new_x_mean + np.dot(G,w)
        
def convexApproximationMatrices(SYS,x,u,k):
    dynMat,costMat = SYS.getApproximationMatrices(x,u,k)
    n = SYS.NumStates
    p = SYS.NumInputs

    # Check cost matrix is not convex and then apply
    # a little hack to fix it    
    eigMin = eigh(costMat[1:n+1,1:n+1],eigvals_only=True,eigvals=(0,0))[0]
    if eigMin < 0:
        z = x.reshape((n,1))
        alpha = -1.1 * eigMin
        costMat[:n+1,:n+1] += alpha * \
                              np.vstack((np.hstack((np.dot(z.T,z),-z.T)),
                                         np.hstack((-z,np.eye(n)))))


    return dynMat,costMat

def buildCorrectionSystem(SYS,X,U):
    """
    Builds the linear-quadratic correction system.
    """
    Horizon = len(U)
    n = SYS.NumStates
    p = SYS.NumInputs
    NumStepVars = n+p+1
    dynMat = np.zeros((Horizon,n,NumStepVars))
    costMat = np.zeros((Horizon,NumStepVars,NumStepVars))
    for k in range(Horizon):
        dynMat[k],costMat[k] = SYS.getCorrectionMatrices(X[k],U[k],k)
        
    return linearQuadraticSystem(dynMat,costMat,
                                 timeInvariant=False,x0=np.zeros(n))


def buildApproximateLQSystem(SYS,X,U):
    """
    Takes a system SYS and a state input trajectory (X,U) and constructs
    the linear-quadratic approxmation to the nonlinear system
    """
    Horizon = len(U)
    n = SYS.NumStates
    p = SYS.NumInputs
    NumStepVars = n+p+1
    dynMat = np.zeros((Horizon,n,NumStepVars))
    costMat = np.zeros((Horizon,NumStepVars,NumStepVars))
    for k in range(Horizon):
        dynMat[k],costMat[k] = convexApproximationMatrices(SYS,X[k],U[k],k)
    return linearQuadraticSystem(dynMat,costMat,timeInvariant=False,x0=SYS.x0)
