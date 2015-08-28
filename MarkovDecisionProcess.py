#### This script defines class MarkovDecisionProcess
#### We want these code to work with time-invariant, time-varying,
#### deterministic and stochastic processes

import numpy as np
from numpy.random import randn
import pylagrange as lag
from scipy.linalg import eigh, eig
import sympy as sym
import sympy_utils as su

class MarkovDecisionProcess:
    def __init__(self,x0=None,NumStates=1,NumInputs=1,NumNoiseInputs=0):
        self.NumStates = NumStates
        self.NumInputs = NumInputs
        self.NumNoiseInputs = NumNoiseInputs
        if x0 is None:
            self.x0 = np.zeros(self.NumStates)
        else:
            self.x0 = x0

    def costStep(self,x,u,k=None):
        return 0

    def step(self,x,u,k=None):
        return x

    def simulatePolicy(self,policy,W=None):
        Horizon = policy.Horizon
        self.generateNoise(Horizon,W)
            
        X = np.zeros((Horizon,self.NumStates))
        U = np.zeros((Horizon,self.NumInputs))

        x = self.x0
        X[0] = x
        cost = 0.
        for k in range(Horizon):
            u = policy.action(x,k)
            cost = cost + self.costStep(x,u,k)
            x = self.step(x,u,k)
            U[k] = u
            if k < Horizon-1:
                X[k+1] = x

        return X,U,cost

    def generateNoise(self,Horizon,W):
        pass


#### Helper functions for Linear Quadratic Systems ####

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

def castToShape(M,Shape):
    if (not isinstance(M,np.ndarray)) and (np.prod(Shape)>1):
        MCast = np.zeros(Shape)
    else:
        MCast = np.reshape(M,Shape)
    return MCast

def buildDynamicsMatrix(A=0,B=0,g=0):
    """
    Create a matrix, M, such that
    x[k+1] = M * [1; x; u] (in Matlab Notation)

    Currently only dealing with time invariant case
    """
    n,p = shapeFromB(B)
    AMat = np.reshape(A,(n,n))
    BMat = np.reshape(B,(n,p))
    gMat = castToShape(g,(n,1))
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
    
    return C

def sizesFromDynamicsMatrix(dynMat):
    # For time-varying system, we'll have A.shape = T*n*n. 
    # In that case, a negtive index would still work    
    NumStates = dynMat.shape[-2]
    NumInputs = dynMat.shape[-1] - NumStates - 1
    return NumStates, NumInputs


#### Basic Linear Quadratic System #### 

class linearQuadraticSystem(MarkovDecisionProcess):
    """
    A discrete-time linear dynamical system with a quadratic cost.

    """
    def __init__(self,dynamicsMatrix,costMatrix,timeInvariant=True,x0=None):

        self.dynamicsMatrix = dynamicsMatrix

        self.costMatrix = costMatrix
        self.timeInvariant = timeInvariant

        NumStates,NumInputs = sizesFromDynamicsMatrix(dynamicsMatrix)
        
        MarkovDecisionProcess.__init__(self,x0,NumStates,NumInputs)

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

class LagrangianSystem(MarkovDecisionProcess, lag.lagrangian_system):
    """
    A Lagrangian system constructed from symbolic 
    """
    def __init__(self,T=0,V=0,fric=0,cost=0,x=0,u=0,dt=0.01,x0=None):
        self.dt = dt
        self.NumStates = len(x)
        if isinstance(u,np.ndarray):
            self.NumInputs = len(u)
        else:
            self.NumInputs = 1
            
        if x0 is None:
            self.x0 = np.zeros(self.NumStates)
        else:
            self.x0 = x0

        # Compute cost and approximations 
        self.cost_fun = su.functify(cost,(x,u))
        z = np.hstack((x,u))
        grad = su.jacobian(cost,z)
        self.cost_grad = su.functify(grad,(x,u))
        hes = su.jacobian(grad,z)
        self.cost_hes = su.functify(hes,(x,u))

        C11 = cost - np.dot(grad,z) - .5*np.dot(z,np.dot(hes,z))
        C1z = .5 * grad - .5 * np.dot(hes,z)
        Czz = .5 * hes

        C11Mat = np.reshape(C11,(1,1))
        C1zMat = np.reshape(C1z,(1,len(z)))
        Cz1Mat = C1zMat.T

        CostMat = np.vstack((np.hstack((C11Mat,C1zMat)),
                             np.hstack((Cz1Mat,Czz))))

        self.costMat_fun = su.functify(CostMat,(x,u))
        
        lag.lagrangian_system.__init__(self,T,V,fric,x)
    def step(self,x,u,k):
        return self.stepEuler(x,u,self.dt)

    def costStep(self,x,u,k):
        return self.cost_fun(x,u)

    def getCorrectionMatrices(self,x,u,k):
        A,B,g = self.linearization(x,u)

        
        Ad = np.eye(len(x)) + self.dt * A
        Bd = self.dt * B

        vA = np.abs(eig(A)[0]).max()
        vAd = np.abs(eig(Ad)[0]).max()

        # print 'MaxEig(A): %g, MaxEig(Ad): %g' % (vA,vAd)
        
        dynMat = np.zeros((self.NumStates,
                           self.NumStates+self.NumInputs+1))

        dynMat[:,1:] = np.hstack((Ad,Bd))

        grad = self.cost_grad(x,u)
        hes = self.cost_hes(x,u)

        grad = castToShape(grad,(self.NumStates+self.NumInputs,1))

        costMat = np.vstack((np.hstack(([[0]],grad.T)),
                             np.hstack((grad,hes))))

        return dynMat, costMat
        
    def getApproximationMatrices(self,x,u,k=0):
        n = len(x)
        A,B,g = self.linearization(x,u)
        Ad = np.eye(n) + self.dt * A
        Bd = self.dt * B
        gd = self.dt * (g-np.dot(A,x)-np.dot(B,u))
        dynMat = buildDynamicsMatrix(Ad,Bd,gd)
        costMat = self.costMat_fun(x,u)
        return dynMat,costMat


class inputAugmentedLagrangian(LagrangianSystem):
    """
    Lagrangian systems with input augmented by a static function. 
    In this case the dynamics have the from

    M(q) ddq + C(q,dq) dq + Phi(q) = inputFunc(u)
    """
    def __init__(self,inputFunc=None,u=0,x=0,*args,**kwargs):
        self.inputFunc = su.functify(inputFunc,u)
        inputFunc_jac = su.jacobian(inputFunc,u)
        self.inputFunc_jac = su.functify(inputFunc_jac,u)

        LagrangianSystem.__init__(self,x=x,u=u,*args,**kwargs)

        
    def step(self,x,u,k):
        uLag = self.inputFunc(u)
        return self.stepEuler(x,uLag,self.dt)

    def linearization(self,x,u):
        uLag = self.inputFunc(u)
        A,BLag,g = LagrangianSystem.linearization(self,x,u)
        inputJac = self.inputFunc_jac(u)
        B = castToShape(np.dot(BLag,inputJac),(self.NumStates,self.NumInputs))
        
        return (A,B,g)
        
