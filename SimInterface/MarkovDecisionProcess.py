#### This script defines class MarkovDecisionProcess
#### We want these code to work with time-invariant, time-varying,
#### deterministic and stochastic processes

import numpy as np
from numpy.random import randn
from copy import deepcopy
import dill
dill.settings['recurse'] = True

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


class differentialEquation(MarkovDecisionProcess):
    def __init__(self,dt=0.05,vectorField=None,costFunc=None,*args,**kwargs):
        self.dt = dt
        self.vectorField = vectorField
        self.costFunc = costFunc
        MarkovDecisionProcess.__init__(self,*args,**kwargs)

    def step(self,x,u,k):
        return x+ self.dt * self.vectorField(x,u,k)

    def costStep(self,x,u,k):
        return self.dt * self.costFunc(x,u,k)

class smoothDiffEq(differentialEquation):
    """ 
    This is a subclass of differentialEquation. 
    The extra required arguments are
    dynamicsDerivatives, the first derivatives of the dynamics
    costDerivatives, the first and second derivatives of the cost
    """
    def __init__(self,dynamicsDerivatives=None,costDerivatives=None,
                 *args,**kwargs):
        
        differentialEquation.__init__(self,*args,**kwargs)

        # A bit of explanation of these derivatives should be given
        self.dynamicsDerivatives = dynamicsDerivatives
        self.costDerivatives = costDerivatives
        
    def getApproximationMatrices(self,x,u,k=None):
        # First calculate the dynamics
        A,B = self.dynamicsDerivatives(x,u,k)
        vf = self.vectorField(x,u,k)
        Ad = np.eye(self.NumStates) + self.dt * A
        Bd = self.dt * B
        gd = self.dt * (vf - np.dot(A,x) - np.dot(B,u))
        gd = nu.castToShape(gd,(self.NumStates,1))
        dynMat = np.hstack((gd,Ad,Bd))

        # Now Calculate the cost
        Jac, Hes = self.costDerivatives(x,u,k)
        z = np.hstack((x,u))
        cost = self.costFunc(x,u,k)
        C11 = cost - np.dot(Jac,z) + .5 * np.dot(z,np.dot(Hes,z))
        C1z = .5 * (Jac - np.dot(z,Hes))
        Cz1 = C1z.T
        Czz = .5 * Hes
        costMat = np.vstack((np.hstack((C11,C1z)),
                             np.hstack((Cz1,Czz))))

        return dynMat,costMat
        
        

class driftDiffusion(MarkovDecisionProcess):
    """
    A stochastic differential equation of the form:
    dx = f(x,u)dt + g(x,u)dw

    Here f = driftFunc
         g = noiseFunc
    """
    def __init__(self,dt,driftFunc,noiseFunc,costFunc,*args,**kwargs):
        self.dt = dt
        self.driftFunc = driftFunc
        self.noiseFunc = noiseFunc
        self.costFunc = costFunc

        MarkovDecisionProcess.__init__(self,*args,**kwargs)

    def costStep(self,x,u,k):
        return self.dt * self.costFunc(x,u,k)

    def generateNoise(self,Horizon,W):
        """
        Currently this was copied from linearQuadraticStochasticSystem

        It would probably be better to make linearQuadraticStochasticSystem
        a subclass of driftDiffusion. 
        """
        if W is None:
            self.W = randn(Horizon,self.NumNoiseInputs)
        else:
            self.W = W

    def step(self,x,u,k):
        w = self.W[k]
        dx = self.dt * self.driftFunc(x,u,k) + \
             np.sqrt(self.dt) * np.dot(self.noiseFunc(x,u,k),w)

        return x + dx


#### Deterministic Subsystem from Stochastic System

class deterministicSubsystem(differentialEquation):
    """
    Create a deterministic system from a stochastic system
    """
    def __init__(self,SYS):
        differentialEquation.__init__(self,
                                      dt = SYS.dt,
                                      vectorField = SYS.driftFunc,
                                      costFunc = SYS.costFunc,
                                      x0 = SYS.x0,
                                      NumStates = SYS.NumStates,
                                      NumInputs = SYS.NumInputs)

##### Input Augmentation Function ####

def augmentInput(SYS,inputFunc,NumInputs):
    """
    This will not 
    """
    
    AugSys = deepcopy(SYS)

    AugSys.NumInputs = NumInputs

    def augmentedStep(x,u,k=0):
        fullInput = inputFunc(x,u,k)
        xnew = SYS.step(x,fullInput,k)
        return xnew

    AugSys.step = augmentedStep

    def augmentedCost(x,u,k=0):
        fullInput = inputFunc(x,u,k)
        return SYS.costStep(x,fullInput,k)

    AugSys.costStep = augmentedCost
    return AugSys

##### Basic Save and Load Commands ####

def save(SYS,name):
    """
    Saves a Markov decision process to a binary file.

    The file will be called name+'.p'
    """
    fid = open(name+'.p','wb')
    dill.dump(SYS,fid)
    fid.close()


def load(name):
    """
    func = MarkovDecisionProcess.load(name)

    This loads a system file saved using MarkovDecisionProcess.save
    """
    fid = open(name+'.p','rb')
    func = dill.load(fid)
    fid.close()
    
    return func
