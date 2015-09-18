#### This script defines class MarkovDecisionProcess
#### We want these code to work with time-invariant, time-varying,
#### deterministic and stochastic processes

import numpy as np
from numpy.random import randn
import pylagrange as lag
import NewtonEuler as ne
from scipy.linalg import eigh, eig
import sympy as sym
import sympy_utils as su
import numpy_utils as nu
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
    def __init__(self,dt,vectorField,costFunc,*args,**kwargs):
        self.dt = dt
        self.vectorField = vectorField
        self.costFunc = costFunc
        MarkovDecisionProcess.__init__(self,*args,**kwargs)

    def step(self,x,u,k):
        return x+ self.dt * self.vectorField(x,u,k)

    def costStep(self,x,u,k):
        return self.dt * self.costFunc(x,u,k)
    
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




#### Basic Linear Quadratic System #### 


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

        grad = nu.castToShape(grad,(self.NumStates+self.NumInputs,1))

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
        B = nu.castToShape(np.dot(BLag,inputJac),(self.NumStates,self.NumInputs))
        
        return (A,B,g)

class NewtonEulerSys(MarkovDecisionProcess, ne.NewtonEuler):
    def __init__(self,m,I,dt,x,u,Gu,cost,x0=None,frix=None,Constraint=None):
        # cost refers to cost at each step
        # cost should be a symbolic expression of x and u
    
        # Initialize parameters
        M = ne.build_M_matrix(m,I)
        self.dt = dt
        self.NumStates = len(x)
        self.NumInputs = len(u)
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
        
        ne.NewtonEuler.__init__(self,M,x,u,Gu,frix,Constraint)
       
    def step(self,x,u,k):
        return x+self.state_diff(x,u,self.dt)
    
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

        grad = nu.castToShape(grad,(self.NumStates+self.NumInputs,1))

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
