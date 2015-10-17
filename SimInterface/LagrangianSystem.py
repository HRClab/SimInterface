import numpy as np
from scipy.linalg import eig
import utils.sympy_utils as su
import numpy_utils as nu
import pylagrange as lag

import MarkovDecisionProcess as MDP
import linearQuadraticSystem as LQS

class LagrangianSystem(MDP.MarkovDecisionProcess, lag.lagrangian_system):
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
        
    def step(self,x,u,k=0):
        return self.stepEuler(x,u,self.dt)

    def costStep(self,x,u,k=0):
        return self.cost_fun(x,u)

    def getCorrectionMatrices(self,x,u,k=0):
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
        dynMat = LQS.buildDynamicsMatrix(Ad,Bd,gd)
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

        
    def step(self,x,u,k=0):
        uLag = self.inputFunc(u)
        return self.stepEuler(x,uLag,self.dt)

    def linearization(self,x,u):
        uLag = self.inputFunc(u)
        A,BLag,g = LagrangianSystem.linearization(self,x,u)
        inputJac = self.inputFunc_jac(u)
        B = nu.castToShape(np.dot(BLag,inputJac),(self.NumStates,self.NumInputs))
        
        return (A,B,g)
