import numpy as np
from scipy.linalg import eig
import utils.sympy_utils as su
import numpy_utils as nu
import NewtonEuler as ne

import MarkovDecisionProcess as MDP

#### Newton Euler Systems #####

class NewtonEulerSys(MDP.MarkovDecisionProcess, ne.NewtonEuler):
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
