"""
An underactuated double pendulum system for comparison of different control 
methods.

The pendulum is actuated at the base link only. 
"""

import numpy as np
from scipy.linalg import block_diag
import pylagrange as lag
import sympy_utils as su
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class UnderactuatedPendulum:
    def __init__(self,dt):
        self.dt = dt
        self.x0 = np.zeros(4)
        self.x0[0] = -np.pi/2.
        self.target = np.array([0.0,2.0])
        
        self.dPend = lag.load('dPend')
        self.pos_fun = su.sympy_load('pos_fun')
        self.targetCost = su.sympy_load('targetCost')
        self.targetCost_jac = su.sympy_load('targetCost_jac')
        self.targetCost_hes = su.sympy_load('targetCost_hes')

        self.q = 1000.0
        self.r = 0.001
        
    def cost_step(self,x,u):
        cost = self.q * self.targetCost(x,self.target) + \
               self.r * np.dot(u,u)
        return cost

    def openLoopSim(self,W=[]):
        """
        Simulates the response of the system to an input
        """
        x = self.x0
        n = len(x)
        nU = 1
        NStep = len(W)/nU

        X = np.zeros((n,NStep+1))
        X[:,0] = x

        cost = 0
    
        for step in range(NStep):
            w = W[nU*step:nU*(step+1)]
            # only actuating the base link
            # by default, dPend is fully actuated
            u = np.array([w,0])
            x = self.dPend.step(x,u,self.dt,lag.EULER)
            X[:,step+1] = x

            cost = cost + self.cost_step(x,w)* self.dt

        return X, cost

    def linearization(self,x,u):
        # Because of underactuation need to put into bigger vector
        U = np.array([u,0])
        A,BFull,g = self.dPend.linearization(x,U)
        B = BFull[:,0]
        return A,B,g

    def discreteTimeLinearization(self,x,u):
        n = len(x)
        A,B,g = self.linearization(x,u)

        Adt = np.eye(n) + self.dt * A
        Bdt = self.dt * B
        gdt = self.dt * g
        return Adt,Bdt,gdt

    def costMatrix(self,xNom):
        """
        Matrix such that if x is near xNom,
        the following holds approximately:
        cost_step = [1,x',u'] * C * [1;x; u]
        """
        n = len(xNom)
        Cuu = self.r

        Hes = self.targetCost_hes(xNom,self.target)
        Cxx = .5 * Hes
        grad = self.targetCost_jac(xNom,self.target)
        Cx1 = .5 * (grad - np.dot(Hes,xNom))
        C11 = self.targetCost(xNom,self.target) - \
              np.dot(grad,xNom) + .5 * np.dot(xNom,np.dot(Hes,xNom))

        C = block_diag(C11,Cxx,Cuu)
        C[1:n+1,0] = Cx1
        C[0,1:n+1] = Cx1
        return C

    def discreteTimeCostMatrix(self,xNom):
        return self.dt * self.costMatrix(xNom)
    
    def movie(self,X):
        fig = plt.figure(1)
        plt.clf()
        ax = fig.add_subplot(111, autoscale_on=False, aspect=1,
                             xlim=(-2.1,2.1), ylim=(-2.1,2.1))

        line = ax.plot([],[],lw=2)[0]
        lineTarget = ax.plot([],[],'r*')[0]

        Joints = np.zeros((2,3))

        def init():
            line.set_data([],[])
            lineTarget.set_data(self.target[0],self.target[1])
            return line,lineTarget

        def animate(k):
            x = X[:,k]
            Joints[:,1:] = self.pos_fun(x)
            line.set_data(Joints[0],Joints[1])
            return line,lineTarget

        ani = animation.FuncAnimation(fig,animate, X.shape[1],
                                      interval=self.dt*1000, blit=False,
                                      init_func=init,repeat=False)

        return ani

