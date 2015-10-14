import numpy as np
import sympy as sym
import utils.sympy_utils as su
from numpy.random import rand
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection

import SimInterface as SI

class unicycle(SI.smoothDiffEq):
    def __init__(self):
        dt = .1
        def unicycleVF(x,u,k=0):
            theta = x[2]
            v,omega = u
            dx = np.array([v*np.cos(theta),
                           v*np.sin(theta),
                           omega])
            return dx

        self.boardLen = 15
        x0 = np.array([-self.boardLen*.1,self.boardLen/2.1,0])
        self.target = np.array([self.boardLen*1.1,self.boardLen/2,0])

        rV = 1e2
        rOmega = 1e1
        qP = 50
        qTheta = 10
        qObs = 10000
        lObs = .1

        # Generate Obstacles
        
        NumObstacles = 20
        self.obstacles = self.boardLen *(.1+.8*rand(NumObstacles,2))

        self.obstacleRadius = .75

        def obstacleDistances(p):
            obsDistSq = np.zeros(NumObstacles,dtype=p.dtype)
            for k in range(2):
                obsDistSq += (self.obstacles[:,k]-p[k])**2

            signedDistSq = obsDistSq - self.obstacleRadius**2
            return signedDistSq

        
        def unicycleCost(x,u,k=0):
            v,omega = u
            p = x[:2]
            theta = x[2]

            pErr = p - self.target[:2]
            thetaErr = theta - self.target[2]

            signedDistSq = obstacleDistances(p)

            EnergyCost = rV * v**2 + rOmega * omega**2
            StateCost = qP * np.dot(pErr,pErr) + \
                        qTheta * (1-np.cos(thetaErr))
            ObstacleCost = qObs * np.exp(-signedDistSq/lObs).sum()

            return EnergyCost + StateCost + ObstacleCost


        # Calculate Symbolic Costs
        x = sym.symarray('x',3)
        u = sym.symarray('u',2)
        z = np.hstack((x,u))

        p = x[:2]
        theta = x[2]
        v = u[0]
        omega = u[1]

        EnergyCost = rV * v**2 + rOmega * omega**2
        
        pErr = p - self.target[:2]
        thetaErr = theta - self.target[2]
        StateCost = qP * np.dot(pErr,pErr) + qTheta * (1-sym.cos(thetaErr))

        signedDistSq = obstacleDistances(p)
        ObstacleCost = 0
        for sd in signedDistSq:
            ObstacleCost += qObs * sym.exp(-sd/lObs)

        cost = EnergyCost + StateCost + ObstacleCost

        Jac = su.jacobian(cost,z)
        Hes = su.jacobian(Jac,z)

        Jac_fun = su.functify(Jac,z)
        Hes_fun = su.functify(Hes,z)
        
        def costDeriv(x,u,k=0):
            z = np.hstack((x,u))
            Jac = Jac_fun(z)
            Hes = Hes_fun(z)
            return Jac, Hes
                                   
            
        def unicycleLin(x,u,k=0):
            v = u[0]
            theta = x[2]
            A = np.array([[0,0,-v*np.sin(theta)],
                          [0,0,v*np.cos(theta)],
                          [0,0,0]])
            B = np.array([[np.cos(theta),0],
                          [np.sin(theta),0],
                          [0,1]])
            return A,B

        
        SI.smoothDiffEq.__init__(self,
                                 dt=dt,
                                 vectorField=unicycleVF,
                                 costFunc=unicycleCost,
                                 dynamicsDerivatives=unicycleLin,
                                 costDerivatives=costDeriv,
                                 x0=x0,NumInputs=2,NumStates=3)
            
sys = unicycle()


chunkLength = int(np.round(1/sys.dt))
numChunks = 10
T = numChunks *chunkLength
ilqrCtrl = SI.iterativeLQR(SYS=sys,
                           Horizon=T,
                           stoppingTolerance=1e-2)
X,U,Cost = sys.simulatePolicy(ilqrCtrl)
                           

def carMovie(filename=None):
    h=.25
    w = 1.

    Car = np.array([[-w/2,w/2,-w/2,-w/2],
                    [-h/2,  0, h/2,-h/2]])

    # Maybe put in an angl

    fig = plt.figure(1)
    plt.clf()
    

    ax = fig.add_subplot(1,1,1,autoscale_on=False,aspect=1,
                         xlim = (-2,sys.boardLen+2),
                         ylim = (-2,sys.boardLen+2))

    ax.set_xticks([])
    ax.set_yticks([])

    patches = []
    for obs in sys.obstacles:
        circle = Circle(obs,sys.obstacleRadius)
        patches.append(circle)

    PCol = PatchCollection(patches,facecolors=('r',),edgecolors=('k',))
    ax.add_collection(PCol)
    lineTarget = ax.plot([],[],lw=1,color='k')[0]
    lineHistory = ax.plot([],[],'.',lw=1,color='g')[0]
    lineCar = ax.plot([],[],lw=3,color='b')[0]

    
    def init():
        pos = sys.target[:2]
        theta = sys.target[2]
        c = np.cos(theta)
        s = np.sin(theta)
        R = np.array([[c,-s],[s,c]])
        rotTar = np.dot(R,Car)
        lineTarget.set_data(pos[0]+rotTar[0],
                            pos[1]+rotTar[1])
        lineCar.set_data([],[])
        
        lineCar.set_data([],[])
                
        return  lineTarget,  lineHistory, lineCar

    def animate(k):
        lineHistory.set_data(X[:k,0],X[:k,1])
        pos = X[k][:2]
        theta = X[k][2]
        c = np.cos(theta)
        s = np.sin(theta)
        R = np.array([[c,-s],[s,c]])
        rotCar = np.dot(R,Car)
        lineCar.set_data(pos[0]+rotCar[0],
                         pos[1]+rotCar[1])


        if filename is not None:
            if k == len(X)-1:
                plt.savefig(filename,transparent=True,
                            bbox_inches=None,format='pdf')

        return lineHistory, lineCar

    ani = animation.FuncAnimation(fig,animate,len(X),
                                  blit=False,init_func=init,
                                  interval=30,
                                  repeat=False)

    
    return ani

ani = carMovie('randomObstacleMPC.pdf')

