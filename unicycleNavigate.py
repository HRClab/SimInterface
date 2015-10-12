import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection

import SimInterface as SI

class unicycle(SI.differentialEquation):
    def __init__(self):
        dt = .1
        def unicycleVF(x,u,k=0):
            theta = x[2]
            v,omega = u
            dx = np.array([v*np.cos(theta),
                           v*np.sin(theta),
                           omega])
            return dx

        self.boardLen = 10
        x0 = np.array([-self.boardLen*.1,self.boardLen/2.1,0])
        self.target = np.array([self.boardLen*1.1,self.boardLen/2,0])

        rV = 1e1
        rOmega = 1e-0
        qP = 10
        qTheta = 10
        qObs = 1000
        qScare = 200

        # Generate Obstacles
        
        NumObstacles = 20
        self.obstacles = self.boardLen *(.2+.6*rand(NumObstacles,2))

        # NumObstacles = 1

        # self.obstacles = np.array([[self.boardLen/2,self.boardLen/2]])
        self.obstacleRadius = .5
        self.scareRadius = self.obstacleRadius * 1.1

        def unicycleCost(x,u,k=0):
            v,omega = u
            p = x[:2]
            theta = x[2]

            pErr = p - self.target[:2]
            thetaErr = theta - self.target[2]

            obsDistSq = np.zeros(NumObstacles)
            for k in range(2):
                obsDistSq += (self.obstacles[:,k]-p[k])**2

            oneVec = np.ones(NumObstacles)
            collisionIndicator = oneVec[obsDistSq<self.obstacleRadius**2]
            scareIndicator = oneVec[obsDistSq<self.scareRadius**2]

            EnergyCost = rV * v**2 + rOmega * omega**2
            StateCost = qP * np.dot(pErr,pErr) + \
                        qTheta * (1-np.cos(thetaErr))
            ObstacleCost = qObs * collisionIndicator.sum()
            ScareCost = qScare * scareIndicator.sum()

            return EnergyCost + StateCost + ObstacleCost + ScareCost

        SI.differentialEquation.__init__(self,dt,unicycleVF,unicycleCost,
                                          x0=x0,NumInputs=2,NumStates=3)
            
sys = unicycle()

T = int(np.round(10/sys.dt))
samplingCtrl = SI.samplingOpenLoop(SYS=sys,Horizon=T,
                                    KLWeight=1e-4,burnIn=2000,
                                    ExplorationCovariance=np.diag([2,.5]),
                                    label='Sampling')

X,U,Cost = sys.simulatePolicy(samplingCtrl)

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

ani = carMovie('randomObstacle.pdf')

