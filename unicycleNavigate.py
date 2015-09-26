import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pyopticon as POC

class unicycle(POC.differentialEquation):
    def __init__(self):
        dt = .1
        def unicycleVF(x,u,k=0):
            theta = x[2]
            v,omega = u
            dx = np.array([v*np.cos(theta),
                           v*np.sin(theta),
                           omega])
            return dx

        x0 = np.zeros(3)
        self.target = np.array([0,3,0])

        rV = 1e-3
        rOmega = 1e-3
        qP = 1
        qTheta = 1
    
        def unicycleCost(x,u,k=0):
            v,omega = u
            p = x[:2]
            theta = x[2]

            pErr = p - self.target[:2]
            thetaErr = theta - self.target[2]

            EnergyCost = rV * v**2 + rOmega * omega**2
            StateCost = qP * np.dot(pErr,pErr) + \
                        qTheta * (1-np.cos(thetaErr))

            return EnergyCost + StateCost

        POC.differentialEquation.__init__(self,dt,unicycleVF,unicycleCost,
                                          x0=x0,NumInputs=2,NumStates=3)
            
sys = unicycle()

T = int(np.round(10/sys.dt))
samplingCtrl = POC.samplingOpenLoop(SYS=sys,Horizon=T,
                                    KLWeight=1e-4,burnIn=1000,
                                    ExplorationCovariance=np.diag([.25,.25]),
                                    label='Sampling')

X,U,Cost = sys.simulatePolicy(samplingCtrl)

def carMovie():
    h=.25
    w = 1.

    Car = np.array([[-w/2,w/2,-w/2,-w/2],
                    [-h/2,  0, h/2,-h/2]])

    # Maybe put in an angl

    fig = plt.figure(1)
    plt.clf()
    

    ax = fig.add_subplot(1,1,1,autoscale_on=False,aspect=1,
                         xlim = (-5,5),ylim=(-5,5))

    ax.set_xticks([])
    ax.set_yticks([])

    lineCar = ax.plot([],[],lw=2)[0]
    lineTarget = ax.plot([],[],lw=1)[0]
    
    def init():
        lineCar.set_data([],[])
        pos = sys.target[:2]
        theta = sys.target[2]
        c = np.cos(theta)
        s = np.sin(theta)
        R = np.array([[c,-s],[s,c]])
        rotTar = np.dot(R,Car)
        lineTarget.set_data(pos[0]+rotTar[0],
                            pos[1]+rotTar[1])
        return lineCar, lineTarget

    def animate(k):
        pos = X[k][:2]
        theta = X[k][2]
        c = np.cos(theta)
        s = np.sin(theta)
        R = np.array([[c,-s],[s,c]])
        rotCar = np.dot(R,Car)
        lineCar.set_data(pos[0]+rotCar[0],
                         pos[1]+rotCar[1])

        return lineCar,

    ani = animation.FuncAnimation(fig,animate,len(X),
                                  blit=False,init_func=init,
                                  interval=30,
                                  repeat=False)

    
    return ani

ani = carMovie()

