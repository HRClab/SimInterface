"""
This is a hand-coded quadcopter model. It will be used for basic navigation
tests. 
"""

import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D


import uquat as uq
import pyopticon as POC

class quadcopter(POC.MarkovDecisionProcess):
    def __init__(self):
        dt = 0.1

        d = .2 # arm length
        cT = 1 # thrust coeff
        cQ = 0.01 # drag coeff
        g = 10.0
        
        m = 0.4
        I = np.diag([0.1,0.1,0.1])
        Iinv = np.diag(1/np.diag(I))

        InputMat = np.array([[cT,cT,cT,cT],
                             [0,d*cT,0,-d*cT],
                             [-d*cT,0,d*cT,0],
                             [-cQ,cQ,-cQ,cQ]])

        aGrav =  np.array([0,0,-g])

        ueq = np.sqrt(m*g/(4*cT)) * np.ones(4)

        
        def quadcopterStep(x,u,k=0):
            p = x[:3]
            q = x[3:7]
            v = x[7:10]
            w = x[10:13]

            vm = (u)**2
            F = np.array([0,0,np.dot(InputMat[0],vm)])
            Tau = np.dot(InputMat[1:],vm)
            
            pdot = uq.rot(q,v)
            vdot = -np.cross(w,v) + F/m + uq.rot(uq.inv(q),aGrav)
            wdot = np.dot(Iinv,-np.cross(w,np.dot(I,w))+Tau)

            pnew = p + dt * pdot
            qnew = uq.mult(q,uq.expq(.5*w*dt))
            vnew = v + dt * vdot
            wnew = w + dt * wdot

            xnew = np.hstack((pnew,qnew,vnew,wnew))

            return xnew

        x0 = np.zeros(13)
        x0[0] = -1
        x0[3] = 1

        translate = np.array([2,0,0])
        target = x0[:3] + translate

        r  = 1e-2
        qTar = 1e2
        qV = 1e1

        def quadcopterCost(x,u,k=0):
            V = x[7:]
            StateErr = x[:3] - target

            InputCost = r * np.dot(u,u)
            StateCost = qTar * np.dot(StateErr,StateErr) + \
                        qV * np.dot(V,V)

            InstCost = InputCost + StateCost
            return dt * InstCost

        POC.MarkovDecisionProcess.__init__(self,x0=x0,
                                           NumStates=13,NumInputs=4)

        self.step = quadcopterStep
        self.costStep = quadcopterCost
        self.dt = dt
        self.target = target
        self.Al = d
        self.Bl = .1
        
sys = quadcopter()
chunkLength = int(np.round(1/sys.dt))
numChunks = 4
T = numChunks * chunkLength
# samplingCtrl = POC.samplingMPC(SYS=sys,
#                                Horizon=T,
#                                KLWeight=1e-4,
#                                ExplorationCovariance=1*np.eye(4),
#                                PredictionHorizon=2*chunkLength,
#                                PredictionBurnIn=2)

samplingCtrl = POC.samplingOpenLoop(SYS=sys,
                                    Horizon=T,
                                    KLWeight=1e-4,
                                    ExplorationCovariance=1*np.eye(4),
                                    burnIn=1000)


X,U,Cost = sys.simulatePolicy(samplingCtrl)

def movie(filename=None):
    fig = plt.figure(1)
    plt.clf()

    l = -1.5
    ax = fig.add_subplot(111, projection='3d', autoscale_on=False, aspect=1,
                     xlim=(-l,l), ylim=(-l,l), zlim = (-l,l))
    ax.view_init(elev=17., azim=34.)

    lw = 2
    nTheta = 50
    theta = np.linspace(0,2*np.pi,nTheta)
    circ = np.zeros((3,50))
    circ[0,:] = sys.Bl * np.cos(theta)
    circ[1,:] = sys.Bl * np.sin(theta)

    frame = np.array([[sys.Al,0,-sys.Al,0],[0,sys.Al,0,-sys.Al],[0,0,0,0]])

    cen = np.zeros((3,2))
    colOrd = np.array(['b','g','r','c'])

    def animate(k):
        ax.clear()

        pltList = []

        pltList.append(ax.plot([sys.target[0]],
                               [sys.target[1]],
                               [sys.target[2]],'*'))

        p = X[k,:3]
        q = X[k,3:7]
        R = uq.mat(q)
        curFrame = np.outer(p,np.ones(4)) + np.dot(R,frame)

        # draw the arms
        for arm in range(2):
            armCoord = curFrame[:,arm::2]
            pltList.append(ax.plot(armCoord[0,:],armCoord[1,:],armCoord[2,:],
                               color='k',linewidth=lw))

        # Draw the blades
        for prop in range(4):
            curCirc = np.outer(p,np.ones(nTheta)) + \
                      np.dot(R,circ+np.outer(frame[:,prop],np.ones(nTheta)))
            pltList.append(ax.plot(curCirc[0],curCirc[1],curCirc[2],
                                   color=colOrd[prop],linewidth=lw))

        # Draw the history

        pltList.append(ax.plot(X[:k,0],X[:k,1],X[:k,2],'.',color='g'))
        
        ax.set_xlim(-l,l)
        ax.set_ylim(-l,l)
        ax.set_zlim(-l,l)
        ax.invert_zaxis() # Somehow it is flipped
        return pltList,

    ani = animation.FuncAnimation(fig,animate, len(X),
                                  interval=30,
                                  blit=False,repeat=False)

    return ani
ani = movie()
