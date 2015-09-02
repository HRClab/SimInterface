import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation
import uquat as uq
from wam_params import *

def wam_movie(X,dt,Nsample,target=[]):
    # sample X according to Nsample    
    NX = X.shape[1] # number of frames in X
    sample_step = (int)(np.floor(NX/Nsample))
    X_sampled = np.zeros((X.shape[0],Nsample))
    for l in range(Nsample):
        X_sampled[:,l] = X[:,sample_step*l]
    
    # Plot
    Thetac, Hc =  np.meshgrid(np.linspace(0,2*np.pi,10),np.linspace(-0.5,0.5,2))

    CylinderPlotShape = Hc.shape
    nc = np.prod(CylinderPlotShape)
    Cylinder = np.zeros((Nlink,3,nc))

    for l in range(Nlink):
        Zc = h[l] * Hc
        Xc = r[l] * np.cos(Thetac)
        Yc = r[l] * np.sin(Thetac)
        Cylinder[l] = np.array([Xc.flatten(),Yc.flatten(),Zc.flatten()])

    Thetab, Hb = np.meshgrid(np.linspace(0,2*np.pi,5),np.linspace(-0.5,0.5,2))
    BoxPlotShape = Hb.shape
    nb = np.prod(BoxPlotShape)

    bh = .5

    Xb = bh * np.cos(Thetab)
    Yb = bh * np.sin(Thetab)
    Zb = bh * Hb
    Box = np.array([Xb.flatten(),Yb.flatten(),Zb.flatten()])
    
    mpl.rcParams['axes.color_cycle'] = ['g','b','r','c','m','y','k']

    fig = plt.figure(1)
    plt.clf()
    ax = fig.add_subplot(111,projection='3d',autoscale_on=False,aspect=1,
             xlim=(-2,2),ylim=(-2,2),zlim=(-2,2))

    def animate(k):
        ax.clear()
        line = []
        
        for l in range(Nlink):
            Pos = X_sampled[l*xstride:3+l*xstride,k]
            Quat = X_sampled[3+l*xstride:7+l*xstride,k]
            R = uq.mat(Quat)

            CylinderRot = np.dot(R,Cylinder[l])
        
            CylinderTrans = CylinderRot + np.outer(Pos,np.ones(nc))
            Xc = np.reshape(CylinderTrans[0],CylinderPlotShape)
            Yc = np.reshape(CylinderTrans[1],CylinderPlotShape)
            Zc = np.reshape(CylinderTrans[2],CylinderPlotShape)

            col = mpl.rcParams['axes.color_cycle'][l]
            line.append(ax.plot_surface(Xc,Yc,Zc,alpha=.3,
                                        cstride=1,color=col))

        if len(target) > 0:
            Pos = target[:3,k]
            Quat = target[3:,k]
            R = uq.mat(Quat)
            BoxRot = np.dot(R,Box)
            BoxTrans = BoxRot + np.outer(Pos,np.ones(nb))
            Xb = np.reshape(BoxTrans[0],BoxPlotShape)
            Yb = np.reshape(BoxTrans[1],BoxPlotShape)
            Zb = np.reshape(BoxTrans[2],BoxPlotShape)
            line.append(ax.plot_surface(Xb,Yb,Zb,alpha=.3,
                                        cstride=1,color='b'))


        ax.grid(False)
        ax.set_xlim(-3,3)
        ax.set_ylim(-3,3)
        ax.set_zlim(-3,3)
        return line,

    speed = 1
    NStep = X_sampled.shape[1]
    ani = animation.FuncAnimation(fig,animate, range(NStep),
                                  interval=dt*1000/speed,
                                  blit=False,repeat=False)

    return ani
