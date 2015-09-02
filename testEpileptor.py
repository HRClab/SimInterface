import MarkovDecisionProcess as MDP
import Controller as ctrl
import numpy as np
from numpy.random import randn
import numpy_utils as nu
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from multiprocessing import Process


class epileptor(MDP.driftDiffusion):
    """
    Epileptor Model as defined by 
    Jirsa et. al. "On the nature of seizure dynamics"
    """
    def __init__(self,inputWeight):
        dt = 0.0005

        # Parameters
        x0 = -1.6             # Not initial condition
        y0 = 1.0
        xtau=0.01
        tau0 = 5.0
        tau1 = 0.015
        tau2 = .1
        Irest1 = 3.2
        Irest2 = 0.45
        gamma = 0.01
        
        def driftFunc(x,u,k):
            x1,y1,x2,y2,z,g = x

            if x1 < 0:
                f1 = x1**3. - 3*x1**2.
            else:
                f1 = (x2 - 0.6*(z-4.)**2.) * x1

            if x2 < -0.25:
                f2 = 0.
            else:
                f2 = 6 * (x2+0.25)
                
            dx1 = (y1 - f1 - z + Irest1) / xtau
            dy1 = (y0 - 5*x1**2. - y1) / tau1
            dx2 = (-y1 + x2 - x2**3. + Irest2 + 2. * g -0.3*(z-3.5)) / xtau
            dy2 = (-y2 + f2) / tau2
            dz = (1./tau0) * (4*(x1+2)-z) + u
            dg = -gamma * (g - 0.1 * x1)/xtau

            dx = np.array([dx1,dy1,dx2,dy2,dz,dg])
            return dx

        noiseMat = np.zeros((6,2))
        noiseMat[0,0] = 0.025
        noiseMat[1,1] = 0.1

        def noiseFunc(x,u,k):
            return noiseMat

        def costFunc(x,u,k):
            LFP = x[0]+x[2]
            return inputWeight * u**2. + 1000 * LFP**2.
            
        initialCondition = np.array([0.022,0.91,-1.11,0.73,3.84,0])
        MDP.driftDiffusion.__init__(self,
                                    dt=dt,
                                    driftFunc=driftFunc,
                                    noiseFunc=noiseFunc,
                                    costFunc=costFunc,
                                    x0=initialCondition,
                                    NumStates=len(initialCondition),
                                    NumInputs=1,
                                    NumNoiseInputs=noiseMat.shape[1])
    




# Reset initial condition
InputWeights = [1,10,20,100]

Systems = []
Controllers = []
lineInput = []
Cov = 1

chunkLength = 200 
NumChunks = 300

for k in range(len(InputWeights)):
    inputWeight = InputWeights[k]
    sys = epileptor(inputWeight)
    Systems.append(sys)
    sysDet = MDP.deterministicSubsystem(sys)
    name = 'Input Penalty: %g' % inputWeight
    controller = ctrl.samplingMPC(SYS=sysDet,
                                  Horizon=chunkLength,
                                  KLWeight=1e-4,
                                  ExplorationCovariance=Cov,
                                  PredictionHorizon=20,
                                  PredictionBurnIn=1,
                                  label=name)
    Controllers.append(controller)
    


NumControllers = len(Controllers)
Time = sys.dt * np.arange(chunkLength * NumChunks)

LFP = np.nan * np.ones((NumControllers,chunkLength * NumChunks))
Input = np.nan * np.ones((NumControllers,chunkLength * NumChunks))




def runningMovie():
    fig = plt.figure(1)
    plt.clf()
    plt.xlim((0,chunkLength*NumChunks*0.0005))
    plt.ylim((-1.5,2.8))
    plt.yticks([-1,0,1,2])
    lineLFP = []
    
    for k in range(NumControllers):
        name = Controllers[k].label
        lineLFP.append(plt.plot([],[],lw=3,alpha=.8,label=name)[0])

    plt.legend(handles=lineLFP)

    
    def initMovie():
        for k in range(NumControllers):
            lineLFP[k].set_data([],[])
        return lineLFP

    def animate(chunk):
        for k in range(NumControllers):
            sys = Systems[k]
            controller = Controllers[k]
            name = controller.label
            X,U,Cost = sys.simulatePolicy(controller)
            sys.x0 = X[-1]
            LFP[k,chunkLength*chunk:chunkLength*(chunk+1)] = X[:,0] + X[:,2]
            Input[k,chunkLength*chunk:chunkLength*(chunk+1)] = U.squeeze()
            lineLFP[k].set_data(Time,LFP[k])

        return lineLFP

    ani = animation.FuncAnimation(fig,animate,NumChunks,blit=False,
                                  interval = sys.dt * 2000,
                                  init_func=initMovie,repeat=False)
    return ani

def extraPlots():
    plt.savefig('LFPComparison.pdf',transparent=True,
                bbox_inches=None,format='pdf')

    plt.figure(2)
    plt.clf()

    lineInput = []

    for k in range(NumControllers):
        controller = Controllers[k]
        name = controller.label
        handle = plt.plot(Time,Input[k],linewidth=3,alpha=.8,label=name)[0]
        lineInput.append(handle)

    plt.legend(handles=lineInput)

ani = runningMovie()
# This is a hack to stop the rest of the code from executing before
# the animation completes.
# Apparently, the animation function does some sort of
# Multiprocessing that I do not know how to debug.
# The data will be saved for better plotting later
plt.show(block=True)
extraPlots()



# Save the data because the animation does weird threading
fid = open('LFPData.npz','w')
np.savez(fid,Time=Time,LFP=LFP,Input=Input)
fid.close()
