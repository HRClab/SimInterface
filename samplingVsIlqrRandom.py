import SimInterface as SI
import numpy as np
from numpy.random import randn
import matplotlib.pyplot as plt
import matplotlib as mpl
import sympy as sym
import sympy_utils as su

np.random.seed(123152)
mpl.rcParams.update({'font.size':22})


#### Define the pendulum on a cart system ####

class pendulumCart(SI.inputAugmentedLagrangian):
    def __init__(self):
        dt = 0.05
        mPole = 1.
        mCart = 2.
        lPole = 1.
        g = 10.
        target = np.array([0,lPole])
        self.target = target
        self.lPole = lPole

        pCart, theta = sym.symbols('pCart theta')
        q = np.array([pCart,theta])
        dq = sym.symarray('dq',2)
        x = np.hstack((q,dq))
        u = sym.symbols('u')

        # Initial condition
        x0 = np.zeros(4)
        x0[1] = -np.pi/2

        self.xDown = x0
        # Cartesian Positions
        xCart = np.array([pCart,0])
        xPole = xCart + lPole * np.array([sym.cos(theta),sym.sin(theta)])

        # Cartesian Velocities
        vCart = np.dot(su.jacobian(xCart,q),dq)
        vPole = np.dot(su.jacobian(xPole,q),dq)

        # Kinetic Energy
        T = .5 * mCart * np.dot(vCart,vCart) + \
            .5 * mPole * np.dot(vPole,vPole)

        # Potential Energy
        V = mPole * g * xPole[1]

        # Friction
        # This is entry-by-entry multiplication
        fric = np.array([.2,.1]) * dq

        # Cost
        EnergyCost = 0.0001 * np.dot(u,u)
        targetError = target - xPole
        targetCost = 1000 * np.dot(targetError,targetError)
        speedError = np.array([.1,.2]) * dq
        speedCost = np.dot(speedError,speedError)

        Cost = dt * (EnergyCost + targetCost + speedCost) / 20000

        # Input function, since only the cart is actuated
        inputFunc = np.array([u,0])

        # Position function for simpler plotting
        pos = np.zeros((2,2),dtype=object)
        pos[0] = xCart
        pos[1] = xPole
        self.pos_fun = su.functify(pos,x)

        # Now initialize all the other bits
        SI.inputAugmentedLagrangian.__init__(self,
                                              inputFunc=inputFunc,
                                              u=u,
                                              x=x,
                                              T=T,V=V,fric=fric,
                                              cost=Cost,dt=dt,x0=x0)
                                     

    def randomizeInput(self):
        self.x0 = self.xDown+.5*randn(4)
sysPendCart = pendulumCart()

##### Prepare the controllers ####

print 'Initializing the Controllers'

T = 100
NumIter = 100
NumTrials = 5


iLQRCosts = np.zeros((NumTrials,NumIter+1))
samplingCosts = np.zeros((NumTrials,NumIter+1))


for trial in range(NumTrials):
    sysPendCart.randomizeInput()
    iLQR = SI.iterativeLQR(SYS=sysPendCart,
                         Horizon=T,
                         maxIter=NumIter,
                         label='iLQR')

    if len(iLQR.costSequence) == (NumIter+1):
        iLQRcost = iLQR.costSequence
    else:
        iLQRcost = np.zeros(NumIter+1)
        iLQRcost[len(iLQR.costSequence):] = iLQR.costSequence[-1]
    
    iLQRCosts[trial] = iLQRcost

    sampling = SI.samplingOpenLoop(SYS=sysPendCart,
                                     KLWeight = 1e-5,
                                     burnIn = NumIter,
                                     ExplorationCovariance = 100,
                                     Horizon = T,
                                     label='Sampling')

    samplingCosts[trial] = sampling.costSequence

iLQRMean = iLQRCosts.mean(axis=0)
iLQRStd = iLQRCosts.std(axis=0)
samplingMean = samplingCosts.mean(axis=0)
samplingStd = samplingCosts.std(axis=0)
plt.figure(1)
plt.clf()
pltHandles = []

handle = plt.plot(np.arange(NumIter+1),iLQRMean,
                  linewidth=2,label='iLQR')[0]
pltHandles.append(handle)
cur_color = handle.get_color()
plt.fill_between(np.arange(NumIter+1),
                 iLQRMean-iLQRStd,
                 iLQRMean+iLQRStd,
                 facecolor=cur_color,alpha=0.2)

handle = plt.plot(np.arange(NumIter+1),samplingMean,
                  linewidth=2,label='Sampling')[0]
pltHandles.append(handle)

cur_color = handle.get_color()

plt.fill_between(np.arange(NumIter+1),
                 samplingMean-samplingStd,
                 samplingMean+samplingStd,
                 facecolor=cur_color,alpha=0.2)


ymin, ymax = plt.ylim()
# plt.ylim([ymin,ymax*1.1])
plt.ylabel('Cost')
plt.xlabel('Iteration Number')
plt.legend(handles=pltHandles)

plt.savefig('costDecreaseRandomStart.pdf',transparent=True,
            bbox_inches=None,format='pdf')



