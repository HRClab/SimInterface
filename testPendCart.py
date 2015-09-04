import MarkovDecisionProcess as MDP
import Controller as ctrl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sympy as sym
import sympy_utils as su


#### Define the pendulum on a cart system ####

class cartPole(MDP.inputAugmentedLagrangian):
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

        Cost = dt * (EnergyCost + targetCost + speedCost)

        # Input function, since only the cart is actuated
        inputFunc = np.array([u,0])

        # Position function for simpler plotting
        pos = np.zeros((2,2),dtype=object)
        pos[0] = xCart
        pos[1] = xPole
        self.pos_fun = su.functify(pos,x)

        # Now initialize all the other bits
        MDP.inputAugmentedLagrangian.__init__(self,
                                              inputFunc=inputFunc,
                                              u=u,
                                              x=x,
                                              T=T,V=V,fric=fric,
                                              cost=Cost,dt=dt,x0=x0)
                                     

sysCartPole = cartPole()

##### Prepare the controllers ####

print 'Initializing the Controllers'

T = 100
Controllers = []



iLQR = ctrl.iterativeLQR(SYS=sysCartPole,
                         Horizon=T,
                         stoppingTolerance=1e-2,
                         label='iLQR')

Controllers.append(iLQR)

sampling = ctrl.samplingOpenLoop(SYS=sysCartPole,
                                KLWeight = 1e-5,
                                burnIn = 100,
                                ExplorationCovariance = 100,
                                Horizon = T,
                                label='Sampling')

Controllers.append(sampling)

samplingIlqr = ctrl.iterativeLQR(SYS=sysCartPole,
                                 Horizon=T,
                                 stoppingTolerance=1e-2,
                                 initialPolicy=sampling,
                                 label='Sampling->iLQR')

Controllers.append(samplingIlqr)

samplingIlqrSampling = ctrl.samplingOpenLoop(SYS=sysCartPole,
                                            Horizon=T,
                                            KLWeight = 1e-5,
                                            burnIn = 100,
                                            ExplorationCovariance=100,
                                            initialPolicy = samplingIlqr,
                                            label='Sampling->iLQR->Sampling')

Controllers.append(samplingIlqrSampling)

##### Simulate all controllers on system #####
print 'Simulating the system with different controllers'
NumControllers = len(Controllers)
X = np.zeros((NumControllers,T,sysCartPole.NumStates))
U = np.zeros((NumControllers,T,sysCartPole.NumInputs))
Cost = np.zeros(NumControllers)

Time = sysCartPole.dt * np.arange(T)

for k in range(NumControllers):
    controller = Controllers[k]
    name = controller.label
    X[k],U[k],Cost[k] = sysCartPole.simulatePolicy(controller)
    print '%s: %g' % (name,Cost[k])


def movie():
    # draw the cart
    h = .2*sysCartPole.lPole
    w = .7*sysCartPole.lPole
    Cart = np.array([[-w/2,w/2,w/2,-w/2,-w/2],
                     [-h,   -h,  0,  0,  -h]])
    
    fig = plt.figure(1)
    plt.clf()

    linePend = []
    lineCart = []
    lineTarget = []

    halfWidth = 6*sysCartPole.lPole
    yBottom = -1.2 * sysCartPole.lPole
    yTop = 1.2 * sysCartPole.lPole
    
    for k in range(NumControllers):
        ax = fig.add_subplot(4,1,k+1,autoscale_on=False,aspect=1,
                             xlim = (-halfWidth,halfWidth),
                             ylim = (yBottom,yTop))

        ax.set_xticks([])
        ax.set_yticks([])
        lineCart.append(ax.plot([],[],lw=2)[0])
        linePend.append(ax.plot([],[],lw=2)[0])
        lineTarget.append(ax.plot([],[],'m*')[0])
        plt.title(Controllers[k].label)

    def init():
        for k in range(NumControllers):
            lineCart[k].set_data([],[])
            linePend[k].set_data([],[])
            lineTarget[k].set_data(sysCartPole.target[0],sysCartPole.target[1])
        return lineCart,linePend,lineTarget
    def animate(k):
        for j in range(NumControllers):
            pos = sysCartPole.pos_fun(X[j][k])
            posCart = pos[0]
            posPend = pos[1]
            lineCart[j].set_data(posCart[0]+Cart[0],
                                 posCart[1]+Cart[1])

            linePend[j].set_data([posCart[0],posPend[0]],
                                [posCart[1],posPend[1]])
            
        return lineCart,linePend,lineTarget

    ani = animation.FuncAnimation(fig,animate,T,
                                  blit=False,init_func=init,
                                  interval=sysCartPole.dt*1000,
                                  repeat=False)

    return ani

ani = movie()
    
