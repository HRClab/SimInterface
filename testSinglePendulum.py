import MarkovDecisionProcess as MDP
import Controller as ctrl
import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
import sympy_utils as su

#### Define the system ####

class pendulum(MDP.LagrangianSystem):
    """
    A single Pendulum
    """
    def __init__(self):
        m = 1.
        g = 10.
        fricCoef = 1.
        q, dq, u = sym.symbols('q dq u')
        x = np.array([q,dq])
        T = .5 * m * dq * dq
        V = m * g * sym.sin(q)
        fric = fricCoef * dq
        dt = 0.1
        # Must start a bit away from equilibrium
        # Otherwise some controllers, such as the artificial
        # potential function, have problems
        x0 = np.array([0,0])
        cost = dt * (u*u + 100 * (1 - sym.sin(q)))
        MDP.LagrangianSystem.__init__(self,T,V,fric,cost,x,u,dt,x0)

        
sys = pendulum()

#### Make a function for artificial potential control ####

def artificialPotential(x):
    OriginalPotentialForce = -sys.phi_fun(x)
    ArtificialPotentialForce = 10*np.cos(x[0])
    Damping = -1 * x[1]
    return ArtificialPotentialForce - OriginalPotentialForce \
        + Damping

#### Make a List of Controllers #####

T = 100
Controllers = []

Controllers.append(ctrl.staticFunction(func=artificialPotential,
                                       Horizon=T,
                                       label = 'Potential'))
Controllers.append(ctrl.samplingControl(SYS=sys,Horizon=T,
                                        KLWeight=1e-5,burnIn=200,
                                        ExplorationCovariance = 20.,
                                        label='Sampling'))
Controllers.append(ctrl.modelPredictiveControl(SYS=sys,Horizon=T,
                                               predictiveHorizon=10,
                                               label='MPC'))
#### Prepare the simulations ####
NumControllers = len(Controllers)
X = np.zeros((NumControllers,T+1,2))
Cost = np.zeros(NumControllers)
Time = sys.dt * np.arange(T+1)
plt.figure(1)
plt.clf()
line = []

print '\nComparing Controllers\n'

#### Simulate all of the controllers ####

for k in range(NumControllers):
    controller = Controllers[k]
    name = controller.label
    X[k], Cost[k] = sys.simulatePolicy(controller)
    print '%s: %g' % (name,Cost[k])
    handle = plt.plot(Time,np.sin(X[k][:,0]),label=name)[0]
    line.append(handle)

plt.legend(handles=line)
