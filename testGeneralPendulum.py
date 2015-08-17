import MarkovDecisionProcess as MDP
import Controller as ctrl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sympy as sym
import sympy_utils as su

#### Define the sysGenPendtem ####


class pendulum(MDP.LagrangianSystem):
    """
    A planar pendulum with an arbitrary number of links
    """
    def __init__(self):
        dt = 0.05
        n = 3
        self.NumLinks = n
        self.Mass = np.ones(n) 
        self.Length = np.ones(n) 
        g = 10.

        q = sym.symarray('q',n)
        dq = sym.symarray('dq',n)
        x = np.hstack((q,dq))
        u = sym.symarray('u',n)

        target = np.array([0,0.8 * self.Length.sum()])
        x0 = np.zeros(2*n)
        x0[0] = -np.pi / 2 + .1
        self.target = target
        # cartesian position
    
        pos = np.zeros((2,n),dtype=object)
        curPos = np.zeros(2,dtype=object)
        theta = q.cumsum()

        for i in range(n):
            curPos = curPos + self.Length[i] * \
                     np.array([sym.cos(theta[i]),sym.sin(theta[i])])
            pos[:,i] = curPos

        # cartesian velocity
        vel = np.zeros((2,n),dtype=object)

        for i in range(n):
            vel[:,i] = np.dot(su.jacobian(pos[:,i],q),dq)

        # Kinetic and potential energy
        T = 0
        V = 0

        for i in range(n):
            T = T + .5 * self.Mass[i] * sym.simplify(np.dot(vel[:,i],vel[:,i]))
            V = V + self.Mass[i] * g * pos[1,i]

        # Friction function
        fric = 1 * dq

        # Cost
        EnergyCost = np.dot(u,u)
        targetError = target - pos[:,-1]
        targetCost = 100000 * np.dot(targetError,targetError)
        speedCost = 10 * np.dot(dq,dq)

        Cost = dt * (EnergyCost + targetCost + speedCost)

        # Tip Jacobian
        jac = su.jacobian(pos[:,-1],q)
        self.tip_jacobian = su.functify(jac,x)

        # Position Function for Plotting
        self.pos_fun = su.functify(pos,x)
        
        MDP.LagrangianSystem.__init__(self,T,V,fric,Cost,x,u,dt,x0)


if not 'sysGenPend' in locals():
    print 'Building the System'
    sysGenPend = pendulum()

#### Build the Controllers ####

def impedanceCtrlFunc(x):
    OrigPotentialForce = -sysGenPend.phi_fun(x)
    F = -OrigPotentialForce
    K = 20
    SpringForce = K * (sysGenPend.target-sysGenPend.pos_fun(x)[:,-1])
    Jac = sysGenPend.tip_jacobian(x)
    SpringInput = np.dot(Jac.T,SpringForce)
    F += SpringInput
    C = 10
    FrictionForce = -C * np.dot(Jac,x[sysGenPend.NumInputs:])
    FrictionInput = np.dot(Jac.T,FrictionForce)
    F += FrictionInput
    return F
    

print 'Initializing the Controllers'

T = 50
Controllers = []

Controllers.append(ctrl.staticGain(gain=np.zeros((sysGenPend.NumInputs,sysGenPend.NumStates)),
                                   Horizon=T,label='Open Loop'))

impedanceCtrl = ctrl.staticFunction(func=impedanceCtrlFunc,Horizon=T,
                                    label='Impedance Control')
Controllers.append(impedanceCtrl)

mpcCtrl = ctrl.modelPredictiveControl(SYS=sysGenPend,
                                      Horizon=T,
                                      predictiveHorizon=10,
                                      label='MPC')
Controllers.append(mpcCtrl)

Controllers.append(ctrl.samplingControl(SYS=sysGenPend,
                                        Horizon=T,
                                        KLWeight=1e-5,
                                        burnIn=10000,
                                        ExplorationCovariance=25.*\
                                        np.eye(sysGenPend.NumInputs),
                                        initialPolicy = impedanceCtrl,
                                        label='Sampling'))

#### Prepare the simulations ####
print 'Simulating the system with the different controllers'
NumControllers = len(Controllers)
X = np.zeros((NumControllers,T+1,sysGenPend.NumStates))
U = np.zeros((NumControllers,T,sysGenPend.NumInputs))
Cost = np.zeros(NumControllers)
Time = sysGenPend.dt * np.arange(T+1)
fig = plt.figure(1)
plt.clf()
line = []
lineTarget = []
MaxLen = sysGenPend.Length.sum()

#### Simulate all of the controllers #### 

for k in range(NumControllers):
    controller = Controllers[k]
    name = controller.label
    X[k], U[k], Cost[k] = sysGenPend.simulatePolicy(controller)
    print '%s: %g' % (name,Cost[k])

    ax = fig.add_subplot(2,2,k+1,autoscale_on=False, aspect = 1,
                         xlim=(-MaxLen,MaxLen),ylim=(-MaxLen,MaxLen))
                        
    line.append(ax.plot([],[],lw=2)[0])
    lineTarget.append(ax.plot([],[],'r*')[0])
    plt.title(name)

#### Play a movie of the controllers in action ####
print 'Playing Movie'

def movie():
    Joints = np.zeros((NumControllers,2,sysGenPend.NumLinks+1))
    
    def init():
        for k in range(NumControllers):
            line[k].set_data([],[])
            lineTarget[k].set_data(sysGenPend.target[0],sysGenPend.target[1])
        return line, lineTarget

    def animate(k):
        for c in range(NumControllers):
            x = X[c][k]
            Joints[c,:,1:] = sysGenPend.pos_fun(x)
            line[c].set_data(Joints[c][0],Joints[c][1])
        return line, lineTarget

    ani = animation.FuncAnimation(fig,animate,X.shape[1],
                                  blit = False, init_func=init,
                                  interval=sysGenPend.dt*1000,repeat=False)
    return ani

ani = movie()
