import SimInterface as SI
import numpy as np
import matplotlib.pyplot as plt


class Integrator(SI.linearQuadraticSystem):
    """
    This is the simplest linear system dynamical system
    """
    def __init__(self):
        self.dt = 0.1
        self.x0 = 1.0
        A = 1.
        B = self.dt
        Q = self.dt * 1.
        R = self.dt * 1.
        dynMat = SI.buildDynamicsMatrix(A,B)
        costMat = SI.buildCostMatrix(Cxx=Q,Cuu=R)
        SI.linearQuadraticSystem.__init__(self,dynamicsMatrix=dynMat,
                                           costMatrix=costMat,x0 = self.x0)
sys = Integrator()

T = 50

Controllers = []
staticCtrl = SI.staticGain(gain=-.5,Horizon=T,label='Static')
Controllers.append(staticCtrl)

lqrCtrl = SI.linearQuadraticRegulator(SYS=sys,Horizon=T,label='LQR')
Controllers.append(lqrCtrl)

mpcCtrl = SI.modelPredictiveControl(SYS=sys,
                                      predictiveHorizon=5,
                                      Horizon=T,
                                      label='MPC')
Controllers.append(mpcCtrl)


NumControllers = len(Controllers)
X = np.zeros((NumControllers,T,1))
Cost = np.zeros(NumControllers)
T = sys.dt * np.arange(staticCtrl.Horizon)
plt.figure(1)
plt.clf()
line = []

print '\nComparing Controllers\n'

for k in range(NumControllers):
    controller = Controllers[k]
    name = controller.label

    X[k], U, Cost[k] = sys.simulatePolicy(controller)
    print '%s: %g' % (name,Cost[k])
    handle = plt.plot(T,X[k],label=name)[0]
    line.append(handle)

plt.legend(handles=line)
