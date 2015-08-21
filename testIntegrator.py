import MarkovDecisionProcess as MDP
import Controller as ctrl
import numpy as np
import matplotlib.pyplot as plt


class Integrator(MDP.LinearQuadraticSystem):
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
        dynMat = MDP.buildDynamicsMatrix(A,B)
        costMat = MDP.buildCostMatrix(Cxx=Q,Cuu=R)
        MDP.LinearQuadraticSystem.__init__(self,dynamicsMatrix=dynMat,
                                           costMatrix=costMat,x0 = self.x0)
sys = Integrator()

T = 100

Controllers = []
staticCtrl = ctrl.staticGain(gain=-.5,Horizon=T,label='Static')
Controllers.append(staticCtrl)

lqrCtrl = ctrl.linearQuadraticRegulator(SYS=sys,Horizon=T,label='LQR')
Controllers.append(lqrCtrl)

mpcCtrl = ctrl.modelPredictiveControl(SYS=sys,
                                      predictiveHorizon=10,
                                      Horizon=T,
                                      label='MPC')
Controllers.append(mpcCtrl)

samplingCtrl = ctrl.samplingControl(SYS=sys,Horizon=T,
                                    KLWeight=1e-5,burnIn=500,
                                    ExplorationCovariance = 3.,
                                    label='Sampling')
Controllers.append(samplingCtrl)

NumControllers = len(Controllers)
X = np.zeros((NumControllers,T,1))
U = np.zeros((NumControllers,T,1))
Cost = np.zeros(NumControllers)
T = sys.dt * np.arange(staticCtrl.Horizon)
plt.figure(1)
plt.clf()
line = []

print '\nComparing Controllers\n'

for k in range(NumControllers):
    controller = Controllers[k]
    name = controller.label
    X[k], U[k], Cost[k] = sys.simulatePolicy(controller)
    print '%s: %g' % (name,Cost[k])
    handle = plt.plot(T,X[k],label=name)[0]
    line.append(handle)

plt.legend(handles=line)
