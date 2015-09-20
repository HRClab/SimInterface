import pyopticon as POC
import pyopticon.samplingControl as SC
import numpy as np
import matplotlib.pyplot as plt


class Integrator(POC.linearQuadraticSystem):
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
        dynMat = POC.buildDynamicsMatrix(A,B)
        costMat = POC.buildCostMatrix(Cxx=Q,Cuu=R)
        POC.linearQuadraticSystem.__init__(self,dynamicsMatrix=dynMat,
                                           costMatrix=costMat,x0 = self.x0)
sys = Integrator()

T = 50

Controllers = []
staticCtrl = POC.staticGain(gain=-.5,Horizon=T,label='Static')
Controllers.append(staticCtrl)

lqrCtrl = POC.linearQuadraticRegulator(SYS=sys,Horizon=T,label='LQR')
Controllers.append(lqrCtrl)

# mpcCtrl = POC.modelPredictiveControl(SYS=sys,
#                                       predictiveHorizon=5,
#                                       Horizon=T,
#                                       label='MPC')
# Controllers.append(mpcCtrl)

# samplingCtrl = POC.samplingOpenLoop(SYS=sys,Horizon=T,
#                                     KLWeight=1e-4,burnIn=100,
#                                     ExplorationCovariance = 1,
#                                     label='Sampling')
# Controllers.append(samplingCtrl)

# gibbsCtrl = SC.gibbsOpenLoop(SYS=sys,
#                              Horizon=T,
#                              KLWeight=1e-2,
#                              burnIn=1000,
#                              stepBurnIn=1,
#                              InputCovariance=2,
#                              StateCovariance=1e-4,
#                              label='Gibbs')

# Controllers.append(gibbsCtrl)


def affineInput(x,u,k=0):
    """
    Affine Scalar Feedback
    """
    vec = np.hstack([1,x])
    return np.dot(vec,u)

AugSys = POC.augmentInput(sys,affineInput,2)
affineGibbsCtrl = SC.gibbsOpenLoop(SYS=AugSys,
                                   Horizon=T,
                                   KLWeight=1e-2,
                                   burnIn=1000,
                                   stepBurnIn=1,
                                   InputCovariance=2,
                                   StateCovariance=1e-4,
                                   label='Affine Gibbs')

Controllers.append(affineGibbsCtrl)

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
    if k < NumControllers-1:
        X[k], U, Cost[k] = sys.simulatePolicy(controller)
    else:
        X[k], U, Cost[k] = AugSys.simulatePolicy(controller)
    print '%s: %g' % (name,Cost[k])
    handle = plt.plot(T,X[k],label=name)[0]
    line.append(handle)

plt.legend(handles=line)