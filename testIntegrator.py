import Integrator as Int
import Controller as ctrl
import numpy as np
import matplotlib.pyplot as plt


sys = Int.Integrator()


staticCtrl = ctrl.staticGain(gain=-.5,Horizon=100)
lqrCtrl = ctrl.linearQuadraticRegulator(SYS=sys,Horizon=100)
mpcCtrl = ctrl.modelPredictiveControl(SYS=sys,
                                      predictiveHorizon=10,
                                      Horizon=100)

XStatic,costStatic = sys.simulatePolicy(staticCtrl)
XLQR, costLQR = sys.simulatePolicy(lqrCtrl)
XMPC, costMPC = sys.simulatePolicy(mpcCtrl)

T = sys.dt * np.arange(staticCtrl.Horizon+1)

plt.figure(1)
plt.clf()
plt.plot(T,XStatic)
plt.plot(T,XLQR)
plt.plot(T,XMPC)
