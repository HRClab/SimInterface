import Integrator as Int
import Controller as ctrl
import numpy as np
import matplotlib.pyplot as plt


sys = Int.Integrator()


staticCtrl = ctrl.staticGain(gain=-.5,Horizon=100)

X,cost = sys.simulatePolicy(staticCtrl)

T = sys.dt * np.arange(staticCtrl.Horizon+1)

plt.figure(1)
plt.clf()
plt.plot(T,X)
