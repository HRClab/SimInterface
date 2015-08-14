import Integrator as Int
import Controller as ctrl
import numpy as np
import matplotlib.pyplot as plt


sys = Int.Integrator()

class staticGain(ctrl.Controller):
    def __init__(self,Horizon=0,gain=0):
        self.Horizon = Horizon
        self.gain = gain

    def action(self,x,k):
        return self.gain*x

staticCtrl = staticGain(Horizon=100,gain=-.5)

X,cost = sys.simulatePolicy(staticCtrl)

T = sys.dt * np.arange(staticCtrl.Horizon+1)

plt.figure(1)
plt.clf()
plt.plot(T,X)
