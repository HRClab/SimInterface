import Integrator as Int
import numpy as np
import matplotlib.pyplot as plt


sys = Int.Integrator()

class controller:
    def __init__(self,Horizon=100):
        self.Horizon = Horizon

    def action(self,x,k):
        return -.5*x

ctrl = controller()

X,cost = sys.simulatePolicy(ctrl)

T = sys.dt * np.arange(ctrl.Horizon+1)

plt.figure(1)
plt.clf()
plt.plot(T,X)
