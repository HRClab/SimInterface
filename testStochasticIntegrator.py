import MarkovDecisionProcess as MDP
import Controller as ctrl
import numpy as np
import matplotlib.pyplot as plt


class stochasticIntegrator(MDP.linearQuadraticStochasticSystem):
    def __init__(self):
        self.dt = 0.1
        x0 = 1.0
        A = 1.
        B = self.dt
        Q = self.dt * 1.
        R = self.dt * 1.
        W = np.sqrt(self.dt) * .1
        noiseMat = np.array([[W]])
        dynMat = MDP.buildDynamicsMatrix(A,B)
        costMat = MDP.buildCostMatrix(Cxx=Q,Cuu=R)

        MDP.linearQuadraticStochasticSystem.__init__(self,
                                                     dynMat,
                                                     costMat,
                                                     noiseMat,
                                                     x0=x0)

sys = stochasticIntegrator()

T = 100

Controllers = []

staticCtrl = ctrl.staticGain(gain=-.5,Horizon=T,label='Static')
Controllers.append(staticCtrl)

lqrCtrl = ctrl.linearQuadraticRegulator(SYS=sys,Horizon=T,label='LQR')
Controllers.append(lqrCtrl)

samplingCtrl = ctrl.stochasticSamplingControl(SYS=sys,
                                              NumSamples = 1,
                                              Horizon=T,
                                              KLWeight=1e-4,burnIn=500,
                                              ExplorationCovariance=3.,
                                              label='Sampling')

Controllers.append(samplingCtrl)

NumControllers = len(Controllers)
XMean = np.zeros((NumControllers,T,1))
XStd = np.zeros((NumControllers,T,1))
CostMean = np.zeros(NumControllers)
CostStd = np.zeros(NumControllers)
Time = sys.dt * np.arange(T)
NumRuns = 10
fig = plt.figure(1)
plt.clf()
line = []

print '\nComparing Controllers\n'

for k in range(NumControllers):
    controller = Controllers[k]
    name = controller.label
    X = np.zeros((NumRuns,T,1))
    Cost = np.zeros(NumRuns)
    for run in range(NumRuns):
        X[run], U, Cost[run] = sys.simulatePolicy(controller)
        
    XMean[k] = X.mean(axis=0)
    XStd[k] = X.std(axis=0)
    CostMean[k] = Cost.mean()
    CostStd[k] = Cost.std()
    print '%s mean: %g, std: %g' % (name,CostMean[k],CostStd[k])
    handle = plt.plot(Time,XMean[k],label=name)[0]
    cur_color = handle.get_color()
    plt.fill_between(Time,
                     (XMean[k]-XStd[k]).squeeze(),
                     (XMean[k]+XStd[k]).squeeze(),
                     facecolor=cur_color,alpha=0.2)
    line.append(handle)

plt.legend(handles=line)

