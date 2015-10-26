import SimInterface as SI
import numpy as np
from numpy.random import randn
import matplotlib.pyplot as plt


def buildSystems():
        dt = 0.1
        x0 = 1.0
        A = 1.
        B = dt
        Q = dt * 1.
        R = dt * 1.
        W = np.sqrt(dt) * .1
        noiseMat = np.array([[W]])
        dynMat = SI.buildDynamicsMatrix(A,B)
        costMat = SI.buildCostMatrix(Cxx=Q,Cuu=R)

        sys = SI.linearQuadraticStochasticSystem(dynMat,
                                                  costMat,
                                                  noiseMat,
                                                  x0=x0)

        sysDet = SI.linearQuadraticSystem(dynMat,
                                           costMat,
                                           x0=x0)

        return sys, sysDet, dt

sys, sysDet, dt = buildSystems()

T = 100

Controllers = []

staticCtrl = SI.staticGain(gain=-.5,Horizon=T,label='Static')
Controllers.append(staticCtrl)

lqrCtrl = SI.linearQuadraticRegulator(SYS=sys,Horizon=T,label='LQR')
Controllers.append(lqrCtrl)

deterministicSampling = SI.samplingOpenLoop(SYS=sysDet,
                                              Horizon=T,
                                              KLWeight=1e-4,
                                              burnIn=100,
                                              ExplorationCovariance=3)

sampleMPCctrl = SI.samplingMPC(SYS=sysDet,
                                 Horizon=T,
                                 KLWeight=1e-4,
                                 ExplorationCovariance=3.,
                                 PredictionHorizon=10,
                                 PredictionBurnIn=3,
                                 initialPolicy=deterministicSampling,
                                 label='Sampling MPC')

Controllers.append(sampleMPCctrl)

NumControllers = len(Controllers)
XMean = np.zeros((NumControllers,T,1))
XStd = np.zeros((NumControllers,T,1))
CostMean = np.zeros(NumControllers)
CostStd = np.zeros(NumControllers)
Time = dt * np.arange(T)
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
        X[run], U, CostSeq = sys.simulatePolicy(controller)
        Cost[k] = CostSeq.sum()
        
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

print '\nTesting Mean Cost Predictions\n'
# Testing 
gain = randn(T*2)

randomPolicy = SI.flatVaryingAffine(gain,1,T,label='Random')


cost = sys.simulatePolicy(randomPolicy)[2]
