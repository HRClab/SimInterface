import pyopticon as POC
import pyopticon.samplingControl as SC
import numpy as np
import matplotlib.pyplot as plt

class doubleSlit(POC.MarkovDecisionProcess):
    def __init__(self):
        self.dt = 0.01
        self.Horizon = int(np.round(1/self.dt))
        x0 = 0
    
        A = 1.
        B = self.dt
        R = self.dt * 1e-7
        
        dynMat = np.zeros((self.Horizon,1,3))
        costMat = np.zeros((self.Horizon,3,3))
        
        for k in range(self.Horizon):
            dynMat[k] = POC.buildDynamicsMatrix(A,B)
            if k < self.Horizon-1:
                Q = 0.
            else:
                Q = 1.
            
            costMat[k] = POC.buildCostMatrix(Q,R)

        lqSys = POC.linearQuadraticSystem(dynMat,costMat,
                                          timeInvariant=False,x0=x0)

            
        # augment the cost to model the double slit.
        Slit = [(-5.,-4.),(6.,8.)]
        # Assume that the slit arises in the middle of the movement.
        SlitIndex = self.Horizon/2

            
        def augmentedCost(x,u,k):
            cost = lqSys.costStep(x,u,k)
            if k == SlitIndex:
                slitCost = 1000
                for slit in Slit:
                    if (slit[0] < x) and (slit[1] > x):
                        slitCost = 0.
                cost += slitCost
            return cost

        POC.MarkovDecisionProcess.__init__(self,x0=x0)

        self.costStep = augmentedCost
        self.step = lqSys.step

sys = doubleSlit()

T = sys.Horizon
Time = sys.dt * np.arange(T)

samplingCtrl = POC.samplingOpenLoop(SYS=sys,Horizon=T,
                                    KLWeight=1e-4,burnIn=1000,
                                    ExplorationCovariance = 2,
                                    label='Sampling')


X,U,cost = sys.simulatePolicy(samplingCtrl)

plt.figure(1)
plt.clf()
plt.plot(Time,X.squeeze())
