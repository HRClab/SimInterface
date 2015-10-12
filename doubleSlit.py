import SimInterface as SI
import SimInterface.samplingControl as SC
import numpy as np
import matplotlib.pyplot as plt

class doubleSlit(POC.MarkovDecisionProcess):
    def __init__(self):
        self.dt = 0.02
        self.Horizon = int(np.round(2/self.dt))
        x0 = 0
    
        A = 1.
        B = self.dt
        R = .5 * self.dt * 1e-1
        
        dynMat = np.zeros((self.Horizon,1,3))
        costMat = np.zeros((self.Horizon,3,3))
        
        for k in range(self.Horizon):
            dynMat[k] = POC.buildDynamicsMatrix(A,B)
            if k < self.Horizon-1:
                Q = 0.
            else:
                Q = 0.5
            
            costMat[k] = POC.buildCostMatrix(Q,R)

        lqSys = POC.linearQuadraticSystem(dynMat,costMat,
                                          timeInvariant=False,x0=x0)

        # augment the cost to model the double slit.
        Slit = [(-6.,-4.),(6.,8.)]
        # Assume that the slit arises in the middle of the movement.
        SlitIndex = self.Horizon/2

        self.Slit = Slit
        self.SlitIndex = SlitIndex

            
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

slitTime = Time[sys.SlitIndex]

NumX = 10
X0 = np.linspace(-10,10,NumX)
Cost = np.zeros(NumX)

plt.figure(1)
plt.clf()
plt.plot([slitTime,slitTime],[-10,sys.Slit[0][0]],'k')
plt.plot([slitTime,slitTime],[sys.Slit[0][1],sys.Slit[1][0]],'k')
plt.plot([slitTime,slitTime],[sys.Slit[1][1],10],'k')

# Calculate appropriate noise covariance 
# dt*s = sqrt(dt) -> s = 1/sqrt(dt)

cov = 1/np.sqrt(sys.dt)

for k in range(NumX):
    print 'Initial Condition %d of %d' % (k+1,NumX)
    sys.x0 = X0[k]
    samplingCtrl = POC.samplingOpenLoop(SYS=sys,Horizon=T,
                                        KLWeight=1e-4,burnIn=1000,
                                        ExplorationCovariance = cov,
                                        label='Sampling')


    X,U,Cost[k] = sys.simulatePolicy(samplingCtrl)
    plt.plot(Time,X.squeeze())

plt.figure(2)
plt.clf()
plt.plot(X0,Cost)
