import numpy as np
from numpy.random import randn
from bovy_mcmc.elliptical_slice import elliptical_slice as eslice
import UnderactuatedPendulum as UAP

dt = 0.05
sys = UAP.UnderactuatedPendulum(dt = dt)

def loglikelihood(W,lam=1):
    cost = sys.openLoopSim(W)[1]
    return -cost / lam

# Now set up the simluation 
NStep = int(round(2/dt))
T = dt * np.arange(NStep+1)

#Initial conditions
U = np.zeros(NStep)

lam = 0.00001
likelihoodParam = (lam,)
logLik = loglikelihood(U,*likelihoodParam)

NSamp = 200

bestCost = np.inf

for samp in range(NSamp):
    W = 40.*randn(NStep)
    U,logLik = eslice(U,W,loglikelihood,likelihoodParam,logLik)
    cost = -logLik * lam
    if cost < bestCost:
        bestCost = cost
        bestU = U
    if (samp+1) % 10 == 0:
        print 'run %d of %d, Cost: %g, Best Cost: %g' % \
            (samp+1,NSamp,cost,bestCost)

X,cost = sys.openLoopSim(bestU)
ani = sys.movie(X)
