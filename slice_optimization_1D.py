"""
This example shows how to use elliptical slice sampling in order to
approximately optimize a non-convex function. 
"""

import numpy as np
from numpy.random import randn
from scipy.linalg import solve
import matplotlib.pyplot as plt
from bovy_mcmc.elliptical_slice import elliptical_slice as eslice
import utils.sympy_utils as su

# Pick some coefficients for a nice looking polynomial
M = np.array([[1,1,1],[4,16,64],[2,16,3*32]])
y = np.array([1,-1,0])
coeff = solve(M,y)

def cost(x):
    return  1+coeff[0]*x**2. + coeff[1]*x**4. + coeff[2]*x**6.

def loglikelihood(x,lam=1):
    """
    
    """
    return -cost(x)/lam

# Plot the cost function
X = np.linspace(-2.4,2.4,100)
Y = cost(X)

plt.figure(1)
plt.clf()
plt.plot(X,Y)
plt.xticks(np.linspace(-3,3,7))
plt.xlabel('x')
plt.ylabel('cost')


# Now do the optimization for various values of the regularization
# parameter, lambda 
BurnIn = 100
NSamp = BurnIn+100

fig = plt.figure(2)
plt.clf()

overlay = plt.figure(3)
plt.clf()

NLam = 3
Lam = np.logspace(0,-2,NLam)
Lam = [2,0.3,0.01]

for run in range(NLam):
    lam = Lam[run]
    W = randn(NSamp,1)
    theta = randn(1,)

    Theta = np.zeros(NSamp)
    logLik = loglikelihood(theta,lam)


    for i in range(NSamp):
        w = W[i]
        theta, logLik = eslice(theta,w,loglikelihood,
                               (lam,),logLik)

        Theta[i] = theta

    Cost = cost(Theta)
    aveCost = np.mean(Cost[BurnIn:])
    plt.figure(2)
    plt.subplot(NLam,1,run+1)
    plt.hist(Theta[BurnIn:],30)
    plt.title('$\lambda$=%g, Average Cost: %g' % (lam,aveCost))
    ymin,ymax = plt.ylim()
    plt.xlim(-2.5,2.5)
    plt.yticks([0,ymax])
    if run < NLam-1:
        plt.xticks([])
    else:
        plt.xticks(np.linspace(-2,2,5))
        plt.xlabel('x')

    plt.figure(3)
    plt.subplot(NLam,1,run+1)
    plt.plot(X,Y)
    plt.plot(Theta[-100:],Cost[-100:],'*')
    plt.title('$\lambda$=%g, Average Cost: %g' % (lam,aveCost))
    plt.ylim([-.3,3.5])
    plt.yticks(np.arange(4))
    plt.ylabel('cost')
    if run < NLam-1:
        plt.xticks([])
    else:
        plt.xticks(np.linspace(-3,3,7))
        plt.xlabel('x')
                   
fig.subplots_adjust(hspace=.5)
overlay.subplots_adjust(hspace=.5)

plt.figure(3)
plt.savefig('costSampleOverlay.pdf',transparent=True,bbox_inches=None,format='pdf')
