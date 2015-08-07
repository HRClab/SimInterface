import numpy as np
from numpy.random import randn
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from bovy_mcmc.elliptical_slice import elliptical_slice as eslice
import sympy_utils as su
import pylagrange as lag

dPend = lag.load('dPend')
pos_fun = su.sympy_load('pos_fun')

x0 = np.zeros(4)
x0[0] = -np.pi/2.
target_pos = np.array([0.0,2.0])

def cost_step(x,u):
    tip_pos = pos_fun(x)[:,-1]
    tip_error = tip_pos - target_pos

    q = 1000.0
    r = 0.00001

    cost = q * np.dot(tip_error,tip_error) + r * np.dot(u,u)
    return cost

def rollout(W=[],x=x0,dt=0.01):
    n = len(x)
    nU = 2
    NStep = len(W)/nU

    X = np.zeros((n,NStep+1))
    X[:,0] = x

    cost = 0
    
    for step in range(NStep):
        w = W[nU*step:nU*(step+1)]
        x = dPend.step(x,w,dt,lag.EULER)
        X[:,step+1] = x

        cost = cost + cost_step(x,w)*dt

    return X, cost

def cost_total(W,x,dt):
    # Pick off the cost term from the rollout
    cost = rollout(W,x,dt)[1]
    return cost

def loglikelihood(W,x,dt,lam=1):
    return -cost_total(W,x,dt)/lam


# Now set up the simluation 
dt = 0.05
NStep = int(round(2/dt))
T = dt * np.arange(NStep+1)

#Initial conditions
U = np.zeros(2*NStep)

lam = 0.00001
likelihoodParam = (x0,dt,lam)
logLik = loglikelihood(U,*likelihoodParam)

NSamp = 200

for samp in range(NSamp):
    W = 40.*randn(2*NStep)
    U,logLik = eslice(U,W,loglikelihood,likelihoodParam,logLik)
    cost = -logLik * lam
    if (samp+1) % 10 == 0:
        print 'run %d of %d: Cost: %g' % (samp+1,NSamp,cost)

X,cost = rollout(U,x0,dt)

fig = plt.figure(1)
plt.clf()
ax = fig.add_subplot(111, autoscale_on=False, aspect=1,
                     xlim=(-2.1,2.1), ylim=(-2.1,2.1))

line = ax.plot([],[],lw=2)[0]
lineTarget = ax.plot([],[],'r*')[0]

Joints = np.zeros((2,3))

def init():
    line.set_data([],[])
    lineTarget.set_data(target_pos[0],target_pos[1])
    return line,lineTarget

def animate(k):
    x = X[:,k]
    Joints[:,1:] = pos_fun(x)
    line.set_data(Joints[0],Joints[1])
    return line,lineTarget

ani = animation.FuncAnimation(fig,animate, range(NStep+1), interval=dt*1000,
                              blit=False,init_func=init,repeat=False)

