import sympy as sym
import numpy as np
import sympy_utils as su
import pylagrange as lag


n = 2
q = sym.symarray('q',n)
dq = sym.symarray('dq',n)

x = np.hstack((q,dq))

# cartesian position
pos = np.zeros((2,n),dtype=object)
curPos = np.zeros(2,dtype=object)
theta = q.cumsum()

for i in range(n):
    curPos = curPos + np.array([sym.cos(theta[i]),sym.sin(theta[i])])
    pos[:,i] = curPos

# cartesian velocity
vel = np.zeros((2,n),dtype=object)

for i in range(n):
    vel[:,i] = np.dot(su.jacobian(pos[:,i],q),dq)


# Kinetic and potential energy
T = 0
V = 0

for i in range(n):
    T = T + .5 * sym.simplify(np.dot(vel[:,i],vel[:,i]))
    V = V + 10 * pos[1,i]

# friction
fric = 1 * dq

SYS = lag.lagrangian_system(T,V,fric,x)
lag.save(SYS,'dPend')
# This is useful for plotting
su.sympy_save(pos,x,'pos_fun')

# Now let us make an error cost function
target = sym.symarray('target',2)
targetError = target - pos[:,-1]

targetCost = np.dot(targetError,targetError)
targetCost_jac = su.jacobian(targetCost,x)
targetCost_hes = su.jacobian(targetCost_jac,x)

su.sympy_save(targetCost,(x,target),'targetCost')
su.sympy_save(targetCost_jac,(x,target),'targetCost_jac')
su.sympy_save(targetCost_hes,(x,target),'targetCost_hes')
