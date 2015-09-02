#### Testing the nes class with mini wam

import NewtonEulerSystem as nes
import pyuquat as uq
import sympy as sym
import numpy as np
from wam_movie import wam_movie

#### To build a NewtonEulerSystem, you need to specify
#### Number of rigid bodies
#### M matrix
#### Form of input and how that transform to generalized force
#### Constraints
#### That's it

Nlink = 2
m = np.ones(Nlink)
I = 0.1*np.ones(3*Nlink)
h = np.array([0.7,0.5,1.8,1.8,.4,.6,.4])
xstride = 13
vstride = 6

M = nes.build_M_matrix(m,I)
X,P,V,Pos,Quat,Vel,Omega = nes.states_generator(Nlink)
U = sym.symarray('U',Nlink)
# transformation matrix from U to generalized force
# this should be a constant matrix regardless of P and V
# GU = TG*U
TG = np.zeros((Nlink*vstride,Nlink),dtype=object)
# indices of axis that each joint torque applies on corresponding link
TauUind = np.zeros(Nlink,dtype=int)
for l in range(Nlink):
    TauUind[l] = 1 + ((l+1) % 2) # 1 if odd, 2 if even
for l in range(Nlink):
    if l == (Nlink-1):
        TG[vstride*l+3+TauUind[l]][l] = 1
    else:
        TG[vstride*l+3+TauUind[l]][l] = 1
        TG[vstride*l+3+TauUind[l+1]][l+1] = -1
GU = np.dot(TG,U)

#### Generate constraints functions
# Link positions
zJ = np.zeros((Nlink,3))
for k in range(Nlink):
    zJ[k] = np.array([0,0,h[k]/2.])
    
# Generate position constraints
ConstraintPos = np.zeros(3*Nlink,dtype=object)

for l in range(Nlink):
    if l == 0:
        CP = Pos[0] - zJ[l]
    else:
        CP = (Pos[l]-uq.rot(Quat[l],zJ[l])) - \
             (Pos[l-1]+uq.rot(Quat[l-1],zJ[l-1]))
        
    ConstraintPos[3*l:3+3*l] = CP

# Generate rotation constraints
ConstraintRot = np.zeros(2*Nlink,dtype=object)
for l in range(Nlink):
    if l==0:
        CR = Quat[l,1:3]
    else:
        RelQuat = uq.mult(uq.inv(Quat[l-1]),Quat[l])
        if (l % 2) == 1:
            # Odd links rotate about relative y-axis
            CR = RelQuat[[1,3]]
        elif (l % 2) == 0:
            # Even links rotate about relative z-axis
            CR = RelQuat[[1,2]]        

    ConstraintRot[2*l:2*(l+1)] = CR

Constraint = np.hstack((ConstraintPos,ConstraintRot))

# Finally creates the mini_wam object
mini_wam = nes.NewtonEulerSystems(X,U,GU,Constraint,M)

######## Clear some variables ############
# They were once symbols
# Now they are just gone
# They will come back as numbers
del X,U,GU,Constraint

######## Here begins numerical simulation ########
Nstep = 100
dt = 0.1
#### Initializing arm states for one time instance
Pos = np.zeros((Nlink,3))
Quat = np.zeros((Nlink,4))
Vel = np.zeros((Nlink,3))
Omega = np.zeros((Nlink,3))

# Quaternions and Positions should be initialized 
# more carefully. Quaternions should be initialized first
for l in range (Nlink):
    if l == 0:
        Quat[l] = uq.expq(np.array([0,0,0]))
    elif l == 1:
        Quat[l] = uq.expq(np.array([0,0.1,0]))
    else:
        Quat[l] = uq.mult(Quat[l-1],uq.expq(np.array([0,0,0])))

for l in range (Nlink):
    if l == 0:
        Pos[l] = zJ[l]
    else:
        Pos[l] = Pos[l-1] + uq.rot(Quat[l-1],zJ[l-1]) \
                          + uq.rot(Quat[l],zJ[l])
                          
# packed states of all links for one time instance
x = np.zeros(Nlink*xstride)
for l in range (Nlink):
    x[l*xstride:(l+1)*xstride] = np.hstack((Pos[l],Quat[l],Vel[l],Omega[l]))

#### Initialize packed states for all time instances
X = np.zeros((len(x),Nstep))
X[:,0] = x

#### Initialize input
U = np.zeros((Nlink,Nstep))
#U[1] = 0.02

#### Main loop ####
for k in range(Nstep-1):
    dx = mini_wam.state_diff(X[:,k],U[:,k],dt)
    X[:,k+1] = X[:,k] + dx
    
ani = wam_movie(X,dt,100)