#### Testing the nes class with mini wam

import NewtonEulerSystem as nes
import MarkovDecisionProcess as MDP
import pyuquat as uq
import sympy as sym
import numpy as np
from wam_movie import wam_movie

class RobotArm_NE(MDP.NewtonEulerSys):
    # This class creates robot arms similar to the Barret Wam arm
    # objects differ from each other by their number of links, mass of links,
    # rotational interia of links and length of links
    
    # To build a MDP.NewtonEulerSys, you need to specify
    # Number of rigid bodies
    # m, I arrays
    # Form of input and how that transform to generalized force
    # friction or any states dependent wrench
    # Constraints    
    def __init__(m,I,h):
        self.m = m
        self.I = I
        self.h = h
        self.Nlink = len(m)



#### Create a mini robot arm
Nlink = 3
xstride = 13
pstride = 7
vstride = 6
m = np.ones(Nlink)
I = 0.1*np.ones(3*Nlink)
h = np.array([0.7,0.5,1.8,1.8,.4,.6,.4])

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

## Friction
# friction on each joint as a Nlink*1 array
fricCoef = 0.1
fric_Nlink = np.zeros((Nlink),dtype=object)
# friction as generalized force
Ffric = np.zeros((Nlink*vstride),dtype=object)

for l in range(Nlink):
    if l == 0:
        fric_Nlink[l] = fricCoef*Omega[l][TauUind[l]]
    else:
        # Relative quaternion that transform current link's body frame to 
        # that of the previous link
        QuatRel = uq.mult(uq.inv(Quat[l-1]),Quat[l])
        # Relative omega w.r.t. the previous link
        OmegaRel = Omega[l] - uq.rot(uq.inv(QuatRel),Omega[l-1])
        # Friction
        fric_Nlink[l] = fricCoef*OmegaRel[TauUind[l]]

Ffric = - np.dot(TG,fric_Nlink)

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

cost = 1

# Finally creates the mini_wam object
dt = 0.01
mini_wam = MDP.NewtonEulerSys(m,I,dt,X,U,GU,cost,Ffric,Constraint)

######## Clear some variables ############
# They were once symbols
# Now they are just gone
# They will come back as numbers
del X,U,GU,Ffric,Constraint

######## Here begins numerical simulation ########
Nstep = 200
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
        Quat[l] = uq.expq(np.pi/4*np.array([0,1,0]))
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
    X[:,k+1] = mini_wam.step(X[:,k],U[:,k],10)
    
ani = wam_movie(X,dt,100)