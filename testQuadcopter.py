#### Testing the nes class with quadcopter

import NewtonEulerSystem as nes
import matplotlib.pyplot as plt
import pyuquat as uq
import sympy as sym
import numpy as np
import matplotlib.animation as animation

#### Actually I should create a class called quadcopter
#### So we can sepcify the parameters and instantiate an object in the class

#### To build a NewtonEulerSystem, you need to specify
#### Number of rigid bodies
#### M matrix
#### Form of input and how that transform to generalized force
#### Constraints
#### That's it

#### Define the quadcopter model and instantiate it

xstride = 13
pstride = 7
vstride = 6

NumQuad = 1

ALcT = 0.1
cT = 1 # thrust coeff
cQ = 0.01 # drag coeff

X,P,V,Pos,Quat,Vel,Omega = nes.states_generator(NumQuad)
m = 0.4
I = [0.1,0.1,0.1]

M = nes.build_M_matrix(m,I)

# input and generalized input (wrench)
U = sym.symarray('U',(4))
GU = np.zeros(vstride,dtype = object)

GU[2] = cT*np.dot(U,U)
GU[3] = ALcT*(U[1]**2-U[3]**2)
GU[4] = ALcT*(U[2]**2-U[0]**2)
GU[5] = cQ*(-U[0]+U[1]-U[2]+U[3])

# instantiation
quad = nes.NewtonEulerSystems(M,X,U,GU)

######## Clear some variables ############
# They were once symbols
# Now they are just gone
# They will come back as numbers
del X,U,GU

######## Here begins numerical simulation ########
Nstep = 100
dt = 0.1
#### Initializing arm states for one time instance
Pos = np.zeros(3)
Quat = np.zeros(4)
Vel = np.zeros(3)
Omega = np.zeros(3)

# Quaternions and Positions should be initialized 
# more carefully. Quaternions should be initialized first
Quat = uq.expq(np.array([0,0,0]))
                          
# packed states of all links for one time instance
x = np.zeros(NumQuad*xstride)
x = np.hstack((Pos,Quat,Vel,Omega))

#### Initialize packed states for all time instances
X = np.zeros((len(x),Nstep))
X[:,0] = x

#### Initialize input
U = np.zeros((4,Nstep))
U = 1.01*np.ones((4,Nstep))

#### Main loop ####
for k in range(Nstep-1):
    dx = quad.state_diff(X[:,k],U[:,k],dt)
    X[:,k+1] = X[:,k] + dx
    
