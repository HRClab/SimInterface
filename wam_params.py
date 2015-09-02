import numpy as np

# Number of links
Nlink = 2

# Currently, hard code the parameters. Later they will be symbolic.
g = 10. # Setting really low because large jerks break constraints
m = np.array([2.,0.8,1.2,1.2,0.3,.5,.3])
r = np.array([0.7,0.6,0.5,0.5,0.5,.4,.5])
h = np.array([0.7,0.5,1.8,1.8,.4,.6,.4])
fricCoef = 10

I = np.zeros((Nlink,3,3))
Iinv = np.zeros((Nlink,3,3))
MinvList = np.zeros(6*Nlink)

# Link positions
zJ = np.zeros((Nlink,3))

for k in range(Nlink):
    iList = np.array([m[k] * r[k]**2.,
                      m[k] * ( 3 * r[k]**2. + h[k]**2.) / 12.,
                      m[k] * ( 3 * r[k]**2. + h[k]**2.) / 12.])
    I[k] = np.diag(iList)
    Iinv[k] = np.diag(1./iList)
    MinvList[6*k:6*(k+1)] = np.hstack((np.ones(3)/m[k],1./iList))

    zJ[k] = np.array([0,0,h[k]/2.])
    
Minv = np.diag(MinvList)

# Use a PD control to fix broken constraints
KcP = 1.      
KcD = 1.

# dimensions on each link
xstride = 13 # all states
qstride = 7 # generalized position
pstride = 6 # generalized velocity
fstride = 6 # generalized force
cstride = 5 # constraint force

# transformation matrix from U to generalized force
# this should be a constant matrix regardless of Q and P
# Finput = GU*U
GU = np.zeros((Nlink*fstride,Nlink),dtype=object)
# indices of axis that each joint torque applies on corresponding link
TauUind = np.zeros(Nlink,dtype=int)
for l in range(Nlink):
    TauUind[l] = 1 + ((l+1) % 2) # 1 if odd, 2 if even
for l in range(Nlink):
    if l == (Nlink-1):
        GU[fstride*l+3+TauUind[l]][l] = 1
    else:
        GU[fstride*l+3+TauUind[l]][l] = 1
        GU[fstride*l+3+TauUind[l+1]][l+1] = -1