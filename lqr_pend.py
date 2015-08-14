import numpy as np
from scipy.linalg import eig, eigh,solve, solve_discrete_are
import UnderactuatedPendulum as UAP

dt = 0.001
sys = UAP.UnderactuatedPendulum(dt = dt)
n = len(sys.x0)

x0 = np.zeros(4)
x0[0] = np.pi/2.
A,B,g = sys.discreteTimeLinearization(x0,0)
Q = 0.1 * np.eye(n)
R = sys.r
XRic = solve_discrete_are(A,np.reshape(B,(4,1)),Q,np.array([[R]]))
KRic = -np.dot(B.T,np.dot(XRic,A)) / (R+np.dot(B.T,np.dot(XRic,B)))
# Augmented Dynamics matrix
H = np.zeros((n+1,n+2))
H[0,0] = 1.
H[1:,0] = g
H[1:,1:n+1] = A
H[1:,n+1] = B

C = sys.discreteTimeCostMatrix(x0)

# Compute the smallest eigenvalue, using the fact that
# C is symmetric
lamMin = eigh(C,eigvals_only = True,eigvals=(0,0))[0]

if lamMin < 0:
    # Augment C to get a convex problem
    alpha = -1.01 * lamMin
    C[0,0] += alpha * np.dot(x0,x0)
    C[1:n+1,1:n+1] += alpha * np.eye(n)
    C[0,1:n+1] -= alpha * x0
    C[1:n+1,0] -= alpha * x0

# Now set up the simluation 
NStep = int(round(.2/dt))
T = dt * np.arange(NStep+1)

RicSol = np.zeros((n+1,n+1,NStep+1))
Gain = np.zeros((n+1,NStep))

def SchurComplement(M):
    A = M[:n+1,:n+1]
    B = M[:n+1,n+1:]
    C = M[n+1:,n+1:]
    SC = A - np.dot(B,solve(C,B.T,sym_pos=True))
    return SC

def GainFromMat(M):
    BT = M[n+1:,:n+1]
    C = M[n+1:,n+1:]
    return -solve(C,BT,sym_pos=True)

RicSol[:,:,-1] = SchurComplement(C)

for k in range(NStep)[::-1]:
    Gam = C + np.dot(H.T,np.dot(RicSol[:,:,k+1],H))
    RicSol[:,:,k] = SchurComplement(Gam)
    Gain[:,k] = GainFromMat(Gam)

x = x0+.2*np.random.randn(4)
X = np.zeros((n,NStep+1))
X[:,0] = x
for k in range(NStep):
    u = np.dot(KRic,x-x0)
    x = sys.dPend.step(x,u,sys.dt,0)

    X[:,k+1] = x

ani = sys.movie(X)
