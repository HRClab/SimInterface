import numpy as np
import numpy.random as rnd
import scipy.linalg as la
import Controller as ctrl

#### Basic Stacking Routines ####

def stackSymmetricMatrix(M):
    """
    Stacks the unique element of a symmetric matrix M in 
    """
    n = len(M)
    v = M[0]
    for k in range(1,n):
        v = np.append(v,M[k,k:])
        
    return v

def unstackSymmetricMatrix(v):
    """
    Stacks a vector into a corresponding symmetric matrix. 
    """
    N = len(v)
    n = int((np.sqrt(1+8*N)-1)/2)
    UpperTriangle = np.zeros((n,n))
    Diagonal = np.zeros(n)

    startIndex = 0
    for k in range(n):
        Diagonal[k] = v[startIndex]
        if k<(n-1):
            endIndex = startIndex+n-k
            UpperTriangle[k,k+1:] = v[startIndex+1:endIndex]
            startIndex = endIndex

    M = UpperTriangle+UpperTriangle.T+np.diag(Diagonal)
    return M
    
class actorCriticLQR(ctrl.varyingAffine):
    """
    Natual Actor Critic method applied to linear quadratic systems
    """
    def __init__(self,SYS,Horizon=1,
                 EpisodeLength=1,EpisodeCount=1,
                 TraceDecayRate=0.,DiscountRate=1.,
                 stateGain=None,Covariance=None,
                 *args,**kwargs):
        p = SYS.NumInputs
        n = SYS.NumStates

        self.n = n
        self.p = p

        if stateGain is None:
            stateGain = np.zeros((p,n))
            
        if Covariance is None:
            Covariance = np.eye(n)


        AffineGain = np.zeros((EpisodeLength,p,n+1))
        ctrl.varyingAffine.__init__(self,AffineGain,Horizon=EpisodeLength,
                                    *args,**kwargs)

        
        self.stateGain = stateGain
        self.Covariance = Covariance
        self.resetGains()

        for episode in range(EpisodeCount):
            X,U,Cost = SYS.simulatePolicy(self)

            

            
        AffineGain = np.zeros((Horizon,p,n+1))
        self.Gain = AffineGain
        self.Horizon = Horizon
        self.resetGains()
        
    def resetGains(self):
        CovChol = la.cholesky(self.Covariance,lower=True)
        W = rnd.randn(self.Horizon,self.n)

        for k in range(self.Horizon):
            self.Gain[k,0,:] = np.dot(CovChol,W[k])
            self.Gain[k,1:,:] = self.stateGain
