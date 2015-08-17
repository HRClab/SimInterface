import MarkovDecisionProcess as MDP
import numpy as np
from numpy.random import randn
from scipy.linalg import solve, cholesky, block_diag, eigh
from bovy_mcmc.elliptical_slice import elliptical_slice as eslice

class Controller:
    def __init__(self,Horizon=0,label=''):
        self.Horizon = Horizon
        self.label = label

    def action(self,x,k):
        return 0

class openLoopPolicy(Controller):
    def __init__(self,U,*args,**kwargs):
        Controller.__init__(self,*args,**kwargs)
        self.U = U
        
    def action(self,x,k):
        return self.U[k]

class flatOpenLoopPolicy(Controller):
    """
    Open loop policy from flat array of inputs
    """
    def __init__(self,U=0,NumInputs=1,*args,**kwargs):
        Controller.__init__(self,*args,**kwargs)
        self.U=U
        self.NumInputs = NumInputs
    def action(self,x,k):
        u = self.U[self.NumInputs*k : self.NumInputs * (k+1)]
        return u
    
class staticGain(Controller):
    def __init__(self,gain=0,*args,**kwargs):
        self.gain = gain
        Controller.__init__(self,*args,**kwargs)

    def action(self,x,k):
        u = np.dot(self.gain,x)
        return u

class staticFunction(Controller):
    def __init__(self,func,*args,**kwargs):
        self.func = func
        Controller.__init__(self,*args,**kwargs)
        
    def action(self,x,k):
        u = self.func(x)
        return u
        
def schurComplement(M,p):
    A = M[:-p,:-p]
    B = M[:-p,-p:]
    C = M[-p:,-p:]
    S = A - np.dot(B,solve(C,B.T,sym_pos=True))
    return S

def gainMatrix(M,p):
    B = M[:-p,-p:]
    C = M[-p:,-p:]
    return -solve(C,B.T,sym_pos=True)

class linearQuadraticRegulator(Controller):
    def __init__(self,SYS,*args,**kwargs):
        Controller.__init__(self,*args,**kwargs)
        self.computeGains(SYS)

    def computeGains(self,sys):
        n = sys.NumStates
        p = sys.NumInputs

        self.RiccatiSolution = np.zeros((self.Horizon,n+1,n+1))
        self.Gain = np.zeros((self.Horizon,p,n+1))

        if sys.timeInvariant:
            costMat = sys.costMatrix
            dynMat = sys.dynamicsMatrix
            bigDynMat = np.vstack((np.hstack((1,np.zeros(n+p))),
                                   dynMat))
        else:
            costMat = sys.costMatrix[-1]
        self.RiccatiSolution[-1] = schurComplement(costMat,p)
        self.Gain[-1] = gainMatrix(costMat,p)
        
        for k in range(self.Horizon-1)[::-1]:
            if not sys.timeInvariant:
                costMat = sys.costMatrix[k]
                dynMat = sys.dynamicsMatrix[k]
                bigDynMat = np.vstack((np.hstack((1,np.zeros(n+p))),
                                       dynMat))

            Ric = self.RiccatiSolution[k+1]
            M = costMat + np.dot(bigDynMat.T,np.dot(Ric,bigDynMat))
            self.RiccatiSolution[k] = schurComplement(M,p)
            self.Gain[k] = gainMatrix(M,p)

        self.Gain = self.Gain.squeeze()

    def action(self,x,k):
        curVec = np.hstack((1,x))
        return np.dot(self.Gain[k],curVec)

class modelPredictiveControl(Controller):
    def __init__(self,SYS,predictiveHorizon,*args,**kwargs):
        Controller.__init__(self,*args,**kwargs)
        self.SYS = SYS
        self.previousAction = np.zeros(SYS.NumInputs).squeeze()
        self.predictiveHorizon = predictiveHorizon
    def action(self,x,k):
        # Currently only supporting time invariant systems
        # It will probably be weird if a time-varying system is used.
        # This will need to be updated for consistency with nonlinear systems
        predictiveController = approximateLQR(self.SYS,
                                              x,
                                              self.previousAction,
                                              k,
                                              Horizon=self.predictiveHorizon)
        # dynMat,costMat = self.SYS.getApproximationMatrices(x,
        #                                                    self.previousAction,
        #                                                    k)
        # predictiveSystem = MDP.LinearQuadraticSystem(dynMat,
        #                                              costMat,
        #                                              self.SYS.timeInvariant)
        # predictiveController = linearQuadraticRegulator(predictiveSystem,
        #                                                 Horizon = self.predictiveHorizon)

        curVec = np.hstack((1,x))
        gain = predictiveController.Gain[0]
        u = np.dot(gain,curVec)
        self.previousAction = u
        return u

class approximateLQR(linearQuadraticRegulator):
    def __init__(self,SYS,x,u,k=0,*args,**kwargs):
        dynMat,costMat = SYS.getApproximationMatrices(x,u,k)
        n = SYS.NumStates
        p = SYS.NumInputs
        # Convexify, assuming that the non-convexity is due to the
        # state cost.
        eigMin = eigh(costMat[1:n+1,1:n+1],eigvals_only=True,eigvals=(0,0))[0]
        if eigMin < 0:
            alpha = -1.1 * eigMin
            xMat = np.reshape(x,(n,1))
            xSqMat = np.reshape(np.dot(x,x),(1,1))
            costMat[:n+1,:n+1] += alpha * \
                                  np.vstack((np.hstack((xSqMat,-xMat.T)),
                                             np.hstack((-xMat,np.eye(n)))))
                                                    
        approxSYS = MDP.LinearQuadraticSystem(dynMat,costMat)
        linearQuadraticRegulator.__init__(self,SYS=approxSYS,*args,**kwargs)
        
class samplingControl(flatOpenLoopPolicy):
    def __init__(self,
                 SYS,
                 KLWeight=1,
                 burnIn=0,
                 ExplorationCovariance=1.,
                 initialPolicy = None,
                 *args, **kwargs):

        flatOpenLoopPolicy.__init__(self,NumInputs=SYS.NumInputs,
                                    *args,**kwargs)

        self.SYS = SYS
        self.KLWeight = KLWeight

        if isinstance(ExplorationCovariance,np.ndarray):
            cholSig = cholesky(ExplorationCovariance,lower=True)
            NoiseGain = np.kron(np.eye(self.Horizon),cholSig)
        else:
            NoiseGain = np.sqrt(ExplorationCovariance) * np.eye(self.Horizon)

        lenW = NoiseGain.shape[0]

        if initialPolicy is None:
            U = np.zeros(lenW)
        else:
            U = SYS.simulatePolicy(initialPolicy)[1].flatten()
            
        logLik = self.loglikelihood(U)
        bestCost = np.inf

        for samp in range(burnIn):
            W = np.dot(NoiseGain,randn(lenW))
            U,logLik = eslice(U,W,self.loglikelihood,cur_lnpdf=logLik)
            cost = -logLik * self.KLWeight
            if cost < bestCost:
                bestCost = cost
                bestU = U

            if (samp+1) % 10 == 0:
                print 'run %d of %d, Cost: %g, Best Cost: %g' % \
                    (samp+1,burnIn,cost,bestCost)

        self.U = bestU

    def loglikelihood(self,U):
        self.U = U        
        cost = self.SYS.simulatePolicy(self)[2]
        return -cost / self.KLWeight
