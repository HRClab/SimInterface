import MarkovDecisionProcess as MDP
import numpy as np
from numpy.random import randn
from scipy.linalg import solve

class Controller:
    def __init__(self,Horizon=0):
        self.Horizon = Horizon

    def action(self,x,k):
        return 0



class staticGain(Controller):
    def __init__(self,gain=0,*args,**kwargs):
        self.gain = gain
        Controller.__init__(self,*args,**kwargs)

    def action(self,x,k):
        return self.gain*x

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
        self.predictiveHorizon = predictiveHorizon
    def action(self,x,k):
        # Currently only supporting time invariant systems
        # It will probably be weird if a time-varying system is used.
        # This will need to be updated for consistency with nonlinear systems
        dynMat = self.SYS.dynamicsMatrix
        costMat = self.SYS.costMatrix
        predictiveSystem = MDP.LinearQuadraticSystem(dynMat,costMat)
        predictiveController = linearQuadraticRegulator(predictiveSystem,
                                                        Horizon = self.predictiveHorizon)

        curVec = np.hstack((1,x))
        gain = predictiveController.Gain[0]
        return np.dot(gain,curVec)

class samplingController(Controller):
    def __init__(self,SYS,KLWeight=1, burnIn=0, *args, **kwargs):
        Controller.__init__(self,*args,**kwargs)
        self.KLWeight = KLWeight
        self.burnIn = burnIn
