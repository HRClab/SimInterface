import MarkovDecisionProcess as MDP
import numpy as np
from numpy.random import randn
from scipy.linalg import solve, cholesky, block_diag, eigh, eig
from bovy_mcmc.elliptical_slice import elliptical_slice as eslice

def eigMin(M):
    return eigh(M,eigvals_only=True,eigvals=(0,0))[0]

class Controller:
    """
    Base class for a controller.
    It is by default an open-loop controller, i.e. it returns input 0
    at each step. 
    """
    def __init__(self,Horizon=0,NumInputs = 1,label=''):
        self.NumInputs = NumInputs
        self.Horizon = Horizon
        self.label = label

    def action(self,x,k):
        u = np.zeros(self.NumInputs).squeeze()
        return u

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

class varyingGainAndFeedforward(Controller):
    def __init__(self,gain,*args,**kwargs):
        self.Gain = gain
        Controller.__init__(self,*args,**kwargs)

    def action(self,x,k):
        vec = np.hstack((1,x))
        u = np.dot(self.Gain[k],vec)
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
        self.RiccatiSolution[-1] = .5*(self.RiccatiSolution[-1]+\
                                       self.RiccatiSolution[-1].T)
        self.Gain[-1] = gainMatrix(costMat,p)
        
        for k in range(self.Horizon-1)[::-1]:
            if not sys.timeInvariant:
                costMat = sys.costMatrix[k]
                dynMat = sys.dynamicsMatrix[k]
                bigDynMat = np.vstack((np.hstack((1,np.zeros(n+p))),
                                       dynMat))

            A = dynMat[:,1:n+1]
            v = eig(A)[0]
            # print 'max eigenvalue: %g' % np.abs(v).max()
            Ric = self.RiccatiSolution[k+1]
            OlDyn = bigDynMat[:,:-p]
            RicBound = costMat[:-p,:-p] + np.dot(OlDyn.T,np.dot(Ric,OlDyn))
            UpdateMat = np.dot(bigDynMat.T,np.dot(Ric,bigDynMat))
            M = costMat + UpdateMat
            M = .5*(M+M.T)
            
            newRic = schurComplement(M,p)
            self.RiccatiSolution[k] = .5*(newRic+newRic.T)
            self.Gain[k] = gainMatrix(M,p)

        # print np.diag(self.RiccatiSolution[0][:-p,:-p])
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

        curVec = np.hstack((1,x))
        gain = predictiveController.Gain[0]
        u = np.dot(gain,curVec)
        self.previousAction = u
        return u

class iterativeLQR(varyingGainAndFeedforward):
    def __init__(self,SYS,initialPolicy = None,Horizon=1,
                 stoppingTolerance=1e-3,*args,**kwargs):
        self.Horizon = Horizon
        if initialPolicy is None:
            gain = np.zeros((SYS.NumInputs,SYS.NumStates)).squeeze()
            initialPolicy = staticGain(gain=gain,Horizon=Horizon)
        else:
            self.Horizon = initialPolicy.Horizon

        X,U,initCost = SYS.simulatePolicy(initialPolicy)

        gainShape = (self.Horizon, SYS.NumInputs, SYS.NumStates+1)

        bestCost = initCost
        bestGain = np.zeros(gainShape)
        for k in range(self.Horizon):
            bestGain[k,:,0] = U[k]

        costChange = np.inf

        # Cost regularization Parameters
        alpha = .1
        n = SYS.NumStates
        p = SYS.NumInputs

        testController = varyingGainAndFeedforward(gain=bestGain,
                                                   Horizon=self.Horizon,
                                                   *args,**kwargs)

        run = 0 
        while np.abs(costChange)>stoppingTolerance:
            run += 1
            if alpha > 1e12:
                print 'regularization parameter too large'
                break

            correctionSys = MDP.buildCorrectionSystem(SYS,X,U)

            
            for k in range(self.Horizon):
                dynMat = correctionSys.dynamicsMatrix[k]
                A = dynMat[:,1:n+1]
                v = eig(A)[0]
                # print 'max eigenvalue: %g' % np.abs(v).max(0)
                curCost = correctionSys.costMatrix[k]
                M = curCost[1:,1:]

                nv = M.shape[0]
                
                minEig = eigMin(M)
                if minEig < 0:
                    beta = -1.1 * minEig
                    M += beta * np.eye(nv)

                M[-p:,-p:] += alpha * np.eye(p)
                curCost[1:,1:] = M
                correctionSys.costMatrix[k] = curCost
                
            correctionCtrl = linearQuadraticRegulator(SYS=correctionSys,
                                                      Horizon=self.Horizon,
                                                      *args,**kwargs)

            testController.Gain = np.zeros(gainShape)
            #print 'largest correction gain: %g' % np.abs(correctionCtrl.Gain).max()
            for k in range(self.Horizon):
                testController.Gain[k,:,0] = correctionCtrl.Gain[k,:,0] + \
                                             U[k] - \
                                             np.dot(correctionCtrl.Gain[k,:,1:],X[k])
                testController.Gain[k,:,1:] = correctionCtrl.Gain[k,:,1:]
                                

            try:
                newX,newU,newCost = SYS.simulatePolicy(testController)
                costChange = newCost-bestCost
                print 'iLQR cost: %g, costChange %g' % (newCost,costChange)
                if newCost < bestCost:
                    X = newX
                    U = newU
                    bestCost = newCost
                    bestGain = testController.Gain
                    alpha = .1
                else:
                    alpha *= 2
                    print 'increasing regularization parameter to %g' % alpha
            except (ValueError,np.linalg.LinAlgError):
                alpha *= 2
                print 'numerical problem'
                print 'increasing regularization parameter to %g' % alpha

        varyingGainAndFeedforward.__init__(self,
                                           gain=bestGain,
                                           Horizon=self.Horizon,
                                           *args,**kwargs)

    
class approximateLQR(linearQuadraticRegulator):
    def __init__(self,SYS,x,u,k=0,*args,**kwargs):
        dynMat,costMat = MDP.convexApproximationMatrices(SYS,x,u,k)
            
        approxSYS = MDP.LinearQuadraticSystem(dynMat,costMat,x0=x)
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
        cost = np.inf

        eps = 1e-3
        
        samp = 0
        while samp < burnIn:
            samp += 1
            W = np.dot(NoiseGain,randn(lenW))
            U,logLik = eslice(U,W,self.loglikelihood,cur_lnpdf=logLik)
            newCost = -logLik * self.KLWeight
            if newCost < bestCost:
                bestCost = newCost
                bestU = U
                
            costChange = newCost - cost
            cost = newCost

            
            if burnIn < np.inf:
                if samp % 10 == 0:
                    print 'run %d of %d, Cost: %g, Best Cost: %g' % \
                        (samp,burnIn,cost,bestCost)
            else:
                if costChange >= 0:
                    break

                if samp % 10 == 0:
                    print 'run %d, Cost: %g, Cost Change %g' % \
                        (samp,cost,costChange)
            
        self.U = bestU

    def loglikelihood(self,U):
        self.U = U        
        cost = self.SYS.simulatePolicy(self)[2]
        return -cost / self.KLWeight
