import numpy as np
from scipy.linalg import solve, eig, eigh
import Controller as ctrl
import linearQuadraticSystem as LQS

##### LQR Helper Functions #####

def eigMin(M):
    return eigh(M,eigvals_only=True,eigvals=(0,0))[0]

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

class linearQuadraticRegulator(ctrl.Controller):
    def __init__(self,SYS,*args,**kwargs):
        ctrl.Controller.__init__(self,*args,**kwargs)
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

    def action(self,x,k):
        curVec = np.hstack((1,x))
        return np.dot(self.Gain[k],curVec)

class modelPredictiveControl(ctrl.Controller):
    def __init__(self,SYS,predictiveHorizon,*args,**kwargs):
        ctrl.Controller.__init__(self,*args,**kwargs)
        self.SYS = SYS
        self.previousAction = np.zeros(SYS.NumInputs)
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

class iterativeLQR(ctrl.varyingAffine):
    def __init__(self,SYS,initialPolicy = None,Horizon=1,
                 stoppingTolerance=1e-3,maxIter=np.inf,*args,**kwargs):
        self.Horizon = Horizon
        if initialPolicy is None:
            gain = np.zeros((SYS.NumInputs,SYS.NumStates))
            initialPolicy = ctrl.staticGain(gain=gain,Horizon=Horizon)
        else:
            self.Horizon = initialPolicy.Horizon

        X,U,initCost = SYS.simulatePolicy(initialPolicy)

        initCost = initCost.sum()
        
        gainShape = (self.Horizon, SYS.NumInputs, SYS.NumStates+1)

        bestCost = initCost
        bestGain = np.zeros(gainShape)
        for k in range(self.Horizon):
            bestGain[k,:,0] = U[k]

        costChange = np.inf

        # Cost regularization Parameters
        alpha = 1
        n = SYS.NumStates
        p = SYS.NumInputs

        testController = ctrl.varyingAffine(gain=bestGain,
                                                   Horizon=self.Horizon,
                                                   *args,**kwargs)

        self.costSequence = [initCost]
        run = 0 
        while run<maxIter:
            run += 1
            if (maxIter == np.inf) and \
               (np.abs(costChange) <= stoppingTolerance):
                break
            
            if alpha > 1e12:
                print 'regularization parameter too large'
                break

            correctionSys = LQS.buildCorrectionSystem(SYS,X,U)

            
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
                newCost = newCost.sum()
                costChange = newCost-bestCost
                print 'iLQR cost: %g, costChange %g' % (newCost,costChange)
                if newCost < bestCost:
                    X = newX
                    U = newU
                    bestCost = newCost
                    bestGain = testController.Gain
                    alpha = 1
                else:
                    alpha *= 2

                self.costSequence.append(bestCost)
            except (ValueError,np.linalg.LinAlgError):
                alpha *= 2
                print 'numerical problem'
                # print 'increasing regularization parameter to %g' % alpha



        ctrl.varyingAffine.__init__(self,gain=bestGain,
                                    Horizon=self.Horizon,
                                    *args,**kwargs)

    
class approximateLQR(linearQuadraticRegulator):
    def __init__(self,SYS,x,u,k=0,*args,**kwargs):
        dynMat,costMat = LQS.convexApproximationMatrices(SYS,x,u,k)
            
        approxSYS = LQS.linearQuadraticSystem(dynMat,costMat,x0=x)
        linearQuadraticRegulator.__init__(self,SYS=approxSYS,*args,**kwargs)
