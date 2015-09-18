import numpy as np
from scipy.linalg import cholesky
from numpy.random import randn
from bovy_mcmc.elliptical_slice import elliptical_slice as eslice
import Controller as ctrl

#### Basic Helper Functions

def initializeFlatOpenLoop(SYS, initialPolicy, Horizon=1):
    if initialPolicy is None:
        U = np.zeros(Horizon * SYS.NumInputs)
    else:
        UFull = SYS.simulatePolicy(initialPolicy)[1]
        U = UFull[:Horizon].flatten()

    return U


#### Slice Sampling Optimizer ####

def sliceSample(sampleObj,X,burnIn=1,resetObject=False):
    """
    sampleObj - an object to be sampled. Must have the following attributes
    sampleObj.priorChol - cholesky factorization of Gaussian prior
    sampleObj.loglikelihood - a log-likelihood function 
    sampleObj.displaySampleInfo - a function that prints relevant info to screen

    if resetObject is True then sampleObj.reset must also be defined
    X - an initial sample
    burnIn - length of burn-in period
    resetObject - flag to reset sampleObj after each sample
    """

    lenW = sampleObj.priorChol.shape[0]    
    logLik = sampleObj.loglikelihood(X)
    bestLik = -np.inf

    eps = 1e-3
        
    samp = 0
    likSequence = [logLik]
    
    while samp < burnIn:
        samp += 1
        W = np.dot(sampleObj.priorChol,randn(lenW))
        if resetObject:
            # resets data associated with sample object
            # This is needed for some stochastic variants
            sampleObj.reset()
            # when the sample object has been reset
            # Older values of the log-likelihood function may not
            # be relevant
            X,newLogLik = eslice(X,W,sampleObj.loglikelihood)
        else:
            X,newLogLik = eslice(X,W,sampleObj.loglikelihood,cur_lnpdf=logLik)
        if newLogLik > bestLik:
            bestLik = newLogLik
            bestX = X
                
        likChange = newLogLik - logLik
        logLik = newLogLik

        likSequence.append(logLik)

        if (burnIn == np.inf) and (likChange <= 0):
            break

        sampleObj.displaySampleInfo(logLik,bestLik,samp,burnIn)
            
    return X, bestX, likSequence




def trajectoryNoiseMatrix(Cov,Horizon):
    if isinstance(Cov,np.ndarray):
        cholSig = cholesky(Cov,lower=True)
        NoiseGain = np.kron(np.eye(Horizon),cholSig)
    else:
        NoiseGain = np.sqrt(Cov) * np.eye(Horizon)
    return NoiseGain

def sliceOptimizationDisplay(logLik,bestLik,samp,burnIn,KLWeight):
    cost = -logLik * KLWeight
    bestCost = -bestLik * KLWeight
        
    if burnIn < np.inf:
        if samp % 10 == 0:
            print 'run %d of %d, Cost: %g, Best Cost: %g' % \
                (samp,burnIn,cost,bestCost)
    else:
        if samp % 10 == 0:
            print 'run %d, Cost: %g, Cost Change %g' % \
                (samp,cost,costChange)


def nullDisplay(logLik,bestLik,samp,burnIn,KLWeight):
    pass

#### Slice Sampling Controllers ####


class samplingOpenLoop(ctrl.flatOpenLoopPolicy):
    def __init__(self,
                 SYS = None,
                 KLWeight=1,
                 burnIn=0,
                 ExplorationCovariance=1.,
                 initialPolicy = None,
                 displayFun = sliceOptimizationDisplay,
                 *args, **kwargs):

        self.displayFun = displayFun
        self.SYS = SYS
        self.KLWeight = KLWeight

        ctrl.flatOpenLoopPolicy.__init__(self,NumInputs=SYS.NumInputs,
                                    *args,**kwargs)


        self.priorChol = trajectoryNoiseMatrix(ExplorationCovariance,
                                               self.Horizon)


        self.U = initializeFlatOpenLoop(SYS, initialPolicy, self.Horizon)


        self.burnIn = burnIn
        self.updatePolicy()

    def updatePolicy(self):
        Ucur, self.U, likSequence = sliceSample(self,self.U,self.burnIn)
        self.costSequence = -np.array(likSequence) * self.KLWeight
        
    def loglikelihood(self,U):
        self.U = U
        cost = self.SYS.simulatePolicy(self)[2]
        return -cost / self.KLWeight

    def displaySampleInfo(self,logLik,bestLik,samp,burnIn):
        self.displayFun(logLik,bestLik,samp,burnIn,self.KLWeight)

class samplingStochasticAffine(ctrl.flatVaryingAffine):
    def __init__(self,SYS=None,Horizon=1,
                 NumSamples = 1,
                 KLWeight=1,burnIn=0,
                 ExplorationCovariance=1.,
                 initialPolicy = None,
                 *args, **kwargs):

        self.NumSamples = NumSamples
        self.SYS= SYS
        self.KLWeight = KLWeight

        
        n = SYS.NumStates
        p = SYS.NumInputs

        if initialPolicy is None:
            gain = np.zeros(Horizon*(n+1)*p)
        else:
            # Currently only extract affine part
            gain = initialPolicy.Gain.flatten()
            
        ctrl.flatVaryingAffine.__init__(self,gain,p,Horizon,*args,**kwargs)

        # Set the auxilliary noise using the reset method
        self.reset()
        
        self.priorChol = trajectoryNoiseMatrix(ExplorationCovariance,
                                               self.Horizon)

        self.Gain = sliceSample(self,gain,burnIn,resetObject=True)[1]


    def loglikelihood(self,gain):
        self.Gain = gain
        cost = 0
        for k in range(self.NumSamples):
            cost += self.SYS.simulatePolicy(self,W=self.W[k])[2]

        cost = cost / self.NumSamples
        return -cost / self.KLWeight

    def reset(self):
        self.W = randn(self.NumSamples,self.Horizon,self.SYS.NumNoiseInputs)

    def displaySampleInfo(self,logLik,bestLik,samp,burnIn):
        sliceOptimizationDisplay(logLik,bestLik,samp,burnIn,self.KLWeight)

class samplingMPC(ctrl.Controller):
    def __init__(self,SYS=None,
                 Horizon=1,
                 KLWeight=1,ExplorationCovariance=1.,
                 PredictionHorizon=1,PredictionBurnIn=1,
                 initialPolicy=None,
                 *args,**kwargs):

        self.predictiveController = samplingOpenLoop(SYS,
                                                     KLWeight,
                                                     PredictionBurnIn,
                                                     ExplorationCovariance,
                                                     initialPolicy,
                                                     Horizon=PredictionHorizon,
                                                     displayFun = nullDisplay,
                                                     *args,**kwargs)

        ctrl.Controller.__init__(self,Horizon=Horizon,NumInputs=SYS.NumInputs,
                                 *args,**kwargs)

        # We'll be resetting this during the action function
        # so we need to recall it so that we can set it back
        self.x0 = SYS.x0
    def action(self,x,k):
        # Set initial condition to current state
        self.predictiveController.SYS.x0 = x
        # Currently, not doing anything fancy with the control
        # Relying on the fact that it should not change too much
        # in a single step
        self.predictiveController.updatePolicy()
        u = self.predictiveController.U[:self.NumInputs]
        
        # Set the initial condition back to the original state
        self.predictiveController.SYS.x0 = self.x0

        return u

class gibbsOpenLoop(ctrl.flatOpenLoopPolicy):
    def __init__(self,
                 SYS = None,
                 KLWeight=1,
                 burnIn=0,
                 InputCovariance=1.,
                 StateCovariance=1.,
                 initialPolicy = None,
                 displayFun = sliceOptimizationDisplay,
                 *args, **kwargs):

        pass
