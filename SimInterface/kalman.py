import numpy as np
import scipy.linalg as la
import functionApproximator as fa
import utils.numpy_utils as nu

def initializeCovariance(Mat,n):
    if Mat is None:
        Mat = np.eye(n)
    elif not isinstance(Mat,np.ndarray):
        Mat = Mat * np.eye(n)

    return Mat

class kalmanEstimator:
    """
    Estimate the coefficients of a function approximator 
    via an (extended) Kalman filter.

    In particular, given input-output pairs, (x,y), coefficients p 
    are estimated. 

    y = phi(x,p)+noise
    """
    def __init__(self,approximator=None,decay=True,
                 driftCovariance=None,measurementCovariance=None,
                 priorCovariance=None,
                 label=''):

        self.approximator = approximator

        n = approximator.NumParams
        q = approximator.NumOutputs

        self.W = initializeCovariance(driftCovariance,n)
        self.V = initializeCovariance(measurementCovariance,q)
        self.P = initializeCovariance(priorCovariance,n)

        if decay:
            self.Winit = np.array(self.W,copy=True)
            self.Vinit = np.array(self.V,copy=True)

        self.stepNumber = 1
        self.label = label
        self.decay = decay

    def step(self,x,y):
        q = self.approximator.NumOutputs
        n = self.approximator.NumParams
        
        mu = self.approximator.parameter
        C = self.approximator.parameterGradient(x)

        Psi = np.dot(C,np.dot(self.P,C.T)) + self.V

        CP = nu.castToShape(np.dot(C,self.P),(q,n))

        L = -la.solve(Psi,CP,sym_pos=True)

        self.P = self.W + self.P + np.outer(L,CP)
        self.P = .5*(self.P+self.P.T)

        muDiff = np.dot(L,np.dot(C,mu)-y)
        mu += nu.castToShape(muDiff,(n,))
        self.approximator.resetParameter(mu)

        if self.decay:
            self.W = self.Winit / self.stepNumber
            self.V = self.Vinit / self.stepNumber
        self.stepNumber += 1


class kalmanPolicyEvaluator(kalmanEstimator):
    def __init__(self,approximator=None,discountFactor=None,
                 driftCovariance=None,measurementCovariance=None,
                 priorCovariance=None,decay=True,label=''):
        """
        A Kalman filter approach to policy evaluation
        """
        self.label=label
        self.approximator = approximator
        NumParams = approximator.NumParams
        NumVars = approximator.NumVars
        NumOutputs = 1
        def tdValue(z,k=None,beta=None):
            x = z[:NumVars]
            xNew = z[NumVars:]
            curVal = approximator.valueVariable(x,k,beta)
            newVal = approximator.valueVariable(xNew,k,beta)
            return curVal - discountFactor * newVal

        def tdGrad(z,k=None,beta=None):
            x = z[:NumVars]
            xNew = z[NumVars:]
            curGrad = approximator.parameterGradientVariable(x,k,beta)
            newGrad = approximator.parameterGradientVariable(xNew,k,beta)
            return curGrad - discountFactor * newGrad

        tdApproximator = fa.functionApproximator(NumVars=NumVars,
                                                 NumParams=NumParams,
                                                 NumOutputs=NumOutputs,
                                                 valueFunc=tdValue,
                                                 parameterGradFunc=tdGrad,
                                                 parameter=approximator.parameter)

        self.tdKalman = kalmanEstimator(approximator=tdApproximator,
                                        driftCovariance=driftCovariance,
                                        measurementCovariance=measurementCovariance,
                                        priorCovariance=priorCovariance,
                                        decay=decay)

    def step(self,val,x,xNew):
        z = np.hstack((x,xNew))
        self.tdKalman.step(z,val)
        newParam = self.tdKalman.approximator.parameter
        self.approximator.resetParameter(newParam)
        
                                   
            
            
