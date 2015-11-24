import numpy as np
import numpy.random as rnd
import functionApproximator as fa
import Controller as ctrl

class parameterizedFunction(ctrl.Controller):
    """
    
    """
    def __init__(self,policyFunction=None,*args,**kwargs):
        self.policyFunction=policyFunction

    def action(self,x,k=None):
        return self.policyFunction.value(x,k)

    def resetParameter(self,policyParam):
        self.policyFunction.resetParameter(policyParam)



    
class noisyLinParamFun(ctrl.Controller):
    """
    Inputs of the form:
    u = B(x,k) beta + C w

    Here B(x,k) is a basis function and beta is a parameter and 
    C the cholesky factorization of the covariance matrix. 

    The parameters are the entries of beta and C
    """
    def __init__(self,basisFunction=None,parameter=None,
                 NumInputs=1,NumStates=1,NumParams=None,
                 *args,**kwargs):

        # Number of covariance terms

        self.NumInputs = NumInputs

        if NumParams is None:
            NumParams = len(parameter)

        self.logApproximator = fa.parameterizedLogGaussian(basisFunction=basisFunction,
                                                           NumU=NumInputs,
                                                           NumX=NumStates,
                                                           NumParams=NumParams)

        if parameter is None:
            self.beta = None
            self.C = None
        else:
            self.resetParameter(parameter)
                                    
        self.basis = basisFunction    

        ctrl.Controller.__init__(self,NumInputs=NumInputs,*args,**kwargs)

    def action(self,x,k):
        B = self.basis(x,k)
        uDet = np.dot(B,self.beta)
        noise = np.dot(self.C,rnd.randn(self.NumInputs))
        return uDet + noise

    def resetParameter(self,parameter):
        p = self.NumInputs
        m = p*(p+1)/2
        self.beta = parameter[:-m]
        Cstacked = parameter[-m:]
        self.C = fa.unstackLower(Cstacked)
        self.logApproximator.resetParameter(parameter)
