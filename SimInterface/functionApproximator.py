import numpy as np
import scipy.linalg as la

class functionApproximator:
    """
    Base class for function approximators. 

    It is assumed that the general value functions and gradients will be 
    passed in initialization.
    """
    def __init__(self,NumVars=1,NumParams=1,NumOutputs=1,
                 valueFunc=None,parameterGradFunc=None,parameter=None):
        self.NumVars = NumVars
        self.NumParams = NumParams
        self.NumOutputs = NumOutputs

        self.parameter=parameter
        # Functions with parameter left as a variable
        self.valueVariable = valueFunc
        self.parameterGradientVariable = parameterGradFunc

    def value(self,x,k=None):
        return self.valueVariable(x,k,self.parameter)

    def parameterGradient(self,x,k=None):
        return self.parameterGradientVariable(x,k,self.parameter)

    def resetParameter(self,parameter):
        self.parameter=parameter

class linearlyParameterizedFunction(functionApproximator):
    """

    """
    def __init__(self,basisFunction=None,*args,**kwargs):
        def valueFun(x,k,theta):
            return np.dot(basisFunction(x,k),theta)

        def gradientFun(x,k,theta):
            return basisFunction(x,k)

        functionApproximator.__init__(self,
                                      valueFunc=valueFun,
                                      parameterGradFunc=gradientFun,
                                      *args,**kwargs)

# Probably should put these helpers in in numpy utils
def stackLower(M):
    """
    Stack the lower triangle of a matrix M into a vector
    """
    n = len(M)
    v = np.zeros(n*(n+1)/2)
    startIndex = 0
    for k in range(n):
        v[startIndex:startIndex+k+1] = M[k,:k+1]
        startIndex += k+1

    return v

def unstackLower(v):
    """
    Turn a vector into a lower-triangular matrix
    """
    m = len(v)
    n = int((np.sqrt(1+8*m)-1)/2)

    M = np.zeros((n,n))
    startIndex = 0
    
    for k in range(n):
        M[k,:k+1] = v[startIndex:startIndex+k+1]
        startIndex += k+1

    return M
    
class parameterizedLogGaussian(functionApproximator):
    """
    Approximates the logarithm of a param
    log( Nor(u | phi(x,k) beta, CC') )

    This function is helful for noisy parameterized policies. 
    """
    def __init__(self,basisFunction=None,NumU=1,NumX=1,*args,**kwargs):
        # Only look at the lower triangle
        # so C has NumU*(NumU+1)/2 non-zero entries. 

        NumC = NumU * (NumU+1) / 2

        
        def valueFun(z,k,theta):
            x = z[:NumX]
            u = z[NumX:]
            
            beta = theta[:-NumC]
            CStacked = theta[-NumC:]

            r = u - np.dot(basisFunction(x,k),beta)
            C = unstackLower(CStacked)
            y = la.solve_triangular(C,r,lower=True)

            quadPart = -.5 * np.dot(y,y)
            detPart = -np.log(np.diag(C)).sum()
            constPart = -(NumU/2.) * np.log(2*np.pi)
            
            return quadPart + detPart + constPart

        def gradFun(z,k,theta):
            x = z[:NumX]
            u = z[NumX:]
            
            beta = theta[:-NumC]
            CStacked = theta[-NumC:]

            r = u - np.dot(basisFunction(x,k),beta)
            C = unstackLower(CStacked)

            # Derivative with respect to beta
            y = la.solve_triangular(C,r,lower=True)
            basisValue = basisFunction(x,k)
            basisSol = la.solve_triangular(C,y,lower=True)

            dVdBeta = np.dot(basisSol.T,y)

            # Derivative with respect to C
            ProjMat = np.eye(NumU) - np.outer(y,y)
            DerivMat = la.solve_triangular(C.T,ProjMat,lower=False)

            dVdC = stackLower(DerivMat)

            return np.hstack((dVdBeta,dVdC))
            

        functionApproximator.__init__(self,
                                      valueFunc=valueFun,
                                      parameterGradFunc=gradFun,
                                      *args,**kwargs)
            
        
