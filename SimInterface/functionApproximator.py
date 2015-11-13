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

def createAffineBasisFunction(NumOutputs=0):
    def affineBasis(x,k,theta):
        # Put in k because some functions require it
        v = np.hstack((1,x))
        if NumOutputs > 0:
            M = np.kron(v,np.eye(NumOutputs))
        else:
            M = v
        return np.dot(M,theta)
    
    return affineBasis

def stackedSymmetricBasis(n):
    """
    Gives a basis matrix for the symmetric matrices after stacking. 

    For a symmetric matrix nxn , M, this returns a matrix B such that 
    
    np.reshape(M,n*n,order='F') == np.dot(B, stackLower(M))
    """
    B = np.zeros((n*n,n*(n+1)/2))

    for k in range(n):
        row = np.zeros((n,B.shape[1]))
        if k == 0:
            row[:n,:n] = np.eye(n)
        else:
            for j in range(k):
                v = np.zeros(n-j)
                v[k-j] = 1
                if j == 0:
                    blkDiag = v
                else:
                    blkDiag = la.block_diag(blkDiag,v)
            blkDiag = la.block_diag(blkDiag,np.eye(n-k))
            row[:n,:blkDiag.shape[1]] = blkDiag
        B[k*n:(k+1)*n] = row
    return B
            
    
def createLQBasisFunction(NumInputs=0):
    """
    Assumming that that we want a scalar quadratic, along with 
    linear and constant terms
    """

    # NumInputs+1 because we want linear quadratic and constant terms
    B = stackedSymmetricBasis(NumInputs+1)
    def lqBasis(x,k):
        """
        Putting k here for consistency with other systems
        """
        z = np.hstack((1,x))
        BMat = np.dot(np.kron(z,z),B)
        return BMat
    
    return lqBasis

class linearlyParameterizedFunction(functionApproximator):
    """

    """
    def __init__(self,basisFunction=None,*args,**kwargs):
        self.basis = basisFunction
        def valueFun(x,k,theta):
            return np.dot(basisFunction(x,k),theta)

        def gradientFun(x,k,theta):
            return basisFunction(x,k)

        functionApproximator.__init__(self,
                                      valueFunc=valueFun,
                                      parameterGradFunc=gradientFun,
                                      *args,**kwargs)

class parameterizedQuadratic(linearlyParameterizedFunction):
    def __init__(self,NumVars=0,*args,**kwargs):
        basisFun = createLQBasisFunction(NumVars)
        NumParams = (NumVars+1)*(NumVars+2)/2
        NumOutputs = 1
        linearlyParameterizedFunction.__init__(self,NumVars=NumVars,
                                               NumParams=NumParams,
                                               NumOutputs=NumOutputs,
                                               basisFunction=basisFun,
                                               *args,**kwargs)

def createRBFbasis(Centers,Lengths,Matrix=None):
    """
    Creates a basis of radial basis functions B(x).

    By default, no Matrix parameter is specified and B(x) has the form

    B(x) z = phi_0(x) z_0 + phi_1(x) z_1 + ... + phi_{q-1}(x) z_{q-1}

    where 
    phi_i(x) = exp(-|| x - Centers_i ||^2 / Lengths_i) .
 
    If a Matrix parameter is specified then 
    B(x) = Matrix diag(phi_0(x),. . . , phi_{q-1}(x) ) 

    so that 
    B(x) z = Matrix_0 phi_0(x) z_0 + ... Matrix_{q-1} phi_{q-1}(x) z_{q-1}
    """
    # This looks a bit funny because we're accounting for either matrix or
    # vector basis functions.
    NumParameters = len(Centers)
    if not isinstance(Lengths,np.ndarray):
        Lengths = Lengths * np.ones(NumParameters)
        
    def basisFun(x,k=None):
        phi = np.zeros(NumParameters)
        for i in range(NumParameters):
            r = x - Centers[i]
            rSq = np.dot(r,r)
            phi[i] = np.exp(-rSq/Lengths[i])

        if Matrix is None:
            basisMat = phi
        else:
            basisMat = np.dot(Matrix,np.diag(phi))
        return basisMat

    return basisFun
    
class rbfNetwork(linearlyParameterizedFunction):
    def __init__(self,Centers,Lengths,*args,**kwargs):
        NumParams,NumVars = Centers.shape
        NumOutputs = 1

        if not isinstance(Lengths,np.ndarray):
            Lengths = Lengths * np.ones(NumParams)

        basisFun = createRBFbasis(Centers,Lengths)

        linearlyParameterizedFunction.__init__(self,
                                               NumVars=NumVars,
                                               NumParams=NumParams,
                                               NumOutputs=NumOutputs,
                                               basisFunction=basisFun,
                                               *args,**kwargs)

    
# Probably should put these helpers in in numpy utils
def stackLower(M):
    """
    Stack the lower triangle of a matrix M into a vector
    """
    n = len(M)
    v = np.zeros(n*(n+1)/2,dtype=M.dtype)
    startIndex = 0
    for k in range(n):
        v[startIndex:startIndex+n-k] = M[k:,k]
        startIndex += n-k

    return v

def unstackLower(v):
    """
    Turn a vector into a lower-triangular matrix
    """
    m = len(v)
    n = int((np.sqrt(1+8*m)-1)/2)

    M = np.zeros((n,n),dtype=v.dtype)
    startIndex = 0
    
    for k in range(n):
        M[k:,k] = v[startIndex:startIndex+n-k]
        startIndex += n-k

    return M
    
class parameterizedLogGaussian(functionApproximator):
    """
    Approximates the logarithm of a gaussian
    log( Nor(u | phi(x,k) beta, CC') )

    This function is helful for noisy parameterized policies. 
    """
    def __init__(self,basisFunction=None,NumU=1,NumX=1,*args,**kwargs):
        # Only look at the lower triangle
        # so C has NumU*(NumU+1)/2 non-zero entries. 

        NumC = NumU * (NumU+1) / 2

        
        def valueFun(z,k=None,theta=None):
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
            basisSol = la.solve_triangular(C,basisValue,lower=True)

            dVdBeta = np.dot(basisSol.T,y)

            # Derivative with respect to C
            ProjMat = np.eye(NumU) - np.outer(y,y)
            DerivMat = -la.solve_triangular(C.T,ProjMat,lower=False)

            dVdC = stackLower(DerivMat)

            return np.hstack((dVdBeta,dVdC))
            

        functionApproximator.__init__(self,
                                      valueFunc=valueFun,
                                      parameterGradFunc=gradFun,
                                      *args,**kwargs)
            
        
