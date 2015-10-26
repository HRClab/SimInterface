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

def explorationGain(stateGain,Covariance,Horizon):
    CovChol = la.cholesky(Covariance,lower=True)
    p,n = stateGain.shape
    W = rnd.randn(Horizon,n)

    Gain = np.zeros((Horizon,p,1+n))
    for k in range(Horizon):
        Gain[k,0,:] = np.dot(CovChol,W[k])
        Gain[k,1:,:] = stateGain
        
    return Gain
class actorCriticLQR(ctrl.staticGain):
    """
    Natual Actor Critic method applied to linear quadratic systems. 
    
    In particular, This is a variant of Table 1 of 
    "Natural Actor-Critic" by Peters, Vijayakumar, and Schaal
    """
    def __init__(self,SYS,verbose=False,
                 EpisodeLength=1,EpisodeCount=1,stepSizeConstant=1.,
                 TraceDecayFactor=0.,DiscountFactor=1.,ForgettingFactor=1,
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



        numFeatures = n*(n+1) + n*p + 1 

        # Auxilliary temporal difference variables
        z = np.zeros(numFeatures)
        A = np.zeros((numFeatures,numFeatures))
        b = np.zeros(numFeatures)

        x0 = SYS.x0
        #### Actor Critic Learning ####
        for episode in range(EpisodeCount):
            AffineGain = explorationGain(stateGain,Covariance,EpisodeLength)
            learningController = ctrl.varyingAffine(AffineGain,
                                                    Horizon=EpisodeLength)

            if episode > 0:
                SYS.x0 = X[-1]
            X,U,Cost = SYS.simulatePolicy(learningController)
            if episode > 0:
                SYS.x0 = x0
                
            ### Policy Evaluation by least squares temporal differences ###

            TrueCost = 0.
            CostFactor = 1.
            for k in range(EpisodeLength-1):
                TrueCost += CostFactor * Cost[k]
                CostFactor *= DiscountFactor 

                xSquare = np.outer(X[k],X[k])
                xSquareStacked = stackSymmetricMatrix(xSquare)

                inputResidual = U[k] - np.dot(stateGain,X[k])
                gainSolve = la.solve(Covariance,inputResidual,sym_pos=True)
                gainDerivative = np.outer(gainSolve,X[k])
                gainDerivativeStacked = np.reshape(gainDerivative,n*p)

                invCov = la.inv(Covariance)
                covDerivative = .5*(np.outer(gainSolve,gainSolve)-invCov)
                covDerivativeStacked = stackSymmetricMatrix(covDerivative)

                PhiHat = np.hstack((1,xSquareStacked,
                                    gainDerivativeStacked,
                                    covDerivativeStacked))

                PhiTilde = np.zeros(numFeatures)
                PhiTilde[0] = 1.
                nextXSquare = np.outer(X[k+1],X[k+1])
                nextXSquareStacked = stackSymmetricMatrix(nextXSquare)
                PhiTilde[1:1+len(xSquareStacked)] = nextXSquareStacked
                
                z = TraceDecayFactor * z + PhiHat
                A += np.outer(z,PhiHat-DiscountFactor*PhiTilde)
                b += z*Cost[k]

            # Optimal Feature Weights
            featureWeights = la.solve(A,b)

            constCost = featureWeights[0]
            costMatrixStacked = featureWeights[1:n*(n+1)/2+1]
            P = unstackSymmetricMatrix(costMatrixStacked)
            costEst = np.dot(X[0],np.dot(P,X[0])) + constCost
            if verbose:
                print 'Episode Cost: %g, Estimated Episode Cost: %g' \
                    % (TrueCost,costEst)
                                                                    

            policyGradient = featureWeights[n*(n+1)/2 + 1:]

            Kdiff = np.reshape(policyGradient[:n*p],(p,n))
            CovDiff = unstackSymmetricMatrix(policyGradient[n*p:])
            stepSize = stepSizeConstant / (episode+1)

            stateGain -= stepSize * Kdiff
            Covariance -= stepSize * CovDiff

            # Forget previous influence slightly
            z = ForgettingFactor * z
            A = ForgettingFactor * A
            b = ForgettingFactor * b

            # May need to figure out how to get the appropriate gain shape
            ctrl.staticGain.__init__(self,gain=stateGain.squeeze(),NumInputs=p,
                                     *args,**kwargs)
        
