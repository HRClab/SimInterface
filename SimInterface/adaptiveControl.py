import numpy as np
import numpy.random as rnd
import scipy.linalg as la
import functionApproximator as fa
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

class naturalActorCritic(ctrl.noisyLinParamFun):
    """
    Natual Actor Critic method applied to general systems.

    In particular, This is a variant of Table 1 of 
    "Natural Actor-Critic" by Peters, Vijayakumar, and Schaal

    this requires two function approximator objects
    policyApproximator - general differentiable approximator
    costApproximator - linear approximator
    """
    def __init__(self,SYS,
                 EpisodeLength=1,EpisodeCount=1,stepSizeConstant=1.,
                 TraceDecayFactor=0.,DiscountFactor=1.,ForgettingFactor=1,
                 policy=None,costApproximator=None,reset=False,
                 *args,**kwargs):

        p = SYS.NumInputs
        n = SYS.NumStates

        self.n = n
        self.p = p

        policyApproximator = policy.logApproximator
        NumPolicyParams = policyApproximator.NumParams

        if policyApproximator.parameter is None:
            policyApproximator.resetParameter(np.zeros(NumPolicyParams))
            
        NumCostParams = costApproximator.NumParams
        if costApproximator is None:
            costApproximator.resetParameter(np.zeros(NumCostParams))

            
        # Features = control params + covariance upper triangle + cost
        numFeatures = NumPolicyParams + NumCostParams

        # Auxilliary temporal difference variables
        z = np.zeros(numFeatures)
        A = np.zeros((numFeatures,numFeatures))
        b = np.zeros(numFeatures)

        x0 = np.array(SYS.x0,copy=True)
        #### Actor Critic Learning ####
        for episode in range(EpisodeCount):
            policy.Horizon=EpisodeLength

            if reset is False:
                if episode > 0:
                    SYS.x0 = X[-1]
            X,U,Cost = SYS.simulatePolicy(policy)
            if reset is False:
                if episode > 0:
                    SYS.x0 = np.array(x0,copy=True)

                
            ### Policy Evaluation by least squares temporal differences ###
            ### This should be changed to use any policy evaluation method ###
            TrueCost = 0.
            CostFactor = 1.
            for k in range(EpisodeLength-1):
                TrueCost += CostFactor * Cost[k]
                CostFactor *= DiscountFactor

                Z = np.hstack((X[k],U[k]))
                costGrad = costApproximator.parameterGradient(X[k])
                policyGrad = policyApproximator.parameterGradient(Z)


                PhiHat = np.hstack((costGrad,policyGrad))

                PhiTilde = np.zeros(numFeatures)
                newCostGrad = costApproximator.parameterGradient(X[k+1])

                PhiTilde[:NumCostParams] = newCostGrad
                
                z = TraceDecayFactor * z + PhiHat
                A += -np.outer(z,DiscountFactor*PhiTilde - PhiHat)
                b += z*Cost[k]


            # Quadratic learning
            featureWeights = la.solve(A,b)
            # Linear learning
            # if episode == 0:
            #     featureWeights = np.zeros(NumFeatures)

            costParams = featureWeights[:NumCostParams]
            policyParams = featureWeights[NumCostParams:]

            stepSize = stepSizeConstant / (episode+1)
            newCostParams = costApproximator.parameter + stepSize * costParams
            newPolicyParams = policyApproximator.parameter - \
                              stepSize * policyParams

            costApproximator.resetParameter(newCostParams)
            policy.resetParameter(newPolicyParams)

            costEst = costApproximator.value(X[0])

            print 'Episode Cost: %g, Estimated Episode Cost: %g, Undiscounted Cost: %g' % (TrueCost,costEst,Cost.sum())


            # Forget previous influence slightly
            z = ForgettingFactor * z
            A = ForgettingFactor * A
            b = ForgettingFactor * b

            # May need to figure out how to get the appropriate gain shape

        ctrl.noisyLinParamFun.__init__(self,
                                       basisFunction=policy.basis,
                                       NumInputs=SYS.NumInputs,
                                       NumStates=SYS.NumStates,
                                       parameter=policyApproximator.parameter,
                                       *args,**kwargs)
                                       
        
class actorCriticLQR(naturalActorCritic):
    """
    Specializeation of the actor critic method to LQR systems. 
    """
    def __init__(self,SYS,EpisodeLength=1,
                 Gain=None,Covariance=None,
                 *args,**kwargs):
        NumInputs = SYS.NumInputs
        NumStates = SYS.NumStates

        policyBasis = fa.createAffineBasisFunction(NumInputs)
        
        if Gain is not None:
            beta = np.reshape(Gain,np.prod(Gain.shape),order='F')
        else:
            beta = np.zeros((NumStates+1)*NumInputs)
        if Covariance is not None:
            C = la.cholesky(Covariance,lower=True)
        else:
            C = np.eye(NumInputs)

        Cs = fa.stackLower(C)

        param = np.hstack((beta,Cs))

        noisyLinearPol = ctrl.noisyLinParamFun(basisFunction=policyBasis,
                                               NumInputs=NumInputs,
                                               NumStates=NumStates,
                                               parameter=np.hstack((beta,Cs)),
                                               Horizon=EpisodeLength)


        n = NumStates
        numCostParams = (n+1)*(n+2)/2
        costParam = np.zeros(numCostParams)
        quadCost = fa.parameterizedQuadratic(NumVars=NumStates,
                                             parameter=costParam)

        naturalActorCritic.__init__(self,SYS,
                                    policy=noisyLinearPol,
                                    costApproximator=quadCost,
                                    EpisodeLength=EpisodeLength,
                                    *args,**kwargs)
