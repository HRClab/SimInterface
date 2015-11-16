import numpy as np
import scipy.linalg as la
    

class temporalDifferenceLearner:
    def __init__(self,approximator=None,
                 discountFactor=1.,eligibilityVector=None,
                 traceDecayFactor=0.,stepSizeConstant=1.,
                 label=''):

        self.label=label
        self.approximator=approximator
        self.stepSizeConstant = stepSizeConstant
        self.stepNumber = 1

        if eligibilityVector is None:
            self.z = np.zeros(approximator.NumParams)
        else:
            self.z = elibilityVector

        self.discountFactor = discountFactor
        self.traceDecayFactor = traceDecayFactor

    def temporalDifference(self,val,x,xNew):
        newVal =  val + self.discountFactor * self.approximator.value(xNew)
        self.d = newVal - self.approximator.value(x)

    def stepSize(self):
        return self.stepSizeConstant / self.stepNumber

    def updateParameter(self):
        stepSize = self.stepSize()
        newParam = self.approximator.parameter + \
                   stepSize * self.d * self.z
        self.approximator.resetParameter(newParam)

    def updateEligibilityVector(self,x):
        currentGrad = self.approximator.parameterGradient(x)
        self.z = self.traceDecayFactor * self.discountFactor * self.z + \
                 currentGrad
    def step(self,measuredValue,x,xNew):
        # Temporal Difference
        self.temporalDifference(measuredValue,x,xNew)
        self.updateParameter()
        
        # Update TD stuff
        self.updateEligibilityVector(x)
        self.stepNumber += 1

class lstdLearner:
    def __init__(self,approximator=None,
                 traceDecayFactor=0.,discountFactor=1.,
                 z=None,b=None,A=None,
                 label=''):

        self.label=label
        self.approximator = approximator
        self.traceDecayFactor = traceDecayFactor
        self.discountFactor = discountFactor

        q = approximator.NumParams
        if z is None:
            self.z = np.zeros(q)
        else:
            self.z = z

        if b is None:
            self.b = np.zeros(q)
        else:
            self.b = b

        if A is None:
            self.A = np.zeros((q,q))
        else:
            self.A = A
                


    def temporalDifference(self,val,x,xNew):
        self.b += self.z * val
        gradDiff = self.approximator.parameterGradient(x) - \
                   self.discountFactor * \
                   self.approximator.parameterGradient(xNew)
        self.A += np.outer(self.z,gradDiff)

    def updateEligibilityVector(self,x):
        currentGrad = self.approximator.parameterGradient(x)
        self.z = self.traceDecayFactor * self.discountFactor * self.z + \
                 currentGrad
        
    def step(self,measuredValue,x,xNew,updateParameter=False):
        # Just compute the tempo
        self.temporalDifference(measuredValue,x,xNew)
        self.updateEligibilityVector(x)
        if updateParameter is True:
            self.updateParameter()

    def updateParameter(self):
        Ainv = la.pinv2(self.A,rcond=1e-6)
        newParam = np.dot(Ainv,self.b)
        self.approximator.resetParameter(newParam)
        
