import numpy as np

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
        u = np.zeros(self.NumInputs)
        return u

class openLoopPolicy(Controller):
    def __init__(self,U,Horizon=None, *args,**kwargs):
        horizon, numInputs = U.shape
        if Horizon is None:
            Horizon = horizon
        
        Controller.__init__(self,Horizon=Horizon,
                            NumInputs=numInputs,*args,**kwargs)
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

class varyingAffine(Controller):
    def __init__(self,gain,*args,**kwargs):
        self.Gain = gain
        Controller.__init__(self,*args,**kwargs)

    def action(self,x,k):
        vec = np.hstack((1,x))
        u = np.dot(self.Gain[k],vec)
        return u

class flatVaryingAffine(Controller):
    def __init__(self,flatGain,NumInputs=1,Horizon=1,*args,**kwargs):
        NumStates = -1 + len(flatGain) / (Horizon*NumInputs)
        self.NumStates = NumStates
        self.Gain = flatGain
        self.Stride = (self.NumStates+1) * NumInputs
        Controller.__init__(self,Horizon=Horizon,NumInputs=NumInputs,
                            *args,**kwargs)

    def action(self,x,k):
        GainMat = np.reshape(self.Gain[self.Stride*k:self.Stride*(k+1)],
                             (self.NumInputs,self.NumStates+1))
        vec = np.hstack((1,x))
        return np.dot(GainMat,vec)
        
class staticFunction(Controller):
    def __init__(self,func,*args,**kwargs):
        self.func = func
        Controller.__init__(self,*args,**kwargs)
        
    def action(self,x,k):
        u = self.func(x)
        return u


