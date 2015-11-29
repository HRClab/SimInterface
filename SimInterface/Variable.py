"""

Variable 

"""

import numpy as np

class Variable:
    """
    This is a variable

    Parameters
       label : str
          what to call the variable

       value : scalar or array_like

    """

    def __init__(self,label='Var',value=0.0):
        self.label = label
        self.value = value

    def __str__(self):
        return self.label

    def setValue(self,Val):
        self.value = Val

    def getValue(self):
        return self.value
    
class VarArray(np.ndarray):
    def __new__(cls,label='Var',shape=(1,),value=None):

        if value is not None:
            shape = value.shape
            
        varInd = np.arange(np.prod(shape))
        varBuf = np.zeros(shape,dtype=object)
        valStr = np.zeros(shape,dtype=object)
        for ind in varInd:
            sub = np.unravel_index(ind,shape)
            varLabel = label+"_".join(str(num) for num in sub)
            valStr[sub] = varLabel
            var = Variable(label=varLabel)
            varBuf[sub] = var

        obj = np.ndarray.__new__(cls,shape,buffer=varBuf,dtype=object)
        obj.str = valStr.__str__()

        if value is None:
            obj.value = np.zeros(shape)
        else:
            obj.value = value
        return obj

    def __array_finalize__(self, obj):
        pass

    def __str__(self):
        return self.str

    def setValue(self,Val):
        self.value = Val
        

    def getValue(self):
        return self.value
        
    
