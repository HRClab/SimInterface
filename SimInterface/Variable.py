"""

Variable 

"""

import numpy as np
import pandas as pd

class Variable:
    def __init__(self,label='Var',data=None,shape=None,TimeStamps=None):
        self.__createDataFrame(label,data,shape,TimeStamps)
        
        self.label = label
        self.Source = None
        self.Targets = []

    def __createDataFrame(self,label,data,shape,TimeStamps):
        """
        An internal function to create a pandas DataFrame object 
        """
        if shape is None:
            # Scalar
            columns = [label]
        else:
            NumEl = np.prod(shape)
            indices = range(NumEl)
            subscriptTuples = np.unravel_index(indices,shape)
            subscriptList = [np.tile(label,NumEl)]
            subscriptList.extend(subscriptTuples)
            columns = pd.MultiIndex.from_arrays(subscriptList)

        if data is None:
            if shape is None:
                dataMat = 0.0
            else:
                dataMat = np.zeros((1,NumEl))
        elif not isinstance(data,np.ndarray):
            # Scalar input
            dataMat = np.array([[data]])
        elif data.shape == shape:
            dataMat = np.array([data.flatten()])
        elif data.shape == (NumEl,):
            dataMat = np.array([data])
        elif data.shape == (1,NumEl):
            dataMat = np.array(data,copy=True)
        elif np.prod(data.shape) > NumEl:
            if data.shape[1:] == (NumEl,):
                dataMat = np.array(data,copy=True)
            elif data.shape[1:] == shape:
                dataMat = np.reshape(data,(len(data),NumEl))
                

        T,n = dataMat.shape
        if TimeStamps is None:
            TimeStamps = np.arange(T)

        self.data = pd.DataFrame(dataMat,
                                 columns=columns,
                                 index=TimeStamps[:T])
        

    def __getitem__(self,item):
        return self.data[item]

class Parameter(Variable):
    def __init__(self,label='Var',data=None,shape=None):
        Variable.__init__(self,label,data,shape)
    
class Signal(Variable):
    """
    This is a variable

    Parameters
       label : str
          what to call the variable

       value : scalar or array_like

    """

    def __init__(self,label='Var',data=None,shape=None,TimeStamps=None):
        """

        """

        Variable.__init__(self,label,data,shape,TimeStamps)

        
