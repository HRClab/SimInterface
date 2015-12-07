"""

Variable 

"""

import numpy as np
import pandas as pd

class Parameter:
    def __init__(self):
        pass
class Signal:
    """
    This is a variable

    Parameters
       label : str
          what to call the variable

       value : scalar or array_like

    """

    def __init__(self,label='Var',data=None,shape=None):
        """

        """

        self.__createDataFrame__(label,data,shape)
        
        self.label = label
        self.Source = None
        self.Targets = []

    def __createDataFrame__(self,label,data,shape):
        """
        An internal function to create a pandas DataFrame object 
        """
        if shape is None:
            # Scalar
            columns = label
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
                

        self.data = pd.DataFrame(dataMat,columns=columns)
        

    def __getitem__(self,item):
        return self.data[item]


        
