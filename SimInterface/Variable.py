"""

Variable 

"""

import numpy as np
import pandas as pd

class Variable:
    def __init__(self,label='Var',data=None,TimeStamp=None):
        self.label = label
        self.__createDataFrame(data,TimeStamp)

        self.Source = None
        self.Targets = set()

    def __createDataFrame(self,data,TimeStamp):
        """
        An internal function to create a pandas DataFrame object 
        """
        shape = data.shape[1:]
            
        if len(shape) > 0:
            # Not Scalar
            NumEl = np.prod(shape)
            indices = range(NumEl)
            subscriptTuples = np.unravel_index(indices,shape)
            subscriptList = [np.tile(self.label,NumEl)]
            subscriptList.extend(subscriptTuples)
            print subscriptList
            subScriptTup = (np.tile(self.label,NumEl),) + subscriptTuples
            columns = pd.MultiIndex.from_arrays(subscriptTuples)
        else:
            columns = [self.label]

        self.columns = columns

        if len(shape) == 1:
            # Already have a table
            dataMat = data
        else:
            dataMat = np.reshape(data,(len(data),np.prod(shape)))

        self.setData(dataMat,TimeStamp)

    def setData(self,dataMat,TimeStamp):
        TimeIndices = pd.MultiIndex.from_arrays([np.tile('Time',len(TimeStamp)),
                                                 TimeStamp])

        self.data = pd.DataFrame(dataMat,
                                 columns=self.columns,
                                 index=TimeIndices)


    def __getitem__(self,item):
        return self.data[item]

class Parameter(Variable):
    def __init__(self,label='Par',data=None,shape=None):
        # Assume that either data or shape is not none
        if data is None:
            data = np.zeros(shape)

        # Cast a scalar to an array
        if not isinstance(data,np.ndarray):
            data = np.array([data])

        # Create a time-stamp
        TimeStamp = np.array([0])
        
        Variable.__init__(self,label,data,TimeStamp)
    
class Signal(Variable):
    """
    This is a variable

    Parameters
       label : str
          what to call the variable

       value : scalar or array_like

    """

    def __init__(self,label='Sig',data=None,shape=None,TimeStamp=None):
        """

        """

        # Assume data or shape is not none
        # shape is the dimensions of the data at each time step
        if data is None:
            data = np.zeros((1,)+shape)
        elif not isinstance(data,np.ndarray):
            data = np.array([data])
            
        # Now assume that the first index is is for time
        if TimeStamp is None:
            TimeStamp = np.arange(len(data))
        elif not isinstance(TimeStamp,np.ndarray):
            TimeStamp = np.array([TimeStamp])

        if len(TimeStamp) < len(data):
            print 'Not enough time stamps'        

        Variable.__init__(self,label,data,TimeStamp[:len(data)])

        
