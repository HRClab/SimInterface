"""
Function objects are the most basic objects in this package. Everything can be constructed from them.

Why make a special class for functions?
---------------------------------------

It is true that the functions that we use will all be functions in the classical sense. However, there are special features that occur repeatedly in simulation:

* Derivatives used in analysis and synthesis schemes, e.g:

  - Lie derivatives for Lyapunov function search

  - Vector field Jacobians for linearization

  - Quadratic Taylor series approximations of cost for approximate optimal control

* Time is a special parameter which functions may or may not depend on

* Variables must be easily and uniquely identified

  - For specifying input / output connections

  - For automatically identifying the relevant state space for feedback systems
"""

import pandas as pd
import numpy as np
try:
    import graphviz as gv
    graphviz = True
except ImportError:
    graphviz = False
    

class Function:
    """
    This is the basic Function class

    Parameters 
    """
    def __init__(self,func=None,InputVars=None,OutputVars=None,label='Fun'):
        self.func = func
        self.label = label

        if isinstance(InputVars,tuple):
            self.InputVars = InputVars
        else:
            self.InputVars = (InputVars,)


        self.setInputData()
        
        for v in self.InputVars:
            v.Targets.append(self)

        if isinstance(OutputVars,tuple):
            OutputData = pd.concat([v.data for v in OutputVars],
                                    axis=1,
                                    join='inner')
            self.OutputVars = OutputVars
        else:
            OutputData = OutputVars.data
            self.OutputVars = (OutputVars,)
            
        self.OutputData = OutputData
        for v in self.OutputVars:
            v.Source = self

        self.Vars = set(self.InputVars) | set(self.OutputVars)


    def __func__(self,iv):
        ov = self.func(*(iv[v.label] for v in self.InputVars))
        return pd.Series(list(np.hstack(ov)),
                         index=self.OutputData.columns)
    
    def setInputData(self):
        self.InputData = pd.concat([v.data for v in self.InputVars],
                                   axis=1,
                                   join='inner')
    def apply(self):
        self.setInputData()
        values = self.InputData.apply(self.__func__,axis=1)
        self.OutputData = pd.DataFrame(values,columns=self.OutputData.columns)
        groups = self.OutputData.groupby(level=0,axis=1)
        for ov in self.OutputVars:
            originalColumns = ov.data.columns
            ov.data = groups.get_group(ov.label)
