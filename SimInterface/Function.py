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
            InputData = pd.concat([v.data for v in InputVars],
                                  axis=1,
                                  join='inner')
            self.InputVars = InputVars
        else:
            InputData = InputVars.data
            self.InputVars = (InputVars,)

        self.InputData = InputData
        for v in self.InputVars:
            v.Child = self

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
            v.Parent = self
        

        self.__createGraph__(InputVars,OutputVars,label)

    def __createGraph__(self,InputVars,OutputVars,label):
        if isinstance(InputVars,tuple):
            InputNodes = [IV.label for IV in InputVars]
        else:
            InputNodes = [InputVars.label]

        if isinstance(OutputVars,tuple):
            OutputNodes = [OV.label for OV in OutputVars]
        else:
            OutputNodes = [OutputVars.label]

        if graphviz:
            dot = gv.Digraph(name=label)

            dot.node(label,shape='box')
            
            for IN in InputNodes:
                dot.node(IN,label='',shape='plaintext')
                dot.edge(IN,label,label=IN)

            for ON in OutputNodes:
                dot.node(ON,label='',shape='plaintext')
                dot.edge(label,ON,label=ON)

            self.graph = dot

    def __func__(self,iv):
        ov = self.func(*(iv[v] for v in self.InputNodes))
        return pd.Series(list(np.hstack(ov)),
                         index=self.OutputData.columns)

    def viewNetwork(self):
        self.graph.view()
        
    def apply(self):
        self.OutputData = self.InputData.apply(self.__func__,axis=1)
        for ov in self.OutputVars:
            ov.data = self.OutputData[ov.label]
