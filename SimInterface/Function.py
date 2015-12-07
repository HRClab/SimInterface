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
    

class StaticFunction:
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

class DifferentialEquation:
    def __init__(self,func=None,StateVars=None,InputVars=None,
                 Time=None,label=None):

        # First create input dataFrame
        if InputVars is None:
            InputData is None
        elif not isinstance(InputVars,tuple):
            self.InputVars = (InputVars,)
            InputData = InputVars.data
        else:
            self.InputVars = InputVars
            InputData = pd.concat([v.data for v in InputVars],
                                  axis=1,
                                  join='inner')

        # Check if there is exogenous input

        # Need to rework this. 
        if (Time is not None) and (InputData is not None):
            InputData.index = Time[:len(InputData)]

            
        self.InputData = InputData

        if not isinstance(StateVars,tuple):
            self.StateVars = (StateVars,)
        else:
            self.StateVars = StateVars
            
        StateData = pd.concat([v.data for v in self.StateVars],
                              axis=1,
                              join='inner')

        self.StateData = StateData

        self.StateGroups = StateData.groupby(level=0,axis=1)
        
        self.__createGraph__(StateVars,InputVars,label)


    def vectorField(self,t,x):
        """
        With what we have, how could we get it done?

        * Build a state variable dataframe

            * This fixes the order of the variables

            * Find out what the lengths are of each to map back

        * Build an input variable dataframe

        * Timestamp the rows.
        
        """

        Dimensions = [0]
        Dimensions.extend([v.data.shape[1] for v in self.StateVars])
        IndexBounds = np.cumsum(Dimensions)
        StateList = [x[i:j] for i,j in zip(IndexBounds[:-1],IndexBounds[1:])]

        if self.InputData is not None:
            TimeIndex = np.argwhere(self.InputData.index < t)[-1,0]
            print TimeIndex
            InputList = [self.InputData.iloc[TimeIndex][v.label] for \
                         v in self.InputVars]
            

    def __createGraph__(self,StateVars,InputVars,label):
        if graphviz:

            dot = gv.Digraph(name=label)

            dot.node(label,shape='box')
            
            if isinstance(InputVars,tuple):
                InputNodes = [IV.label for IV in InputVars]
            elif InputVars is not None:
                InputNodes = [InputVars.label]

            if InputVars is not None:
                for IN in InputNodes:
                    dot.node(IN,label='',shape='plaintext')
                    dot.edge(IN,label,label=IN)


            if isinstance(StateVars,tuple):
                OutputNodes = [OV.label for OV in StateVars]
            else:
                OutputNodes = [StateVars.label]

            for ON in OutputNodes:
                dot.node(ON,label='',shape='plaintext')
                dot.edge(label,ON,label=ON)

            dot.node('integrator',shape='box')

            DerivativeLabels = ['d%s/dt' % ON for ON in OutputNodes]
            DLabel = ','.join(DerivativeLabels)

            SLabel = ','.join(OutputNodes)

            dot.edge(label,'integrator',label=DLabel)
            dot.edge('integrator',label,label=SLabel)
                
            self.graph = dot
