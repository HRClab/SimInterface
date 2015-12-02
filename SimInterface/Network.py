"""
Network
"""

import pandas as pd
from collections import deque

try:
    import graphviz as gv
    graphviz = True
except ImportError:
    graphviz = False


class Network:
    """
    The network
    """

    def __init__(self,Blocks=None,label='Net'):

        InternalVariables = []
        InputVariables = []
        OutputVariables = []

        Parents = dict()
        Children = dict()
        Executable = deque()
        
        for block in Blocks:
            Parents[block] = []
            Children[block] = []
            for v in block.InputVars:
                if v.Parent is None:
                    InputVariables.append(v)
                else:
                    InternalVariables.append(v)
                    Parents[block].append(v.Parent)
            if len(Parents[block]) == 0:
                Executable.append(block)
                
            for v in block.OutputVars:
                if v.Child is None:
                    OutputVariables.append(v)
                else:
                    InternalVariables.append(v)
                    Children[block].append(v.Child)

        # Find an order to execute the blocks
        BlockOrder = []
        while len(Executable) > 0:
            block = Executable.pop()
            BlockOrder.append(block)
            for child in Children[block]:
                Parents[child].remove(block)
                if len(Parents[child]) == 0:
                    Executable.append(child)

        self.BlockOrder = BlockOrder
        # Hack to remove duplicates
        InternalVariables = list(set(InternalVariables))

        self.InputVars = InputVariables
        self.OutputVars = OutputVariables
        self.InternalVars = InternalVariables

        self.InputData = pd.concat([v.data for v in InputVariables],
                                   axis=1,
                                   join='inner')

        self.OutputData = pd.concat([v.data for v in OutputVariables],
                                    axis=1,
                                    join='inner')
        self.InternalData = pd.concat([v.data for v in InternalVariables],
                                      axis=1,
                                      join='inner')
    
        self.__createGraph__(Blocks,label)
        
    def __createGraph__(self,Blocks,label):
        if graphviz:
            dot = gv.Digraph(name=label)
            InternalSignals = []

            for block in Blocks:
                dot.node(block.label,shape='box')
                for v in block.InputVars:
                    if v.Parent is None:
                        dot.node(v.label,label='',shape='plaintext')
                        dot.edge(v.label,block.label,label=v.label)
                    else:
                        InternalSignals.append(v)
                for v in block.OutputVars:
                    if v.Child is None:
                        dot.node(v.label,label='',shape='plaintext')
                        dot.edge(block.label,v.label,label=v.label)
                    else:
                        InternalSignals.append(v)
            InternalSignals = list(set(InternalSignals))
            for v in InternalSignals:
                dot.edge(v.Parent.label,v.Child.label,label=v.label)

            self.graph = dot

    def apply(self):
        for block in self.BlockOrder:
            block.apply()

        self.InternalData = pd.concat([v.data for v in self.InternalVars],
                                      axis=1,
                                      join='inner')
        self.OutputData = pd.concat([v.data for v in self.OutputVars],
                                    axis=1,
                                    join='inner')
