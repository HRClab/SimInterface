"""
Network
"""

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
