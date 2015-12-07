"""
The fundamental objects of the SimInterface are systems. 

Each system is characterized by a collection of spaces and mappings between them. 

The spaces are given by:

* U - Input Space
* X - State Space
* Y - Output Space

The mappings are given by:

* :math:`x' = f(x,u)`
* :math:`y = g(x,u)`

In general the mappings could be deterministic, random, uncertain, etc. It would be sufficient, however, to assume everything is deterministic, and to assume that randomness enters as explicit inputs. 

For easiest analysis, it would be helpful if every variable was either an input,
and output, or a state. 

How do we deal with synthesis problems? In a synthesis problem, we typically pass some description of the system as a parameter. We've seen before that this problem is really only meaningful in the intialization of the controller. Once it is plugged in, it becomes just another system. 


What we need are the following:

* Composition rules for systems
* A typing system that can be used to specify allowable operations on systems
* A method for computing and visualizing the interconnections between systems. 
* A collection of allowable simulation rules 

# What type information do we need to track?

* We must track differentiability with respect to the signals  and the corresponoding derivatives. This is for controller / filter design methods that rely on derivative information. 
* We must track the constraints on allowable simulation methods. For example, if all systems operate in continuous time, then all relevant continuous-time simulation methods should work. If we are composing a continuous-time system with a sampled-data system, then this should be explicit. If we have some event triggering, well, you should probably use a different engine. 
* We must track constraints on simulation time. For example, systems that result from linearizing around a trajectory will typically be only defined for a finite set of sampling instants. We must be clear about this. In other words, a relevant "time domain" is required. 

# How can we track differentiability?

Perhaps the cleanest way to do it would be to have a derivative matrix/tensor for each function, with respect to each signal. Basically, it would be some way to specify derivatives as needed. Then we could use just chain-rule-based techniques to propagate the derivatives through the system. 

# What are relevant simulation techniques?

* The basic for-loop method
* Euler / Runge-Kutta / other time-stepping schemes 
* Quadrature

# Does the time-domain specify the applicable simulation methods?

I think so. 

# Do we need spaces explicitly? 

I would argue that we do. For instance, if you performed a naive sequence of 
series interconnections, and then one of them connected back on the original 
system, the space would be too big. It also would not lead to correct results. 

However, we may not need to actually need to specify them explicitly, as long as a system graph is maintained. Namely, there should be a graph algorithm to identify the appropriate state space 

# What information should be exposed to analysis / synthesis programs?

* In many cases, you would pass the entire system object to the controller / filter, and it would construct the controller / filter based on the relevant data. 




Coding Needs:

* Know how to make unique identifiers. 

Definitely need to figure out how to use RST for 

* Latex
* Headings


"""

try:
    import graphviz as gv
    graphviz = True
except ImportError:
    graphviz = False

import pandas as pd
import numpy as np
import collections as col
import Variable as Var
import Function as Fun

def castToTuple(Vars):
    if Vars is None:
        return tuple()
    elif isinstance(Vars,tuple):
        return Vars
    else:
        return (Vars,)

def castToSet(S):
    if isinstance(S,set):
        return S
    elif isinstance(S,col.Iterable):
        return set(S)
    elif S is None:
        return set()
    else:
        return set([S])
    

class System:
    """
    I think a better solution would be obtained by just forcing
    StateFunc, and OutputFuncs to simply be function objects, so that
    the variables would just inherit.
    """
    def __init__(self,Funcs=set(),label=''):
        self.label=label
        self.__buildSystem(Funcs)
            

    def add(self,func):
        Funcs = self.Funcs | set([func])
        self.__buildSystem(Funcs)

    def update(self,NewFuncs):
        NewFuncSet = castToSet(NewFuncs)
        Funcs = self.Funcs | NewFuncSet
        self.__buildSystem(Funcs)
        
    def __buildSystem(self,Funcs):
        self.Funcs = castToSet(Funcs)
        
        # Get all the variables
        self.Vars = reduce(lambda a,b : a|b,
                           [f.Vars for f in self.Funcs],
                           set())
        
        # Build a dictionary from functions to inputs
        self.funcToInputs = {f : f.InputVars for f in self.Funcs}
        # Also need a dictionary from functions to outputs
        self.funcToOutputs = {f : f.OutputVars for f in self.Funcs}

        # We will now build an execution order for the output functions 
        Parents = dict()
        Children = dict()
        Executable = col.deque()
        self.ExecutionOrder = []

        for f in self.Funcs:
            Parents[f] = set(v.Source for v in f.InputVars) & self.Vars
            if len(Parents[f]) == 0:
                # If a function has no parents it is executable immediately
                Executable.append(f)

            if f in Parents[f]:
                # Figure out how to make an error statement here.
                print 'Not well-posed, function depends on itself'
                
        # # For convencience we also construct the inverse dictionary
        # # For each function, the set of functions that it is used to produce
        Children = {f : set() for f in self.Funcs}
        for f in self.Funcs:
            for g in Children[f]:
                Children[g].union(set(f))

        # Now finally we create the execution order
        while len(Executable) > 0:
            f = Executable.pop()
            self.ExecutionOrder.append(f)
            for child in Children[f]:
                Parents[child].remove(f)
                if len(Parents[child]) == 0:
                    Executable.append(child)

        
        self.__createGraph()

    def __createGraph(self):
        """
        Create a graph using the graphviz module.

        It may be advisable to make this a bit more separated. 

        Namely, make a separate add-on that you pass the system to and it
        would produce a graph. 

        Basically make a separate submodule called "SystemGraph"
        """
        if not graphviz:
            return
        
        dot = gv.Digraph(name=self.label)

        for f in self.Funcs:
            dot.node(f.label,shape='box')

        for v in self.Vars:
            if v.Source not in self.Funcs:
                dot.node(v.label,label='',shape='plaintext')
                for tar in (set(v.Targets) & self.Funcs):
                    dot.edge(v.label,tar.label,label=v.label)

            else:
                for tar in (set(v.Targets) & self.Funcs):
                    dot.edge(v.Source.label,tar.label,label=v.label)

            if len(set(v.Targets) & self.Funcs) == 0:
                dot.node(v.label,label='',shape='plaintext')
                if v.Source in self.Funcs:
                    dot.edge(v.Source.label,v.label,label=v.label)

        self.graph = dot
        

class DifferentialEquation:
    def __init__(self,func=None,StateVars=None,InputVars=None,
                 Time=None,label='DiffEq'):

        self.label = label
        # Split the variables between parameters and signals.

        InputVarSet = castToSet(InputVars)
        StateVarSet = castToSet(StateVars)
        Vars = VarSet | StateVarSet

        self.InputVars = InputVarSet

        # The current function is a target of all of both the
        # input variables and the state variables
        map(lambda v : v.Targets.add(self),[v for v in Vars])

        
        # Now define a dummy signal for the time derivative

        self.OutputVars = set()
        for v in StateVarSet:
            dvdt = Var.Signal(label='d%s/dt' % v.label,
                              data=np.zeros((1,v.data.shape[1])),
                              TimeStamp=np.zeros(1))
            self.OutputVars.add(v)

        
