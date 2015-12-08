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
import inspect as ins

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
        

        # We will now build an execution order for the output functions 
        Parents = dict()
        Children = dict()
        Executable = col.deque()
        self.ExecutionOrder = []

        for f in self.Funcs:
            Parents[f] = set(v.Source for v in f.InputVars) & self.Vars
            if (len(Parents[f]) == 0) and (isinstance(f,StaticFunction)):
                # If a function has no parents it is executable immediately
                # Only put static functions in the execution order
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
                if (len(Parents[child]) == 0) and \
                   (isinstance(child,StaticFunction)):
                    Executable.append(child)

        self.__createGraph()

        # Build a dictionary from labels to current values
        self.labelToValue = {v.label : np.array(v.data.iloc[0]) \
                             for v in self.Vars}
                             
        # Not sure if these are needed
        # Build a dictionary from functions to inputs
        self.funcToInputs = {f : f.InputVars for f in self.Funcs}
        # Also need a dictionary from functions to outputs
        self.funcToOutputs = {f : f.OutputVars for f in self.Funcs}

        ##### Things needed for Vector Field ######
        self.StateFuncs = [f for f in self.Funcs if len(f.StateVars)>0]
        self.funcToState = {f : f.StateVars for f in self.Funcs}

        StateVarSet = reduce(lambda a,b : a|b,
                             [f.StateVars for f in self.Funcs],
                             set())
        self.StateVars = list(StateVarSet)

        self.stateToFunc = {v : [] for v in self.StateVars}

        for f in self.StateFuncs:
            for v in f.StateVars:
                self.stateToFunc[v].append(f)

        # Create auxilliary states for exogenous signals
        self.InputSignals = [v for v in self.Vars if \
                             (v.Source not in self.Funcs) and \
                             (isinstance(v,Var.Signal))]

        self.IndexSlopes = []
        for v in self.InputSignals:
            slopeList = np.zeros(len(v.data.index))
            slopeList[:-1] = 1./np.diff(v.data.index)
            self.IndexSlopes.append(slopeList)
            
        ##### Initial Condition for ODE Integration ######
        Dimensions = [0]
        Dimensions.extend([v.data.shape[1] for v in self.StateVars])
        self.StateIndexBounds = np.cumsum(Dimensions)

        NumStates = len(self.StateVars)
        self.InitialState = np.zeros(self.StateIndexBounds[-1] + \
                                     len(self.InputSignals))
        for k in range(NumStates):
            InitVal = np.array(self.StateVars[k].data.iloc[0])
            indLow,indHigh = self.StateIndexBounds[k:k+2]
            self.InitialState[indLow:indHigh] = InitVal
            
        
    def UpdateState(self,Time=[],State=[]):
        for k in range(len(self.StateVars)):
            v = self.StateVars[k]
            # Probably better to just use a set method here
            # but this will work as long as the method for
            # initializing data does not change
            indLow,indHigh = self.StateIndexBounds[k:k+2]
            v.data = pd.DataFrame(State[:,indLow:indHigh],
                                  columns=v.data.columns,
                                  index=Time)

    def VectorField(self,Time,State):
        """
        Something suitable for passing to ODE methods.
        """

        State_dot = np.zeros(len(State))

        # Update the state values
        ## Split the states into a list
        NumStates = len(self.StateVars)

        for k in range(NumStates):
            v = self.StateVars[k]
            indLow,indHigh = self.StateIndexBounds[k:k+2]
            curVal = State[indLow:indHigh]
            self.labelToValue[v.label] = curVal
        
        # Compute the exogenous inputs 
        ##  The index states
        NumIndexStates = len(self.InputSignals)
        IndexStateList = State[-NumIndexStates:]

        IndexSlopes = np.zeros(NumIndexStates)
        
        for k in range(NumIndexStates):
            ctsIndex = IndexStateList[k]
            curInd = int(np.floor(ctsIndex))
            nextInd = curInd+1

            IndexSlopes[k] = self.IndexSlopes[k][curInd]

            v = self.InputSignals[k]
            # Linearly interpolate exogenous inputs
            # Presumably this could help smoothness.
            # and it is not very hard. 
            prevInput = v.data.iloc[curInd]
            nextInput = v.data.iloc[nextInd]
            lam = IndexStateList[k] - curInd
            # this can be called later.
            inputVal = (1-lam) * prevInput + lam * nextInput
            self.labelToValue[v.label] = np.array(inputVal)

        ## Plug in the derivative of the index slopes. 
        State_dot[-NumIndexStates:] = IndexSlopes
    
        # Apply the static functions in the appropriate order 
        ## Handle this once we actually have a static function.
        ## This will involve updating self.labelToValue
        
        # Apply the vector fields

        ## Compute vector field        
        for k in range(NumStates):
            v = self.StateVars[k]
            dvdt = np.zeros(v.data.shape[1])
            for f in self.stateToFunc[v]:
                argList = ins.getargspec(f.func)[0]
                # Need to map labels to variables
                # and then map variables to current values
                valList = [self.labelToValue[lab] for lab in argList]
                dvdt += f.func(*valList)

            indLow,indHigh = self.StateIndexBounds[k:k+2]
            State_dot[indLow:indHigh] = dvdt

        return State_dot

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

        

class Function(System):
    def __init__(self,func=lambda : None,label='Fun',
                 StateVars = set(), InputVars = set(), OutputVars = set()):

        self.func = func
        self.label = label
        StateVarSet = castToSet(StateVars)
        InputVarSet = castToSet(InputVars)
        OutputVarSet = castToSet(OutputVars)

        self.StateVars = StateVarSet
        self.InputVars = InputVarSet
        self.OutputVars = OutputVarSet

        self.Vars = StateVarSet | InputVarSet | OutputVarSet

        map(lambda v : v.Targets.add(self),StateVarSet | InputVarSet)
        for v in OutputVarSet:
            v.Source = self

class StaticFunction(Function):
    def __init__(self,func=None,InputVars=None,OutputVars=None,label='Fun'):
        Function.__init__(func=func,label=label,
                          InputVars=InputVars,OutputVars=OutputVars)
        
class DifferentialEquation(System):
    def __init__(self,func=None,StateVars=None,InputVars=None,
                 Time=None,label='DiffEq'):
        
        # Dummy signals for the time derivatives
        # These "outputs" are fed into a dummy integrator function
        
        OutputVars = set()
        StateVarSet = castToSet(StateVars)
        
        for v in StateVarSet:
            dvdt = Var.Signal(label='d%s/dt' % v.label,
                              data=np.zeros((1,v.data.shape[1])),
                              TimeStamp=np.zeros(1))
            OutputVars.add(dvdt)

        VectorField = Function(func=func,label=label,
                               InputVars=InputVars,
                               OutputVars=OutputVars,
                               StateVars=StateVars)

        Integrator = Function(label='Integrator',
                              InputVars=OutputVars,
                              OutputVars=StateVars)

        System.__init__(self,
                        Funcs=set([VectorField,Integrator]),label=label)
