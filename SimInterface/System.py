"""
The fundamental objects of the SimInterface are systems. 

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
    elif isinstance(Vars,col.Iterable):
        return set(Vars)
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

        # Build a dictionary from labels to current values
        self.labelToValue = {v.label : np.array(v.data.iloc[0]) \
                             for v in self.Vars}
                             

        ##### Things needed for Vector Field ######
        self.StateFuncs = [f for f in self.Funcs if len(f.StateVars)>0]

        StateVarSet = reduce(lambda a,b : a|b,
                             [set(f.StateVars) for f in self.Funcs],
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
            TimeIndex = np.array(v.data.index.levels[1])
            slopeList = 1./np.diff(TimeIndex)
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
            
        self.__createGraph()

    def UpdateParameters(self):
        Parameters = [v for v in self.Vars if isinstance(v,Var.Parameter)]
        for v in Parameters:
            self.labelToValue[v.label] = np.array(v.data.iloc[0])
        
    def UpdateSignals(self,Time=[],State=[]):
        T = len(Time)
        Signals = set([v for v in self.Vars if isinstance(v,Var.Signal)])
        InternalSignals = Signals - set(self.InputSignals)
        NewData = {v : np.zeros((T,v.data.shape[1])) \
                   for v in InternalSignals}
        for k in range(len(Time)):
            t = Time[k]
            S = State[k]
            self.__setSignalValues(t,S)
            for v in InternalSignals:
                NewData[v][k] = self.labelToValue[v.label]

        for v in InternalSignals:
            # Need a reset 
            v.setData(NewData[v],Time)

    def __setSignalValues(self,Time,State):
        # Set State Values
        NumStates = len(self.StateVars)
        for k in range(NumStates):
            v = self.StateVars[k]
            indLow,indHigh = self.StateIndexBounds[k:k+2]
            curVal = State[indLow:indHigh]
            self.labelToValue[v.label] = curVal

        # Set Input Signal Values

        NumIndexStates = len(self.InputSignals)
        IndexStateList = State[-NumIndexStates:]

        for k in range(NumIndexStates):
            ctsIndex = IndexStateList[k]
            curInd = int(np.floor(ctsIndex))
            nextInd = curInd+1

            if nextInd < len(self.IndexSlopes[k]):
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
            else:
                # If out of bounds just stay at the last value.
                self.labelToValue[v.label] = np.array(v.data.iloc[-1])
        

        # Set Intermediate Signal Values

        for f in self.ExecutionOrder:
            argList = ins.getargspec(f.func)[0]
            valList = [self.labelToValue[lab] for lab in argList]
            outTup = f.func(*valList)

            if len(f.OutputVars) > 1:
                for k in len(f.OutputVars):
                    outVariable = f.OutputVars[k]
                    outValue = outTup[k]
                    self.labelToValue[outVariable.label] = outValue
            else:
                self.labelToValue[f.OutputVars[0].label] = outTup

        
    def VectorField(self,Time,State):
        """
        Something suitable for passing to ODE methods.
        """
        State_dot = np.zeros(len(State))

        self.__setSignalValues(Time,State)
        NumStates = len(self.StateVars)    
                
        # Apply the vector fields

        ## Compute
        NumIndexStates = len(self.InputSignals)
        IndexStateList = State[-NumIndexStates:]
        IndexSlopes = np.zeros(NumIndexStates)
        
        for k in range(NumIndexStates):
            ctsIndex = IndexStateList[k]
            curInd = int(np.floor(ctsIndex))
            nextInd = curInd+1
            if nextInd < len(self.IndexSlopes[k]):
                # Not too near end
                IndexSlopes[k] = self.IndexSlopes[k][curInd]
            else:
                # If out of bounds just stay at the last value.
                IndexSlopes[k] = 0.

        ## Plug in the derivative of the index slopes. 
        State_dot[-NumIndexStates:] = IndexSlopes    
        
        ## Compute main vector field 
        dvdt = {v : np.zeros(v.data.shape[1]) for v in self.StateVars}
        for f in self.StateFuncs:
            argList = ins.getargspec(f.func)[0]
            valList = [self.labelToValue[lab] for lab in argList]
            dxdt = f.func(*valList)
            nx = len(f.StateVars)

            # output may or may not be a tuple. 
            if nx > 1:
                for k in range(nx):
                    dvdt[f.StateVars[k]] += dxdt[k]
            else:
                dvdt[f.StateVars[0]] += dxdt
            
        for k in range(NumStates):
            indLow,indHigh = self.StateIndexBounds[k:k+2]
            State_dot[indLow:indHigh] = dvdt[self.StateVars[k]]
        
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
            self.graph = None
            return
        
        dot = gv.Digraph(name=self.label)

        # Ignore the integrators
        NonIntegrators = set([f for f in self.Funcs if f.ftype != 'integrator'])

        for f in NonIntegrators:
            dot.node(f.label,shape='box')

        # Handle state vars and nonstate vars separately
        StateVars = set(self.StateVars)
        NonState = self.Vars - StateVars
        for v in self.Vars:
            if (v.Source not in NonIntegrators) and (v in NonState):
                dot.node(v.label,shape='plaintext')
                for tar in (set(v.Targets) & NonIntegrators):
                    dot.edge(v.label,tar.label)

            else:
                for tar in (set(v.Targets) & NonIntegrators):
                    if v.Source in NonIntegrators:
                        dot.edge(v.Source.label,tar.label,label=v.label)

            if len(set(v.Targets) & self.Funcs) == 0:
                dot.node(v.label,shape='plaintext')
                if v.Source in NonIntegrators:
                    dot.edge(v.Source.label,v.label)

        # Special handling for states
        for v in StateVars:
            for f in self.stateToFunc[v]:
                for g in (v.Targets & NonIntegrators):
                    dot.edge(f.label,g.label,label=v.label)
        self.graph = dot

def Connect(Systems=set(),label='Sys'):
    Funcs = reduce(lambda a,b : a | b, [S.Funcs for S in Systems],set())

    return System(Funcs,label)
    
class Function(System):
    def __init__(self,func=lambda : None,label='Fun',
                 StateVars = tuple(),
                 InputVars = tuple(),
                 OutputVars = tuple(),
                 ftype=None):

        self.func = func
        self.label = label
        self.ftype = ftype

        # In general, ordering of variables matters.
        # This is especially true of parsing outputs
        self.StateVars = castToTuple(StateVars)
        self.InputVars = castToTuple(InputVars)
        self.OutputVars = castToTuple(OutputVars)

        # Sometimes, however, we only care about
        # an un-ordered set
        StateSet = set(self.StateVars)
        InputSet = set(self.InputVars)
        OutputSet = set(self.OutputVars)
        
        self.Vars = StateSet | InputSet | OutputSet

        map(lambda v : v.Targets.add(self),StateSet | InputSet)
        for v in OutputSet:
            v.Source = self

        System.__init__(self,self,label)

class StaticFunction(Function):
    def __init__(self,func=None,InputVars=None,OutputVars=None,label='Fun'):
        Function.__init__(self,func=func,label=label,ftype='static',
                          InputVars=InputVars,OutputVars=OutputVars)
        
class DifferentialEquation(System):
    def __init__(self,func=None,StateVars=None,InputVars=None,
                 Time=None,label='DiffEq'):
        
        # Dummy signals for the time derivatives
        # These "outputs" are fed into a dummy integrator function
        
        OutputVars = []

        
        StateVars = castToTuple(StateVars)

        # Make an ordered list of derivative variables corresponding
        # to the state variables. 
        for v in StateVars:
            dvdt = Var.Signal(label='d%s/dt' % v.label,
                              data=np.zeros((1,v.data.shape[1])),
                              TimeStamp=np.zeros(1))
            OutputVars.append(dvdt)

        
        VectorField = Function(func=func,label=label,
                               InputVars=InputVars,
                               OutputVars=OutputVars,
                               StateVars=StateVars,
                               ftype='vector_field')
        

        Integrator = Function(label='Integrator',
                              InputVars=OutputVars,
                              OutputVars=StateVars,
                              ftype='integrator')

        System.__init__(self,
                        Funcs=set([VectorField,Integrator]),label=label)
