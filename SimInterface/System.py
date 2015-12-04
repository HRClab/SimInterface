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


import Variable as Var
import Function as Fun

class System:
    def __init__(self,StateFunction=None,OutputFunction=None,
                 stateVars=None,inputVars=None):
        if isinstance(StateFunction,Fun.Function):
            self.StateFunction = StateFunction
        else:
            self.StateFunction = Fun.Function(StateFunction,
                                              (stateVars,inputVars))
        if isinstance(OutputFunction,Fun.Function):
            self.OutputFunction = OutputFunction
        else:
            self.OutputFunction = Fun.Function(OutputFunction,
                                               (stateVars,outputVars))

    def connect(self):
        pass

class DifferentialEquation:
    def __init__(self,func=None,StateVars=None,InputVars=None,label=None):
        self.__createGraph__(StateVars,InputVars,label)

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
