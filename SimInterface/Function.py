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

import Variable as Var

class Function:
    """
    This is the basic Function class

    Parameters 
        TimeVarying : bool (optional)
            Specify if time-varying or time-invariant  
            Default: `True`

        TimeDomain : str {'continuous', 'discrete'}, optional 
            Specify if continuous or discrete time, or neither

        TimeSpan : array_like (optional)
            An array of time points where the function is defined. 

    """
    def __init__(self,func=None,vars=None):
        self.func = func
        self.vars = vars

    def getValue(self):
        # Need a way to accomodate multiple variables here.
        return self.func(self.vars.getValue())
