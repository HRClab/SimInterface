import MarkovDecisionProcess as MDP
import Controller as ctrl
import numpy as np
from numpy.random import randn
import matplotlib.pyplot as plt

class epileptor(MDP.driftDiffusion):
    """
    Epileptor Model as defined by 
    Jirsa et. al. On the nature of seizure dynamics

    In 
    """
    def __init__(self):
        dt = 0.1

        # Parameters
        x0 = -1.6             # Not initial condition
        y0 = 1.0
        tau0 = 2857
        tau1 = 1
        tau2 = 10
        Irest1 = 3.1
        Irest2 = 0.45
        gamma = 0.01
        
        def drifFunc(x,u,k):
            x1,y1,x2,y2,z,g = x

            if x1 < 0:
                f1 = x1**3. - 3*x1**2.
            else:
                f1 = (x2 - 0.6*(z-4.)**2.) * x1

            if x2 < -0.25:
                f2 = 0
            else:
                f2 = 6 * (x2+0.25)
                
            dx1 = y1 - f1 - z + Irest1
            dy1 = y0 - 5*x1**2. - y1
            dx2 = -y1 + x2 - x2**3. + Ires2 + 0.002 * g -0.3*(z-3.5)
            dy2 = (-y2 + f2) / tau2
            dz = (1./tau0) * (4*(x1-x0)-z) + u
            dg = -gamma * g + x1

            dx = np.array([dx1,dy1,dz,dx2,dy2,dg])
            return dx

        def noiseFun(x,u,k):
            
        
    
    
sys = epileptor()
