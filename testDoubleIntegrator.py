import MarkovDecisionProcess as MDP
import Controller as ctrl
import numpy as np
import matplotlib.pyplot as plt

class doubleIntegrator(MDP.LinearQuadraticSystem):
    """
    Basic Newton's Laws
    """
    def __init__(self):
        dt = 0.1
        self.x0 = np.array([1.0,0.0])
        A = np.eye(2) + dt * np.array([[0,1],[0,0]])
        B = dt * np.array([0,1])
        Q = dt * np.eye(2)
        R = dt * 1.
        dynMat = MDP.buildDynamicsMatrix(A,B)
        costMat = MDP.buildCostMatrix(Cxx=Q,Cuu=R)
        MDP.LinearQuadraticSystem.__init__(self,
                                           dynamicsMatrix=dynMat,
                                           costMatrix=costMat)

sys = doubleIntegrator()
