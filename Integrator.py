import MarkovDecisionProcess as MDP


class Integrator(MDP.LinearQuadraticSystem):
    """
    This is the simplest linear system dynamical system
    """
    def __init__(self):
        self.dt = 0.1
        self.x0 = 1.0
        A = 1.
        B = self.dt
        Q = self.dt * 1.
        R = self.dt * 1.
        dynMat = MDP.buildDynamicsMatrix(A,B)
        costMat = MDP.buildCostMatrix(Cxx=Q,Cuu=R)
        MDP.LinearQuadraticSystem.__init__(self,dynamicsMatrix=dynMat,
                                  costMatrix=costMat)
                
