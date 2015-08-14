import LinearQuadraticSystem as LQS


class Integrator(LQS.LinearQuadraticSystem):
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
        LQS.LinearQuadraticSystem.__init__(self,A=A,B=B,Cxx=Q,Cuu=R)
                
