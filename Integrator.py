import MDP


class Integrator(MDP.MDP):
    def __init__(self):
        self.dt = 0.1
        self.x0 = 1.0
        
    def costStep(self,x,u,k):
        return self.dt*(x*x+u*u)

    def step(self,x,u,k):
        return x+self.dt*u
        
