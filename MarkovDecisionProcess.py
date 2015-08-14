import numpy as np

class MarkovDecisionProcess:
    def __init__(self):
        self.x0 = 0

    def costStep(self,x,u,k):
        return 0

    def step(self,x,u,k):
        return x

    def simulatePolicy(self,policy):
        Horizon = policy.Horizon
        if isinstance(self.x0,np.ndarray):
            n = len(self.x0)
            X = np.zeros((Horizon+1,n))
        else:
            X = np.zeros(Horizon+1)

        x = self.x0
        X[0] = x
        cost = 0.
        for k in range(Horizon):
            u = policy.action(x,k)
            cost = cost + self.costStep(x,u,k)
            x = self.step(x,u,k)
            X[k+1] = x

        return X,cost
