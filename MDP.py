import numpy as np

class MDP:
    def __init__(self):
        self.x0 = 0

    def costStep(self,x,u,k):
        return 0

    def step(self,x,u,k):
        return x

    def simulatePolicy(self,policy):
        NStep = policy.NStep
        if isinstance(self.x0,np.ndarray):
            n = len(self.x0)
            X = np.zeros((NStep+1,n))
        else:
            X = np.zeros(NStep+1)

        x = self.x0
        X[0] = x
        cost = 0.
        for k in range(NStep):
            u = policy.action(x,k)
            cost = cost + self.costStep(x,u,k)
            x = self.step(x,u,k)
            X[k+1] = x

        return X,cost
