#### Testing the nes class with mini wam
import MarkovDecisionProcess as MDP
import numpy as np
import Controller as ctrl
from wam_movie import wam_movie

mini_wam = MDP.load('mini_wam_model')

#### Initialize controllers
Horizon = 200
Controllers = []

# openloop control
U = np.zeros((Horizon,mini_wam.NumInputs))
openloopctrl = ctrl.openLoopPolicy(U,Horizon=Horizon)

Controllers.append(openloopctrl)

# sampling controller
sampling = ctrl.samplingOpenLoop(SYS=mini_wam,
                                    Horizon=Horizon,
                                    KLWeight=1e-4,
                                    burnIn=40,
                                    ExplorationCovariance=15.*\
                                    np.eye(mini_wam.NumInputs),
                                    label='Sampling')
Controllers.append(sampling)

# simulate different controllers
NumCon = len(Controllers)
X = np.zeros((NumCon,Horizon,mini_wam.NumStates))
U = np.zeros((NumCon,Horizon,mini_wam.NumInputs))
for l in range(NumCon):
    controller = Controllers[l]
    X[l],U[l],cost = mini_wam.simulatePolicy(controller)


ani = wam_movie(X[1].T,0.01,100)

