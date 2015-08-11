import numpy as np
import UnderactuatedPendulum as UAP

dt = 0.05
sys = UAP.UnderactuatedPendulum(dt = dt)

A,B,g = sys.discreteTimeLinearization(sys.x0,0)
