import numpy as np
from numpy.random import randn
import matplotlib.pyplot as plt
from bovy_mcmc import elliptical_slice as eslice
import sympy_utils as su

def cost(x):
    return  -2*x**2. + 1*x**4.

def likelihood(x):
    lam = 1.
    return np.exp(-cost(x)/lam)


X = np.linspace(-2,2,100)
Y = cost(X)

plt.figure(1)
plt.clf()
plt.plot(X,Y)
