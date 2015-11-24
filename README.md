# PyOptiCon

Code for rapidly developing simulations and control schemes

# Dependencies

* Scipy stack
* bovy_mcmc - for sampling - 
https://github.com/jobovy/bovy_mcmc
* pylagrange https://github.umn.edu/HRCLab/pylagrange
* pyNewEuler https://github.umn.edu/HRCLab/pyNewEuler
* sympy_utils https://github.umn.edu/HRCLab/sympy_utils
* pyuquat https://github.umn.edu/HRCLab/pyuquat

# TODO

- [ ] Make classes with external dependencies load conditionally. 
- [ ] Enable automated installation in setup.py
- [ ] UPDATE THE DOCUMENTATION
- [ ] Generalize class rbfNetwork to allow for vector functions (see createRBFbasis)
- [ ] Put in an x gradient in the general functionApproximator class
- [ ] Update existing function approximator classes to have their x gradients
- [ ] A notebook on how to use the function approximators, and detail their capabilities.
- [ ] Merge in varying function
- [ ] Make parameterized function a subclass of varying function
- [ ] Put parameterized function into a different file
  