#!/usr/bin/env python
from setuptools import setup

setup(name='SimInterface',
      version='0.1',
      description='Simulation package of optimal control problems',
      author=['Andy Lamperski','Bolei Di'],
      author_email=['alampers@umn.edu','dixxx047@umn.edu'],
      packages = ['SimInterface'],
      install_requires=['numpy','sympy','dill','scipy_utils','uquat', \
                        'pylagrange','scipy','bovy_mcmc'],
      )
      
     
     
    
    
