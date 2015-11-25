#!/usr/bin/env python
from setuptools import setup

# This is a bit spurious, as it does not explicilty have a scipy dependence
setup(name='SimInterface',
      version='0.1',
      description='Simulation package of optimal control problems',
      author=['Andy Lamperski','Bolei Di'],
      author_email=['alampers@umn.edu','dixxx047@umn.edu'],
      packages = ['SimInterface'],
      install_requires=['numpy','sympy','scipy']
)
      
     
     
    
    
