#!/usr/bin/env python

from distutils.core import setup

setup(name='pyopticon',
      version='0.1',
      description='Simulation package of optimal control problems',
      author=['Andy Lamperski','Bolei Di'],
      author_email=['alampers@umn.edu','dixxx047@umn.edu'],
      py_modules=['Controller','MarkovDecisionProcess'],
      install_requires=['numpy','sympy','dill','sympy_utils','pyuquat', \
                        'pylagrange','pynes','scipy','bovy_mcmc'],
      )
      
setup(name='pyuquat',
      version='0.1',
      description='Unit quaternion operations',
      author=['Andy Lamperski'],
      author_email=['alampers@umn.edu'],
      download_url = ['https://github.umn.edu/HRCLab/pyuquat.git'],
      py_modules=['pyuquat'],
      install_requires=['numpy','dill','scipy']
      )
     
     
    
    