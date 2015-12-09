.. SimInterface documentation master file, created by
   sphinx-quickstart on Tue Nov 24 18:31:56 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SimInterface Documentation
========================================

Oh hey! This is the documentation for the SimInterface, a Python package for simulation and control. The package, and its documentation, are very much "Work in Progress", so this will likely change drastically on a regular basis.

What Does the  SimInterface Do?
-------------------------------

The SimInterface is meant to be a modular environment for constructing and maniupulating dynamical systems. In particular, the main task of the SimInterface is to handle bookkeeping tasks such as the value of internal signals, which variables get passed to which functions, and so on. 

Features
~~~~~~~~

* Construction of interconnected system from subsystems

* Generation of vector fields that can be passed to simulators

* Organization of signals in Pandas DataFrames 

What doesn't the SimInterface Do?
---------------------------------

The SimInterface is not intended to be a full simulation environment. This code is to help with constructing and manipulating the systems. To simulate a system, you can use separate, dedicated tools, such as scipy.itegrate. Thus, you are free to simulate the generated systems however you wish.

Additionally, by keeping the focus of base modules narrow, the code stays small and with minimal dependencies. 

Modules
=======
  
.. toctree::
   :maxdepth: 1

   Variable
   System


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

