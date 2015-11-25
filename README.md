# Documentation

Link to documentation as soon as it renders properly!

# What is SimInterface?

It is a python-based simulation environment. Currently, it is primarily geared to testing optimal control algorithms. The scope will increase with time.


Its main aims are:

* [Object Oriented Design](#object)
* [Portable Controllers](#portable)

# <a name="object"></a> Object Oriented Design

## Encapsulate System and Controller Data
Plants and controllers are objects. Plant objects should encapsulate
all information required for simulation and control
design. Controller design methods can take plant objects as inputs.  

## Inheritance
Designing new plant types and controller methods based on existing
schemes simplifies. 

# <a name="portable"></a> Portable Controllers
Controllers are be designed to work for every plant in a class. The
syntax for controller becomes quite simple.