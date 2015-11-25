"""
These are some utilities to make sympy easier to use 
"""

import sympy as sym
import numpy as np
import dill
dill.settings['recurse'] = True

def cast_as_array(x):
    if isinstance(x,np.ndarray):
        return x
    else:
        return np.array(x)

def jacobian(F,x):
    """ 
    Computes: 
    J = dF / dx

    If x is a scalar, then J is just the derivative of each entry of F.

    If F is a scalar and x is a vector, then J is the gradient of F with respect to x. 

    If F and x are 1D vectors, then J is the standard Jacobian.

    More generally, the shape of J will be given by
    
    J.shape = (F.shape[0],...,F.shape[-1], x.shape[0],..., x.shape[-1])
    
    This assumes that F and x are scalars or arrays of symbolic variables.
    
    
    """

    Farr = cast_as_array(F)
    xarr = cast_as_array(x)

    Fflat = Farr.flatten()
    xflat = xarr.flatten()

    nF = len(Fflat)
    nx = len(xflat)
    # a matrix to hold the derivatives
    Mat = np.zeros((nF,nx),dtype=object)

    for i in range(nF):
        for j in range(nx):
            Mat[i,j] = sym.diff(Fflat[i],xflat[j])

    Jac = np.reshape(Mat, Farr.shape + xarr.shape)
    return Jac

def simplify(x):
    if isinstance(x,np.ndarray):
        xflat = x.flatten()
        nX = len(xflat)
        x_simp_list = [sym.simplify(expr) for expr in xflat]
        x_simp = np.reshape(x_simp_list,x.shape)
        return x_simp
    else:
        return sym.simplify(x)

def arg_to_flat_tuple(argTup):
    """
    Takes in a tuple of arguments. If any are np.ndarrays, they will 
    be flattended and added to the tuple
    """
    argList = []
    for variable in argTup:
        if isinstance(variable,np.ndarray):
            argList.extend(variable.flatten())
        else:
            argList.append(variable)
    return tuple(argList)
    

def functify(expr, args):
    """
    Usage: 
    name = functify(expr,args)
    This creates a function of the form
    expr = name(args)

    For more information, see
    https://www.youtube.com/watch?v=99JS6ym5FNE
    """
    if isinstance(args,tuple):
        argTup = args
    else:
        argTup = (args,)

    FuncData = {'nvar': len(argTup),
                'shape': (),
                'squeeze': False,
                'flatten': False}


    flatTup = arg_to_flat_tuple(argTup)
    if isinstance(expr,np.ndarray):
        FuncData['shape'] = expr.shape
        if len(expr.shape) < 2:
            FuncData['squeeze'] = True
            exprCast = sym.Matrix(expr)
        elif len(expr.shape) > 2:
            FuncData['flatten'] = True
            exprCast = sym.Matrix(expr.flatten())
        else:
            exprCast = sym.Matrix(expr)

        mods = [{'ImmutableMatrix': np.array}, 'numpy']
        func = sym.lambdify(flatTup,exprCast,modules=mods)
    else:
        func = sym.lambdify(flatTup,expr)

    def foo(*new_arg):
        new_flatTup = arg_to_flat_tuple(new_arg)
        result = func(*new_flatTup)
        if FuncData['squeeze']:
            return result.squeeze()
        elif FuncData['flatten']:
            return np.reshape(result,FuncData['shape'])
        else:
            return result

    return foo
    
def sympy_save(expr, args, name):
    """
    This first calls functify to create a function:
    name = functify(expr,args)

    so that the expression can be evaluated as 
    expr = name(args).

    It then saves the function (by pickling) to a file name+'.p'
    """

    func = functify(expr,args)
    

    # Pickle the function
    # this stores the calculation as a function in binary format
    fid = open(name+'.p','wb')
    dill.dump(func,fid)
    fid.close()

def sympy_load(name):
    """
    func = sympy_load(name)

    returns the function that was created by the call
    sympy_save(expr,args,name)
    """
    # load the pickled function
    fid = open(name+'.p','rb')
    func = dill.load(fid)
    fid.close()
    
    return func



