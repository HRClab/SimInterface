import numpy as np

def castToShape(M,Shape):
    if (not isinstance(M,np.ndarray)) and (np.prod(Shape)>1):
        MCast = np.zeros(Shape)
    else:
        MCast = np.reshape(M,Shape)
    return MCast

def stack(tupleToStack):
    """
    Horizontally stack 
    M = stack((A,B,C))

    Create a 2D array from
    M = stack(((A,B),
               (C,D)))
    """

    if isinstatance(tupleToStack[0],tuple):
        # Horizontal Stacking
        return np.hstack(tupleToStack)
        
    else:
        # Vertically stack horizontally stacked rows
        rows = []
        for row in tupleToStack:
            rows.append(stack(row))

        return np.vstack(tuple(rows))
