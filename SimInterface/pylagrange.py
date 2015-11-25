import sympy as sym
import numpy as np
from scipy.linalg import solve, inv
import sympy_utils as su

EULER = 0
SYMPLECTIC = 1

class lagrangian_system:
    def __init__(self,T=0,V=0,fric=0,x=0):
        self.build_system(T,V,fric,x)

    def build_system(self,T,V,fric,x):
        x = x
        n = len(x) / 2
        q = x[:n]
        dq = x[n:]
        # Mass matix
        dT = su.jacobian(T,dq)
        M = su.jacobian(dT,dq)
    
        # Christoffel symbols
        c = np.zeros((n,n,n),dtype=object)
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    c[i,j,k] = .5 * (sym.diff(M[k,j],q[i]) + \
                                     sym.diff(M[k,i],q[j]) - \
                                     sym.diff(M[i,j],q[k]))

        # Coriolis matrix
        C = np.zeros((n,n),dtype=object)
        for k in range(n):
            for j in range(n):
                for i in range(n):
                    C[k,j] = C[k,j] + c[i,j,k] * dq[i]

        # Potential energy matrix
        phi = su.jacobian(V,q)

        # store the symbolic expressions as functions
        self.M_fun = su.functify(M,x)
        self.C_fun = su.functify(C,x)
        self.phi_fun = su.functify(phi,x)
        self.fric_fun = su.functify(fric,x)

        # Now compute the Jacobians, which will be used in
        # linearizations

        # It will be more convienient to take jacobians w.r.t. q and dq
        # separately
        M_jac_q = su.jacobian(M,q)
        # For more uniform calling the function depends on x
        # with the dq terms ignored.
        self.M_jac_q = su.functify(M_jac_q,x)
        # Ignoring inputs since these go to zero with jacobian
        RHS = -(np.dot(C,dq)+phi+fric)
        RHS_jac_q = su.jacobian(RHS,q)
        RHS_jac_dq = su.jacobian(RHS,dq)
        # The functions depend on x=(q,dq)
        self.RHS_jac_q = su.functify(RHS_jac_q,x)
        self.RHS_jac_dq = su.functify(RHS_jac_q,x)
        
    def qddotFun(self,x,u=0):
        n = len(x)/2
        qdot = x[n:]
    
        M = self.M_fun(x)
        C = self.C_fun(x)

        phi = self.phi_fun(x)
        fric = self.fric_fun(x)
    
        RHS = -np.dot(C,qdot) - phi - fric + u
    
        qddot = solve(M,RHS,sym_pos=True) 
        return qddot
    
    def vectorField(self,x,u=0):
        n = len(x)/2
        qdot = x[n:]
        qddot = self.qddotFun(x,u)
        return np.hstack((qdot,qddot))

    def stepEuler(self,x,u=0,dt=0.01):
        xdot = self.vectorField(x,u)
        return x+dt*xdot

    def stepSymplectic(self,x,u=0,dt=0.01):
        n = len(x)/2
        qddot = self.qddotFun(x,u)
        qdotNew = x[n:] + dt * qddot
        qNew = x[:n] + dt * qdotNew
        return np.hstack((qNew,qdotNew))

    def step(self,x,u=0,dt=0.01,integrator=SYMPLECTIC):
        if integrator == EULER:
            return self.stepEuler(x,u,dt)
        elif integrator == SYMPLECTIC:
            return self.stepSymplectic(x,u,dt)
        else:
            print 'Undefined Integrator'

    def linearization(self,xNom,uNom):
        """
        Compute 
        A,B,g = self.linearization(xNom,uNom)

        so that for (x,u) near (xNom,uNom) we have the approximate equality
        xdot = A (x-xNom) + B (u-Nom) + g
        """
        n = len(xNom)/2

        # B is computed from M^{-1}
        M = self.M_fun(xNom)
        M_inv = inv(M)
        B = np.vstack((np.zeros((n,n)),M_inv))


        # A is more complex
        xdot = self.vectorField(xNom,uNom)
        qddot = xdot[n:]
        Atop = np.hstack((np.zeros((n,n)),np.eye(n)))
        M_jac_q = self.M_jac_q(xNom)
        RHS_jac_q = self.RHS_jac_q(xNom)
        RHS_jac_dq = self.RHS_jac_dq(xNom)
        Abottom_left = np.dot(M_inv,RHS_jac_q-np.einsum('ijk,j',M_jac_q,qddot))
        Abottom_right = np.dot(M_inv,RHS_jac_dq)
        Abottom = np.hstack((Abottom_left,Abottom_right))
        A = np.vstack((Atop,Abottom))

        # g is just the nominal derivative
        g = xdot
        return (A,B,g)
    
