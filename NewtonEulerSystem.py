from scipy.linalg import solve
import numpy as np
import sympy as sym
import uquat as uq
import sympy_utils as su
import dill
dill.settings['recurse'] = True

# This script class that of Newton-Euler systems
# The dynamics of such systems are computed through Newton-Euler equations of 
# rigid bodies and constraints

#### helper functions
def states_generator(NumBody):
    # generate symbols of states
    xstride = 13
    pstride = 7
    vstride = 6    
    # takes in number of rigid bodies and generates symbols of states
    Pos = sym.symarray('Pos',(NumBody,3))
    Quat = sym.symarray('Quat',(NumBody,4))
    Vel = sym.symarray('Vel',(NumBody,3))
    Omega = sym.symarray('Omega',(NumBody,3))
    
    X = np.zeros(NumBody*xstride,dtype=object) # packed all states 
    P = np.zeros(NumBody*pstride,dtype=object) # packed position and quaternions
    V = np.zeros(NumBody*vstride,dtype=object) # packed velocity and rotational velocity
    
    return X,P,V,Pos,Quat,Vel,Omega

def build_M_matrix(m,I):
    # convert array of mass and rotational inertia to M matrix
    # m: NumBody by 1
    # I: 3*NumBody by 1
    NumBody = len(m)
    M = np.ones(6*NumBody)
    for l in range(NumBody):
        M[6*l:6*l+3] = m[l]*np.ones(3)
        M[6*l+3:6*(l+1)] = I[3*l,3*(l+1)]
    M = np.diag(M)
    Minv = np.invert(M)
    return M,Minv

def inertia_extractor(M):
    Numbody = M.shape[0]/6
    m = np.zeros(Numbody)
    I = np.zeros((Numbody,3,3))
    for l in range(Numbody):
        m[l] = M[6*l][6*l]
        I[l] = M[6*l+3:6*l+6][6*l+3:6*l+6]
    Minv = np.invert(M)
    return m,I,Minv


def states_extractor(x):
    NumBody = int( x.shape[0] / 13 )
    xstride = 13
    pstride = 7
    vstride = 6  
    #### dismantle x into P, V and Pos, Quat, Vel, Omega such that
    #### P[l], V[l], Pos[l], Quat[l], Vel[l] and Omega[l] are corresponding
    #### states of the lth link
    P = np.zeros((NumBody,pstride))
    V = np.zeros((NumBody,vstride))
    Pos = np.zeros((NumBody,3))
    Quat = np.zeros((NumBody,4))
    Vel = np.zeros((NumBody,3))
    Omega = np.zeros((NumBody,3))
    for l in range(NumBody):
        P[l] = x[l*xstride:l*xstride+pstride]
        V[l] = x[l*xstride+vstride:(l+1)*xstride]
        Pos[l] = x[l*xstride:l*xstride+3]
        Quat[l] = x[l*xstride+3:l*xstride+7]
        Vel[l] = x[l*xstride+pstride:l*xstride+pstride+3]
        Omega[l] = x[l*xstride+pstride+3:(l+1)*xstride]
    
    return P,V,Pos,Quat,Vel,Omega

#### class definition
class NewtonEulerSystems():
    def __init__(self,x,u,Gu,Constraint,M):
        # x should be 13*self.NumBody by 1 symbol array
        # self.NumBody is the number of rigid bodies
        # u is the input symbol array
        # Gu is generalized input that directly apply on the rigid bodies
        # Gu should be a symbolic expression of u and is 6*self.NumBody by 1
        # Constraint describes the rigid body connection
        # Currently Constraint can only handle P related constraints
        # M is a diagonal mass matrix
        self.M = M
        self.m,self.I,self.Minv = inertia_extractor(M)
        self.xstride = 13
        self.pstride = 7
        self.vstride = 6
        self.build_system_functions(x,u,Gu,Constraint)
    
    def build_system_functions(self,x,u,Gu,Constraint):
        # This function converts symbolic expressions to callable numerical
        # functions which are then used to build system dynamic functions and
        # approximate or correction systems
        
        #### get number of rigid bodies
        self.NumBody = int( x.shape[0] / 13 )
        self.Gu_fun = su.functify(Gu,u)
        
        #### Dismantle states
        P,V,Pos,Quat,Vel,Omega = states_extractor(x)
        
        #### Computing Pdot and its Jacobian w.r.t. V, needed for computing constraint force
        Pdot = np.zeros(self.NumBody*self.pstride,dtype=object)
        for l in range(self.NumBody):    
            PosDot = uq.rot(Quat[l],Vel[l])
            QuatDot = uq.mult(Quat[l],Omega[l]/2.)
            Pdot[l*self.pstride:(l+1)*self.pstride] = np.hstack((PosDot,QuatDot))
            
        Pdot_V_Jac = su.jacobian(Pdot,V)
        
        #### Compute constraint related matrices and functify them
        Constraint_x_Jac = su.jacobian(Constraint,x)

        # Compute constraint matrices
        ConstraintMat = np.dot(su.jacobian(Constraint,P),Pdot_V_Jac)
        ConstraintTen = su.jacobian(ConstraintMat,P)
        ConstraintMatDot = np.dot(ConstraintTen,Pdot)
        
        # Convert symbolic expressions to callable numerical functions
        self.Constraint_fun = su.functify(Constraint,x)  # vector of all equality constraints
        self.Constraint_x_Jac_fun = su.functify(Constraint_x_Jac,x) # Jacobian of constraints w.r.t. x
        self.ConstraintMat_fun = su.functify(ConstraintMat,x) # the A(Q) matrix
        self.ConstraintMatDot_fun = su.functify(ConstraintMatDot,x) # derivative of A(Q) matrix
        
        # Initialize the Lagrane multiplier for constraint forces
        # It is only at this point that the dimension of the contraints is cleared
        lam = sym.symarray('lam',len(Constraint)) # lambda is reserved in python
        
        #### Generate linearized model functions
        #### Assume the model takes this form:
        #### Pdot = Pdot_V_Jac*V
        #### Vdot = Minv*(F-C(V))
        #### where F is the generalized force:
        #### F = Gu(u) + ConstrainMat(P).T*lam + Gravity(P), all in body frame
        #### Assume the linearized model takes this form:
        #### f(x)_approximate = linearized_model_A * X_perturbed + \ 
        ####                    linearized_model_B * U_perturbed + \
        ####                    linearized_model_G * lam_perturbed + \
        ####                    linearized_model_g
        
        ## The C(V) matrix
        CV = np.zeros(self.NumBody*self.pstride,dtype=object)
        for l in range(self.NumBody):
            CV[l*self.vstride:l*self.vstride+3] = self.m[l]*np.cross(Omega[l],Vel[l])
            CV[l*self.vstride+3:l*self.vstride+6] = np.cross(Omega[l],np.dot(self.I[l],Omega[l]))
        self.CV_fun = su.functify(CV,x)
        
        ## Gravitational force
        g = 10.0
        Fgrav = np.zeros((self.NumBody*self.vstride),dtype=object)
        for l in range(self.NumBody):
            Fgrav[l*self.vstride:l*self.vstride+3] = uq.rot(uq.inv(Quat[l]),self.m[l]*g*np.array([0,0,-1]))
        self.Fgrav_fun = su.functify(Fgrav,P)        
        
        ## Compute Vdot, take jacobians
        Vdot = np.dot(self.Minv,(Gu + Fgrav + np.dot(ConstraintMat.T,lam) - CV))
        Xdot = np.zeros(self.NumBody*self.xstride,dtype = object)
        for l in range(self.NumBody):
            Xdot[l*self.xstride:l*self.xstride + self.pstride] = \
                                Pdot[l*self.pstride:(l+1)*self.pstride]
            Xdot[l*self.xstride+self.pstride:(l+1)*self.xstride] = \
                                Vdot[l*self.vstride:(l+1)*self.vstride]
            
        linearized_model_A = su.jacobian(Xdot,x)
        linearized_model_B = su.jacobian(Xdot,u)
        linearized_model_G = su.jacobian(Xdot,lam)
        
        self.Xdot_fun = su.functify(Xdot,np.hstack((x,u,lam)))
        self.linearized_model_A_fun = su.functify(linearized_model_A,np.hstack((x,u,lam)))
        self.linearized_model_B_fun = su.functify(linearized_model_B,np.hstack((x,u,lam)))
        self.linearized_model_G_fun = su.functify(linearized_model_G,np.hstack((x,u,lam)))
        
    #### Computes derivative of generalized velocity for a single rigid body
    def V_dot_NE(self,V,W,m,I): # P is generalized velocity
        """
        This takes in a body velocity and a wrench and outputs the derivatives
        of the body velocity
        """
        Vel = V[:3]
        Omega = V[3:]
        F = W[:3]
        Tau = W[3:]
    
        Vel_dot = -np.cross(Omega,Vel) + F/m
        Iinv = np.diag(1./np.diag(I))
        Omega_dot = np.dot(Iinv,-np.cross(Omega,np.dot(I,Omega))+Tau)
    
        return np.hstack((Vel_dot,Omega_dot))
    
    #### Compute increment of generalized position of a single rigid body
    #### given dt
    def twist_diff(self,X,dt):
        """
        Rather than computing the derivative, dPos/dt, we compute the differential
        dPos. This is because we use a modified Euler rule to maintain a normalized
        quaternion
        """
        Quat = X[3:7]
        Vel = X[7:10]
        Omega = X[10:13]
    
        dPos = dt * uq.rot(Quat,Vel)
        dQuat = uq.mult(Quat,uq.expq(Omega * dt/2.)) - Quat
    
        return np.hstack((dPos,dQuat))    
 
    def state_diff(self,x,u,dt):
        # Everything here are numerical        
        
        #### Dismantle states
        P,V,Pos,Quat,Vel,Omega = states_extractor(x)
        V_vec = np.reshape(P,self.NumBody*self.vstride)
    
        ## Compute the total wrench on the links, not counting constraint wrenches
        ## This wrench will be used to compute free response without constraint
        Fgrav = self.Fgrav_fun(P)
        Gu = self.Gu_fun(u)
        Wren_vec = Gu + Fgrav
        
        ## apply the wrench on all links to compute free response
        Vdot_free_vec = np.zeros(self.vstride * self.NumBody)
        for l in range(self.NumBody):
            # Second derivatives without constraints
            Wl = Wren_vec[l*self.vstride:(l+1)*self.vstride] # pick out the wrench on lth link
            Vdot_free_vec[l*self.vstride:(l+1)*self.vstride] = self.V_dot_NE(P[l],Wl,self.m[l],self.I[l])
        
        ## Constraint force
        ConMat = self.ConMat_fun(x)
        ConMatDot = self.ConMatDot_fun(x)
        lamMat = np.dot(ConMat,np.dot(self.Minv,ConMat.T))
        lam = -solve(lamMat,np.dot(ConMatDot,V_vec)+np.dot(ConMat,Vdot_free_vec))
        # lamBreak is an extra force that pulls the joints back together
        # if the constraint force is not sufficient. This will happen if
        # the speed is high, compared to the time-step
        KcP = 1.0
        KcD = 1.0
        lamBreak = -KcP * self.Con_fun(x) - KcD * np.dot(ConMat,V_vec)
        ConForce = np.dot(ConMat.T,lam + lamBreak)
            
        ## Stuff velocity terms into dx
        dx = np.zeros(self.xstride * self.NumBody)
        dV = dt * (Vdot_free_vec + np.dot(self.Minv,ConForce)) # generalized acceleration
        for l in range(self.NumBody):
            Vxind = range(7 + l*self.xstride, 13 + l*self.xstride)
            Vpind = range(l*self.vstride, (l+1)*self.vstride)
            dx[Vxind] = dV[Vpind]
    
            # Position derivatives
            Pxind = range(l*self.xstride,7+l*self.xstride)
            CurP = x[Pxind]
            NewV = x[Vxind] + dV[Vpind]
            SymplecticX = np.hstack((CurP,NewV))
            dx[Pxind] = self.twist_diff(SymplecticX,dt)
    
        return dx
        
        def linearization(self,x,u,lam):
            A = self.linearized_model_A_fun(np.hstack((x,u,lam)))
            B = self.linearized_model_B_fun(np.hstack((x,u,lam)))
            G = self.linearized_model_G_fun(np.hstack((x,u,lam)))
            
            return A,B,G
            
        def equil_finder(self):
            pass
        
        def con_force_finder(self):
            pass