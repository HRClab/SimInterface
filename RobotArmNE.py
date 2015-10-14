#### Testing the nes class with mini wam

import NewtonEuler as ne
import MarkovDecisionProcess as MDP
import utils.sympy_utils as su
import pyuquat as uq
import sympy as sym
import numpy as np

class RobotArmNE(MDP.NewtonEulerSys):
    # This class creates robot arms similar to the Barret Wam arm
    # objects differ from each other by their number of links, mass of links,
    # rotational interia of links and length of links

    # Different cost functions are offerred and 
    
    # To build a MDP.NewtonEulerSys, you need to specify
    # Number of rigid bodies
    # m, I arrays
    # Form of input and how that transform to generalized force
    # friction or any states dependent wrench
    # Constraints    
    def __init__(self,m,I,h,theta0,V0,dt,cost_type=None,cost_par=None):
        # theta0 is the initial joint angles
        # V0 is the initial body frame velocities
        self.h = h
        self.Nlink = len(m)
        self.xstride = 13
        self.pstride = 7
        self.vstride = 6
        # Link positions
        self.zJ = np.zeros((self.Nlink,3))
        for k in range(self.Nlink):
            self.zJ[k] = np.array([0,0,h[k]/2.])
        self.build_theta_x_funcs()     
        
        #### Generate input, constraint, cost and initial states
        #### These are the difference between an arm and a general ne
        X,P,V,Pos,Quat,Vel,Omega = ne.states_generator(self.Nlink)
        U = sym.symarray('U',self.Nlink)
        # transformation matrix from U to generalized force
        # this should be a constant matrix regardless of P and V
        # GU = TG*U
        TG = np.zeros((self.Nlink*self.vstride,self.Nlink),dtype=object)
        # indices of axis that each joint torque applies on corresponding link
        TauUind = np.zeros(self.Nlink,dtype=int)
        for l in range(self.Nlink):
            TauUind[l] = 1 + ((l+1) % 2) # 1 if odd, 2 if even
        for l in range(self.Nlink):
            if l == (self.Nlink-1):
                TG[self.vstride*l+3+TauUind[l]][l] = 1
            else:
                TG[self.vstride*l+3+TauUind[l]][l] = 1
                TG[self.vstride*l+3+TauUind[l+1]][l+1] = -1
        GU = np.dot(TG,U)
        
        ## Friction
        # friction on each joint as a Nlink*1 array
        fricCoef = 0.1
        fric_Nlink = np.zeros((self.Nlink),dtype=object)
        # friction as generalized force
        Ffric = np.zeros((self.Nlink*self.vstride),dtype=object)
        
        for l in range(self.Nlink):
            if l == 0:
                fric_Nlink[l] = fricCoef*Omega[l][TauUind[l]]
            else:
                # Relative quaternion that transform current link's body frame to 
                # that of the previous link
                QuatRel = uq.mult(uq.inv(Quat[l-1]),Quat[l])
                # Relative omega w.r.t. the previous link
                OmegaRel = Omega[l] - uq.rot(uq.inv(QuatRel),Omega[l-1])
                # Friction
                fric_Nlink[l] = fricCoef*OmegaRel[TauUind[l]]
        
        Ffric = - np.dot(TG,fric_Nlink)
        
        ## Generate constraints functions
               
        # Generate position constraints
        ConstraintPos = np.zeros(3*self.Nlink,dtype=object)
        
        for l in range(self.Nlink):
            if l == 0:
                CP = Pos[0] - self.zJ[l]
            else:
                CP = (Pos[l]-uq.rot(Quat[l],self.zJ[l])) - \
                     (Pos[l-1]+uq.rot(Quat[l-1],self.zJ[l-1]))
                
            ConstraintPos[3*l:3+3*l] = CP
        
        # Generate rotation constraints
        ConstraintRot = np.zeros(2*self.Nlink,dtype=object)
        for l in range(self.Nlink):
            if l==0:
                CR = Quat[l,1:3]
            else:
                RelQuat = uq.mult(uq.inv(Quat[l-1]),Quat[l])
                if (l % 2) == 1:
                    # Odd links rotate about relative y-axis
                    CR = RelQuat[[1,3]]
                elif (l % 2) == 0:
                    # Even links rotate about relative z-axis
                    CR = RelQuat[[1,2]]        
        
            ConstraintRot[2*l:2*(l+1)] = CR
        
        Constraint = np.hstack((ConstraintPos,ConstraintRot))
        
        ## Generate cost function
        # Tip function
        tip = np.zeros((3),dtype=object)
        for l in range(self.Nlink):
            tip += uq.rot(Quat[l],2*self.zJ[l])
        self.tip_fun = su.functify(tip,P)
        
        if cost_type == 'TipTarget':
            tip_target = cost_par            
            tip_error = tip - tip_target
            target_cost = np.dot(tip_error,tip_error)
            energy_cost = 0.0*np.dot(U,U)
            cost = target_cost + energy_cost
        elif cost_type == 'JointTarget':
            cost = 1
            pass
        else:
            print 'I don recognize this kind of cost function'
            print 'Setting cost to constant'
            cost = 1
        
        ## Generate initial states
        P0 = self.theta2P(theta0)
        x0 = self.PV2X(P0,V0)
        
        MDP.NewtonEulerSys.__init__(self,m,I,dt,X,U,GU,cost,x0,Ffric,Constraint)
        
        
    def build_theta_x_funcs(self):
        # Build some useful arm functions
        # Mainly related to conversion between joint angles and states        
        theta = sym.symarray('theta',Nlink)
        theta_dot = sym.symarray('theta_dot',Nlink)
        
        # re-initialize the quaternions
        Quat = np.zeros((Nlink,4),dtype=object)
        Pos = np.zeros((Nlink,3),dtype=object)
        Omega = np.zeros((Nlink,3),dtype=object)
        Vel = np.zeros((Nlink,3),dtype=object)
        H_link = np.zeros((Nlink),dtype=object) # heights of all links
        
        for l in range(Nlink):
            QuatRel = np.zeros(4,dtype=object)
            QuatRel[0] = sym.cos(theta[l]/2.)
            QOther = sym.sin(theta[l]/2.)
            OmegaRel = np.zeros(3,dtype=object)
            if (l%2) == 0:
                QuatRel[3] = QOther
                OmegaRel[2] = theta_dot[l]
            else:
                QuatRel[2] = QOther
                OmegaRel[1] = theta_dot[l]
        
            if l==0:
                Pos[l] = self.zJ[l]
                Quat[l] = QuatRel
                Omega[l] = OmegaRel
            else:
                Rrel = su.simplify(uq.mat(QuatRel))
        
                # Forward kinematics
                Quat[l] = uq.mult(Quat[l-1],QuatRel)
                Pos[l] = Pos[l-1] + uq.rot(Quat[l-1],self.zJ[l-1]) + \
                         uq.rot(Quat[l],self.zJ[l])
        
                # Jacobian
                Omega[l] = OmegaRel + np.dot(Rrel.T,Omega[l-1])
                Vel[l] = np.cross(Omega[l],self.zJ[l]) \
                         + np.dot(Rrel.T,Vel[l-1]) \
                         + np.dot(Rrel.T,np.cross(Omega[l-1],self.zJ[l-1]))
            
            # extract heights of the links
            H_link[l] = Pos[l][2]
            
            H_link_theta_Jac = su.jacobian(H_link,theta)
            self.H_link_theta_Jac_fun = su.functify(H_link_theta_Jac,theta)
                
            P = np.hstack((Pos,Quat))
            P_vec = np.reshape(P,self.pstride*Nlink)
            self.kinematics_fun = su.functify(P_vec,theta)
                                
    def ArmTip(self,P):
        return self.tip_fun(P)
    
    def theta2P(self,theta):
        # Convert joint angles to states
        P = np.zeros((self.Nlink,self.pstride))
        Pos = np.zeros((self.Nlink,3))
        Quat = np.zeros((self.Nlink,4))
        for l in range(self.Nlink):
            if l == 0:
                Pos[l] = np.array([0,0,self.h[l]/2])
                Quat[l] = uq.expq(theta[l]/2*np.array([0,0,1.]))
            else:
                RotAxis = np.zeros(3)
                RotAxis[2 - l%2] = 1.0
                QuatRel = uq.expq(theta[l]/2*RotAxis)
                Quat[l] = uq.mult(Quat[l-1],QuatRel)
                Pos[l] = Pos[l-1] + uq.rot(Quat[l-1],self.zJ[l-1]) + uq.rot(Quat[l],self.zJ[l])
                
        for l in range(self.Nlink):
            P[l] = np.hstack((Pos[l],Quat[l]))
        return P
    
    def P2theta(self,P):
        pass
    
    def PV2X(self,P,V):
        # Stuff P and V to X
        # X is a vector but P and V are arrays
        X = np.zeros(self.xstride*self.Nlink)
        for l in range(self.Nlink):
            X[l*self.xstride:l*self.xstride+self.pstride] = P[l]
            X[l*self.xstride+self.pstride:(l+1)*self.xstride] = V[l]
        return X
    
    def joint_angles(self,x):
        theta = np.zeros(Nlink)
        for l in range(Nlink):
            Quat = x[3 + l*self.xstride: 7 + l*self.xstride]
    
            if l == 0:
                QuatRel = Quat
            else:
                QuatLast = x[3+(l-1)*self.xstride:7+(l-1)*self.xstride]
                QuatRel = uq.mult(uq.inv(QuatLast),Quat)
    
            cTheta = QuatRel[0]
            if (l % 2) == 1:
                sTheta = QuatRel[2]
            elif (l % 2) == 0:
                sTheta = QuatRel[3]
            theta[l] = 2*np.arctan2(sTheta,cTheta)
        return theta
    
    def PQVO2PQ(self,Pos,Quad,Vel,Omega):
        pass
    
    def PQVO2X(self,Pos,Quad,Vel,Omega):
        pass
    
    def TipJointJac(self,P):
        pass
    
    def equil_input(self,x):
        ## extract P_vec from x
        P,V,Pos,Quat,Vel,Omega = ne.states_extractor(x)    
        
        ## input
        Fgrav_Nlink = self.Fgrav_NumBody.fun(P)
        u_eq = np.dot(self.H_link_theta_Jac_fun(self.joint_angles(x)).T,Fgrav_Nlink)
        return u_eq
    
    def moive(self,X):
        pass

# Generate and same a 3-link small robot arm
Nlink = 3
m = np.ones(Nlink)
I = 0.1*np.ones(3*Nlink)
h = np.array([0.7,0.5,1.8,1.8,.4,.6,.4])
tip_target = np.array([0,2.3,0])

mini_wam = RobotArmNE(m,I,h,
                      theta0=np.zeros(Nlink),
                      V0=np.zeros((Nlink,6)),
                      dt=0.01,
                      cost_type='TipTarget',
                      cost_par=tip_target)

MDP.save(mini_wam,'mini_wam_model')

# Generate a 7-DoF Wam
