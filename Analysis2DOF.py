''' 
Nicolò Pollini, Jan. 2024
Haifa, Israel
'''

# Import packages and modules
import numpy as np
import math
import scipy.io as sio

##  Set up classes:

# Optimization class
class DAMP_OPT:
    
    def __init__(self):
        self.pnorm = 100
        self.dallow = 0.009
        self.q = 100
        
    def cal_sensitivity(self,Str):
        # Parameters
        gamma = Str.gamma; beta = Str.beta; Dt = Str.Dt
        M = Str.M
        C = Str.C
        K = Str.K
        H = Str.H
        Ndof = Str.Ndof_u
        Nt = len(Str.time)
        tf = Str.time[-1]
        dallow = self.dallow
        
        # Divide the drifts by the allowable value dallow
        drift = Str.drift
        d_bar = (1.0/self.dallow) * drift
        # Max drift value, one for each drift DOF
        d_barMax = np.amax(np.absolute(d_bar),axis=1)
        # Drifts normalized by the respective max vlaue, the max value now is 1
        d_cap = np.dot(np.diag(1/d_barMax),d_bar)
        # Weigths for integration, p-norm max approximation
        weigth = Dt*np.ones((1,Nt))
        weigth[0,0] = Dt/2
        weigth[0,-1] = weigth[0,0]
        # Time length
        dcr = (np.sum((1/tf)*weigth*d_cap**self.pnorm, axis=1))**(1/self.pnorm)
        dc_vec = np.dot(np.diag(d_barMax),dcr)
        dc_vec_max = np.amax(dc_vec)
        dc_vec = dc_vec/dc_vec_max
        one = np.ones((Ndof,1))
        nom = np.dot(one.reshape(1,2),np.dot(np.diag(dc_vec**(self.q+1)),one))
        den = np.dot(one.reshape(1,2),np.dot(np.diag(dc_vec**(self.q)),one))

        # Coefficient matrix
        MatA = np.zeros((3*Ndof,3*Ndof))
        MatA[0:Ndof,0:Ndof] = M.T
        MatA[0:Ndof,2*Ndof:3*Ndof] = -np.eye(Ndof)
        MatA[Ndof:2*Ndof,0:Ndof] = C.T
        MatA[Ndof:2*Ndof,Ndof:2*Ndof] = -np.eye(Ndof)
        MatA[2*Ndof:3*Ndof,0:Ndof] = K.T
        MatA[2*Ndof:3*Ndof,Ndof:2*Ndof] = +(gamma/(beta*Dt))*np.eye(Ndof)
        MatA[2*Ndof:3*Ndof,2*Ndof:3*Ndof] = +(1/(beta*Dt**2.0))*np.eye(Ndof)
        
        l = np.zeros((3*Ndof,Nt))
        b = np.zeros((3*Ndof,1))
        MatAInv = np.linalg.inv(MatA)
        
        # Initial (terminal) conditions
        DdcDcd1 = -(H.T).dot(np.diag(d_barMax)).dot(np.diag((np.sum((1/tf)*weigth*d_cap**self.pnorm, axis=1))**((1-self.pnorm)/self.pnorm)))
        DdcDcd2 = (1/tf) *weigth[0,-1]*np.diag(d_cap[:,-1]**(self.pnorm-1)).dot(np.diag(1/d_barMax)).dot((1/dallow)*np.eye(Ndof))
        DdcDcd3 = (1/den**2)*(den*(self.q+1)*np.dot(np.diag(dc_vec**self.q),one)-\
                   nom*self.q*np.dot(np.diag(dc_vec**(self.q-1)),one))
        DdcDcd = DdcDcd1.dot(DdcDcd2).dot(DdcDcd3)
        b[2*Ndof:3*Ndof,0] = DdcDcd[:,0] 
        l[:,-1] = np.dot(MatAInv,b)[:,0]
        
        for i in range(Nt-2,0,-1):
            x2 = l[Ndof:2*Ndof,i+1]
            x3 = l[2*Ndof:3*Ndof,i+1]
            b[0:Ndof,0] = -Dt*(1-gamma/(2*beta))*x2 + (0.5/beta-1)*x3
            b[Ndof:2*Ndof,0] = -(1-gamma/beta)*x2 + (1/(beta*Dt))*x3
            b[2*Ndof:3*Ndof,0] = +(gamma/(beta*Dt))*x2 + (1/(beta*Dt**2.0))*x3
            #DdcDcd1 = -H.T.dot(np.diag(max_drift)).dot(np.diag((np.sum((1/tf)*weigth*drift_normal**self.pnorm, axis=1))**((1-self.pnorm)/self.pnorm)))
            DdcDcd2 = (1/tf) *weigth[0,i]*np.diag(d_cap[:,i]**(self.pnorm-1)).dot(np.diag(1/d_barMax)).dot((1/dallow)*np.eye(Ndof))
            #DdcDcd3 = (1/den**2)*(den*(self.q+1)*np.dot(np.diag(dc**self.q),one)-\
            #       nom*self.q*np.dot(np.diag(dc**(self.q-1)),one))
            DdcDcd = DdcDcd1.dot(DdcDcd2).dot(DdcDcd3)
            b[2*Ndof:3*Ndof,0] += DdcDcd[:,0]
            li = np.dot(MatAInv,b)
            l[:,i] = li[:,0]
        
        DCdDcd = np.zeros((Ndof,Ndof,Str.Ndamper))
        DCdDcd[:,:,0] = Str.H.T.dot(np.array([[1,0],[0,0]])).dot(Str.H)
        DCdDcd[:,:,1] = Str.H.T.dot(np.array([[0,0],[0,1]])).dot(Str.H)
        grad = np.zeros((Str.Ndamper,1))
        for j in range(0,Str.Ndamper):
            for i in range(0,Nt):
                grad[j,0] += Str.vel[:,i].T.dot(DCdDcd[:,:,j].T).dot(l[0:Ndof,i])
        self.grad = grad*Str.cdamp
        
        
    def fin_diff(self,Str,x0,h):
        print("Finite difference step: {}".format(h))
        SaveC = Str.C
        grad_ff = np.zeros((Str.Ndamper,1))
        dc0 = Str.dc
        for i in range(0,Str.Ndamper):
            xh = np.copy(x0)
            xh[i] += h
            cd_temp = Str.cdamp * xh
            Cd_temp = np.diag(cd_temp)
            Str.C = Str.Cs + np.dot(Str.H.T,np.dot(Cd_temp,Str.H))
            Str.time_hist()
            Str.calc_peakdrift(self)
            grad_ff[i,0] = (Str.dc-dc0)/h

        print("The finite differences gradient is: {}".format(grad_ff[:,0]))
        print("The analytical gradient is: {}".format(self.grad[:,0]))
        print("Ratio. ", (self.grad/grad_ff).reshape(1,2)[0,:])
        Str.dc = dc0    
        Str.C = SaveC
            
        
        
# Structural analysis class       
class FE_FRAME:
    
    def __init__(self):
        Mtemp = sio.loadmat('M.mat')
        Ktemp = sio.loadmat('K.mat')
        Htemp = sio.loadmat('H.mat')
        etemp = sio.loadmat('e.mat')
        self.M = Mtemp['M'] # mass matrix
        self.K = Ktemp['K'] # stiffness matrix
        self.H = Htemp['H'] # transofrmation matrix
        self.e = etemp['e'] # load distribution vector
        self.xi = 5.0/100
        self.Ndamper = 2
        self.cdamp = 3000.0 # kNs/m, Maximum available damping coefficient
        self.Ndof_u, self.Ndof_d = np.shape(self.H) # Number of degrees of freedom in displacement and drift cooridnates
        
    def load(self, filename):
        Ptemp = sio.loadmat(filename+'.mat')
        self.P = Ptemp[filename]
        self.Dt = self.P[0,1] - self.P[0,0]
        self.time = self.P[0]
    
    def calc_eigen(self):
        matMK = np.dot(np.linalg.inv(self.M),self.K)
        w, v = np.linalg.eig(matMK)
        self.omega = np.sort(np.sqrt(w))
        self.period = math.pi*2/self.omega
        a0 = self.xi*((2.0*self.omega[0]*self.omega[1])/(self.omega[0]+self.omega[1]))
        a1 = self.xi*(2.0/(self.omega[0]+self.omega[1]))
        self.Cs = a0*self.M + a1*self.K
        
    
    def time_hist(self):
        # Constant average acceleration method:
        beta = 1.0/4; gamma = 1.0/2
        '''# Linear acceleration method:
        beta = 1.0/6; gamma = 1.0/2'''
        self.beta = beta; self.gamma = gamma
        Ndof = self.Ndof_u
        M = self.M
        C = self.C
        K = self.K
        e = self.e
        Dt = self.Dt
        u0 = np.zeros((Ndof,1))
        v0 = np.zeros((Ndof,1))
        Ntsteps = np.shape(self.P)
        Ntsteps = Ntsteps[1]
        u = np.zeros((2,Ntsteps))
        v = np.zeros((2,Ntsteps))
        a = np.zeros((2,Ntsteps))
        u[:,0] = u0[:,0]
        v[:,0] = v0[:,0]
        P =  - np.dot(M,e*self.P[1])
        a0 = np.dot(np.linalg.inv(M),(P[:,0].reshape(2,1)-C.dot(v0)-K.dot(u0)))
        a[:,0] = a0[:,0]
        a1 = (1/(beta*Dt**2))*M + (gamma/(beta*Dt))*C
        a2 = (1/(beta*Dt))*M + (gamma/beta-1)*C
        a3 = (1/(2*beta)-1)*M + Dt*(gamma/(2*beta)-1)*C
        ui = u0; vi = v0; ai = a0
        Khat = K + a1
        KhatInv = np.linalg.inv(Khat)
        for i in range(0,Ntsteps-1):
            Phatj = P[:,i].reshape(2,1) + a1.dot(ui) + a2.dot(vi) + a3.dot(ai)
            uj = np.dot(KhatInv,Phatj)
            vj = (gamma/(beta*Dt))*(uj-ui) + (1-gamma/beta)*vi + Dt*(1-gamma/(2*beta))*ai
            aj = (1/(beta*Dt**2))*(uj-ui) - (1/(beta*Dt))*vi - (1/(2*beta)-1)*ai
            u[:,i+1] = uj[:,0]
            v[:,i+1] = vj[:,0]
            a[:,i+1] = aj[:,0]
            ui = uj; vi = vj; ai = aj
        self.disp = u; self.vel = v; self.acc = a
    
    def calc_peakdrift(self,Opt):
        # Drifts
        drift = np.dot(self.H,self.disp)
        self.drift = drift
        # Divide the drifts by the allowable value dallow
        d_bar = (1.0/Opt.dallow) * drift
        # Max drift value, one for each drift DOF
        d_barMax = np.amax(np.absolute(d_bar),axis=1)
        # Drifts normalized by the respective max vlaue, the max value now is 1
        d_cap = np.dot(np.diag(1/d_barMax),d_bar)
        # Weigths for integration, p-norm max approximation
        weigth = self.Dt*np.ones((1,len(self.time)))
        weigth[0,0] = self.Dt/2
        weigth[0,-1] = weigth[0,0]
        # Time length
        tf = self.time[-1]
        dcr = (np.sum((1/tf)*weigth*d_cap**Opt.pnorm, axis=1))**(1/Opt.pnorm)
        dc_vec = np.dot(np.diag(d_barMax),dcr)
        dc_vec_max = np.amax(dc_vec)
        dc_vec = dc_vec/dc_vec_max
        one = np.ones((2,1))
        nom = (np.dot(one.reshape(1,2),np.dot(np.diag(dc_vec**(Opt.q+1)),one)))
        den = (np.dot(one.reshape(1,2),np.dot(np.diag(dc_vec**(Opt.q)),one)))
        dc = dc_vec_max*nom/den
        self.dc = dc
        
def main(x):
    
    # Initialize optimization object
    Opt = DAMP_OPT()
    # Initialize Str object
    Str = FE_FRAME()
    # Load ground acceleration record:
    Str.load('LA02')
    
    # Calculate eigenvalues and Rayleigh damping matrix Cs:
    Str.calc_eigen()

    # If I want to switch from drift coordinates to displacements:
    Str.M = np.dot(Str.H.T,np.dot(Str.M,Str.H))
    Str.K = np.dot(Str.H.T,np.dot(Str.K,Str.H))
    Str.Cs = np.dot(Str.H.T,np.dot(Str.Cs,Str.H))
    cd = Str.cdamp * x
    Str.cd = cd
    Cd = np.diag(cd)
    Str.C = Str.Cs + np.dot(Str.H.transpose(),np.dot(Cd,Str.H))
    # Forward time-history analysis
    Str.time_hist()
    
    # Calculate maximum drift
    Str.calc_peakdrift(Opt)
    
    # Sensitivity analysis
    Opt.cal_sensitivity(Str)
    
    # Finite differences check
    '''h = 1e-6
    Opt.fin_diff(Str,x,h)'''
    
    return Str, Opt

if __name__ == '__main__':
	x = np.array([0.9, 0.9])
	main(x)

'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This code was written by Nicolò Pollini,                                %
% Technion - Israel Institute of Technology                               %
% https://mdo.net.technion.ac.il/                                         %
%                                                                         %
%                                                                         %
% Contact: nicolo@technion.ac.il                                          %
%                                                                         %
% Code repository: https://github.com/pollinico/TopOpt_Wind_Farm          %
%                                                                         %
% Disclaimer:                                                             %
% The author reserves all rights but does not guarantee that the code is  %
% free from errors. Furthermore, the author shall not be liable in any    %
% event caused by the use of the program.                                 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
