''' 
Nicolò Pollini, Jan. 2024
Haifa, Israel
'''
#--------------------------------------
# Import relevant modules:
from scipy.optimize import minimize, Bounds, NonlinearConstraint
import numpy as np
import matplotlib.pyplot as plt
import Analysis2DOF as AS
#--------------------------------------

#--------------------------------------
# Functions:
def runOptimization(params):
    x0 = params['x0']
    lb = params['lb']
    ub = params['ub']
    objHist = []
    gHist = []
    iter = 0
    # Initialize optimization object
    Opt = AS.DAMP_OPT()
    # Initialize Str object
    Str = AS.FE_FRAME()
    # Load ground acceleration record:
    Str.load('LA02')
    # Calculate eigenvalues and Rayleigh damping matrix Cs:
    Str.calc_eigen()
    # If I want to switch from drift coordinates to displacements:
    Str.M = np.dot(Str.H.T,np.dot(Str.M,Str.H))
    Str.K = np.dot(Str.H.T,np.dot(Str.K,Str.H))
    Str.Cs = np.dot(Str.H.T,np.dot(Str.Cs,Str.H))
    Opt.q = 100
    Opt.pnorm = 100
    f0 = 1.0
    def objcon(x):
        nonlocal Opt, Str, objHist, gHist, iter, f0
        iter += 1
        # Update damping
        cd = Str.cdamp * x
        Cd = np.diag(cd)
        Str.C = Str.Cs + np.dot(Str.H.T,np.dot(Cd,Str.H))
        # Forward time-history analysis
        Str.time_hist()
        # Calculate maximum drift
        Str.calc_peakdrift(Opt)
        # Sensitivity analysis
        Opt.cal_sensitivity(Str)
        f = np.sum(x)
        df = np.ones(2)
        if iter == 1:
            f0 = f
            f = f/f0
            df = df/f0
        else:
            f = f/f0
            df = df/f0
        objHist.append(f)
        g = Str.dc
        gHist.append(g[0,0])
        dg = Opt.grad.reshape(1,2)
        return f, df, g[0,0], dg
    
    #callback(xk, OptimizeResult state) if trust-constr
    def callback(xk):
        nonlocal Opt
        # Continuation scheme
        if iter % 2 == 0:
            Opt.q = min(Opt.q+100, 1.0E3)
            Opt.pnorm = min(Opt.pnorm+100, 1.0E3)
        print('Iter {0:2d}, objective f: {1:.3f}, constraint g<=1: {2:1.3f}, p: {3:.2f}, q: {4:.2f}'.format(iter, flast, glast, Opt.pnorm, Opt.q))
        return False

    # ---- general code ----
    xlast = []
    flast = []
    dflast = []
    glast = []
    dglast = []

    def obj(x):
        nonlocal xlast, flast, dflast, glast, dglast
        if not np.array_equal(x, xlast):
            flast, dflast, glast, dglast = objcon(x)
            xlast = x
        return (flast, dflast)
    
    def con (x):
        nonlocal xlast, flast, dflast, glast, dglast
        if not np.array_equal(x, xlast):
            flast, dflast, glast, dglast = objcon(x)
            xlast = x
        return glast

    def jac (x):
        nonlocal xlast, flast, dflast, glast, dglast
        if not np.array_equal(x, xlast):
            flast, dflast, glast, dglast = objcon(x)
            xlast = x
        return dglast
    # ----------------------
    nlc = NonlinearConstraint(con, lb=-np.inf, ub=1.0, jac=jac)
    options = {'disp': True, 'ftol': 1e-5} # with SLSQP
    #options = {'disp': True} # with trust-constr
    bounds = Bounds(lb, ub, keep_feasible=True)
    sol = minimize(obj, x0, jac=True, constraints=nlc, options=options, bounds=bounds, callback=callback, method='slsqp')
    print("x =", sol.x)
    print("f =", sol.fun)
    print("cd =", sol.x*Str.cdamp)
    print(sol.success)
    return sol.x, sol.fun, objHist, gHist, Str

# Main function
if __name__ == "__main__":
    x0 = [.9, .9]
    lb = [0., 0.]
    ub = [1., 1.]
    params = {'x0': x0, 
              'lb': lb,
              'ub': ub}
    xopt, fopt, objHist, gHist, Str = runOptimization(params)

    # Update damping
    cd = Str.cdamp * xopt
    Cd = np.diag(cd)
    Str.C = Str.Cs + np.dot(Str.H.T,np.dot(Cd,Str.H))
    # Forward time-history analysis
    Str.time_hist()
    drift = np.dot(Str.H, Str.disp)

    fig = plt.figure()
    plt.plot(Str.time, drift[0,:]*1e3, label='d1=u1', alpha=0.9, linewidth=2)
    plt.plot(Str.time, drift[1,:]*1e3, label='d2=u2-u1', alpha=0.7, linewidth=2)
    plt.plot(Str.time, 9*np.ones(len(Str.time)), "k--")
    plt.plot(Str.time, -9*np.ones(len(Str.time)), "k--")
    plt.title("Inter-story drifts in time with optimal damping")
    plt.xlabel("time [s]")
    plt.ylabel("d1, d2 [mm]")
    plt.legend()
    fig.savefig('SLSQP_opt_d1d2.png', bbox_inches='tight', dpi=300)

    fig = plt.figure()
    plt.plot(objHist/max(objHist), label='obj=cd1+cd2', linewidth=2)
    plt.plot(gHist, label='g<=1', linewidth=2)
    plt.title("Normalized objective and drift constraint functions")
    plt.legend()
    fig.savefig('SLSQP_opt_iters.png', bbox_inches='tight', dpi=300)
    plt.show()
#--------------------------------------
    
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
