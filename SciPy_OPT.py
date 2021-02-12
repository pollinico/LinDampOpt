''' 
Nicolò Pollini, May 2021
Copenhagen, Denmark
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
    def objcon(x):
        nonlocal Opt, Str, objHist, gHist, iter
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
        objHist.append(f)
        df = np.ones(2)
        g = Str.dc
        gHist.append(g[0,0])
        dg = Opt.grad.reshape(1,2)
        print('Iter {0:2d}, objective f: {1:.3f}, constraint g<=1: {2:1.3f}'.format(iter,f,g[0,0]))
        #print("Iter: ",iter," f: ", f, "g: ", g[0,0])
        return f, df, g[0,0], dg

    def callback(xk):
        nonlocal Opt
        # Continuation scheme
        Opt.q = min(Opt.q+100, 1.0E3)
        Opt.pnorm = min(Opt.pnorm+100, 1.0E3)
        return True

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
    nlc = NonlinearConstraint(con, lb=-np.inf, ub=1.0, jac=jac, keep_feasible=True)
    options = {'disp': True}
    bounds=Bounds(lb, ub, keep_feasible=True)
    sol = minimize(obj, x0, jac=True, constraints=nlc, options=options, bounds=bounds, callback=callback, method='SLSQP')
    print("x =", sol.x)
    print("f =", sol.fun)
    print("cd =", sol.x*Str.cdamp)
    print(sol.success)
    return sol.x, sol.fun, objHist, gHist

# Main function
if __name__ == "__main__":
    x0 = [0.95, 0.95]
    lb = [0., 0.]
    ub = [1., 1.]
    params = {'x0': x0, 
              'lb': lb,
              'ub': ub}
    xopt, fopt, objHist, gHist = runOptimization(params)

    plt.figure()
    plt.plot(objHist/max(objHist), label='obj')
    plt.plot(gHist, label='g')
    plt.title("Normalized objective and drift constraint")
    plt.legend()
    plt.show()
#--------------------------------------
'''
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# This code was written by Nicolò Pollini,                                %
# Department of Wind Energy,                                              %  
# Technical University of Denmark.                                        %
#                                                                         %
# Contact: nicolo@alumni.technion.ac.il                                   %
#                                                                         %
#                                                                         %
# Disclaimer:                                                             %
# The author reserves all rights but does not guarantee that the code is  %
# free from errors. Furthermore, the author shall not be liable in any    %
# event caused by the use of the program.                                 %
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''