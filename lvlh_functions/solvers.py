import numpy  as np
from scipy import integrate 

# system evolution 

def evolve_system(B, R, x0, ts):
    '''
    Given an initial condition, interaction matrix, and growth vector/demographic matrix, 
    evolve the system in time to see species dynamics and long time behavior. (n = S*L)

    A/B = interaction matrix (can be B): nxn
    r/R = growth rates or growth matrix: nxn
    x0 = vector of initial states: 1xn
    ts = array of times to use for integration
    returns sol 
    '''
    if R.ndim == 1: R = np.diag(R)
    def derivative(x, t, B, R):
        dxdt = np.dot(R, x) + np.multiply(x, np.dot(B, x))
        for i in range(len(x0)):
            if x[i] <= 1e-8:        # numerical noise eliminated 
                dxdt[i] = 0
        return dxdt
    sol = integrate.odeint(derivative, x0, ts, args = (B, R))
    return sol


# Jacobian 
def Jacobian(B, R):
    '''
    find the equilibrium Jacobian (at x*=1) given the interaction matrix and demographic rates. 
    
    A/B = interaction matrix (can be B): nxn
    r/R = growth rates or growth matrix: nxn
    
    '''
    if R.ndim == 1: R = np.diag(R)
    n = B.shape[0]
    xf = np.ones(n)
    diag_term = np.diag(np.dot(B, xf))    
    Bx = np.multiply(np.outer(xf, np.ones(n)), B)
    return R + diag_term + Bx
