import numpy as np
import lvlh_functions as lvf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os 

# evolve and get Jacobian for a single run with 1 stage per species 
def single_run_single_stage(S, K, C, ts, Aseed, x0seed=None):
    if x0seed == None: x0seed = Aseed 
    x0 = lvf.init_cond(S, 1, 0.2, x0seed)

    sigma = K * (S*C)**(-0.5)
    A = lvf.A_onestage(S, C, sigma, seed=Aseed)
    r = lvf.r_star(A)

    result = lvf.evolve_system(A, r, x0, ts)
    Jac = lvf.Jacobian(A, r)
    
    return result, Jac

def main():
    # common parameters: 
    S = 2     # number of species 
    C = 1       # connectance
    ts = np.linspace(0, 100, 1000)

    # low K run: 
    K1 = 20
    Aseed1 = 1
    result1, Jac1 = single_run_single_stage(S, K1, C, ts, Aseed1)
    J1eig, _ = np.linalg.eig(Jac1)

    

    

    fig, res1 = plt.subplots(figsize=(14, 7))
    box_props = dict(boxstyle='square', facecolor='white', alpha=0.7, edgecolor='None')
    K1text = f'Single Stage \nS = {S}, C = {C}, K = {K1}'
    axfontsize = 12
    
   
    

    res1.plot(ts, result1[:, 10:], alpha = 0.1, lw = 0.8, color='grey')
    res1.plot(ts, result1[:, :10], lw = 1.5)
    #res1.grid()
    res1.set_xlabel('time', fontsize=axfontsize)
    res1.set_ylabel('abundance', fontsize=axfontsize)
    #res1.set_title(f'K={K1}')
    res1.text(0.05, 0.97, K1text, transform=res1.transAxes, fontsize=9, verticalalignment='top', bbox=box_props)
    res1.set_xlim((0, 100))
    #res1.set_ylim(bottom=0)


    plt.show()



    
if __name__ == "__main__": main()