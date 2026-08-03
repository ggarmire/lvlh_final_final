'''
For a grid of delta/rho values, find the K value at which 50% of runs are stable (using large Delta limit)
'''
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import matplotlib.pyplot as plt
import lvlh_functions as lvf

import time 
from contextlib import redirect_stdout

def main():

    save_on = True     # flag to save the data at the end. 
    #save_on = False     # flag to save the data at the end. 
    ndeltas = 3
    nrhos = 4

    nruns = 100
    S = 25
    
    deltas = np.logspace(-3, 3, ndeltas) 
    rhos = np.linspace(-1./3., 1, nrhos)

    C = 1.0
    L = 2

    #Kguess = 1      # guess a K to make it go a bit faster 

    # save data stuff here: 
    filestart = f'data/K50_data/K50_deltarhomap/'
    output_dir = os.path.dirname(filestart)
    if output_dir:  
        os.makedirs(output_dir, exist_ok=True)
    filename = f"{filestart}K50_deltarhomap_S{S}_{nruns}rpk.npz"

    if save_on: 
        print(f'data will be saved to {filename}')
    else:
        print('data from this will NOT BE SAVED.')

    #initialize output arrays
    K50s = np.zeros((nrhos, ndeltas)) 
    K50_errs = np.zeros((nrhos, ndeltas)) 
    runtimes = np.zeros((nrhos, ndeltas))

    for irho, rho in enumerate(rhos):
        Kguess = 2/((1+3*rho)**(-0.5)) if rho>-0.25 else 5      # for a closer starting point without having a 1/0 issue
        for jdelta, delta in enumerate(deltas):
            print(f'on rho = {rho}, delta = {delta}')
            start = time.time()
            with open(os.devnull, 'w') as f, redirect_stdout(f):        # surpress print statements
                K50, K50_err, _ = lvf.find_K50_threshold_rho_delta(S, C, rho, delta, nruns, K_guess=Kguess, maxworkers=6)
                print(f'    K50={K50:.3f} +- {K50_err:.3f}')
            end = time.time()

            K50s[irho, jdelta] = K50
            K50_errs[irho, jdelta] = K50_err
            runtimes[irho, jdelta] = end-start

    # save the data! 
    if save_on:
        np.savez_compressed(filename, deltas=deltas, rhos=rhos, K50s=K50s, K50_errs=K50_errs, runtimes=runtimes, nruns=nruns, S=S)
        print(f"\nData saved successfully to: {filename}")


    # plot to see 
    plt.figure(figsize=(7,7))
    im = plt.pcolormesh(deltas, rhos, K50s, cmap='viridis', shading='auto')
    cbar = plt.colorbar(im)
    cbar.set_label('K50 Threshold', rotation=270, labelpad=15)
    plt.xscale('symlog', linthresh=0.5)
    plt.xticks(ticks=deltas, labels=[str(d) for d in deltas])
    plt.yticks(ticks=rhos, labels=[f"{r:.2f}" for r in rhos])
    plt.xlabel('Scale Parameter ($\\delta$) - SymLog Scale')
    plt.ylabel('Correlation ($\\rho$)')
    plt.title(f'$K_{{50}}$ Stability Heatmap (S={S}, nruns={nruns})')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()







    