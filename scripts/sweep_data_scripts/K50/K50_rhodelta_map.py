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

    save_on = True
    #save_on = False

    ndeltas = 10
    nrhos = 15

    nruns = 100
    S = 500
    
    deltas = np.logspace(-3, 3, ndeltas) 
    rhos = np.linspace(-1./3., 1, nrhos)

    C = 1.0

    # save data stuff here: 
    filestart = f'data/K50_data/K50_deltarhomap/' if save_on else None


    K50s, K50_errs, runtimes = lvf.generate_rhodelta_map(S, C, rhos, deltas, nruns, filestart, maxworkers=80)





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







    