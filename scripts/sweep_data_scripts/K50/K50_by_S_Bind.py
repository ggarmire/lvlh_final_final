'''
For a set of S values, find the K value at which 50% of runs are stable (using large Delta limit)
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

def main():

    save_on = True     # flag to save the data at the end. 
    #save_on = False     # flag to save the data at the end. 

    # takes ~30 min to run for Ss = [500, 200, 100, 50, 10]
    Ss = [1000, 500, 200, 100, 50, 25, 10]
    C = 1.0

    L = 2
    delta = 1000
    nruns = 100

    Kguess= 1

    # save data stuff here: 
    filestart = f'data/K50_data/K50_ofS/delta={delta}/'
    output_dir = os.path.dirname(filestart)
    if output_dir:  
        os.makedirs(output_dir, exist_ok=True)
    filename = f"{filestart}K50byS_Bind_S{np.min(Ss)}-{np.max(Ss)}_{nruns}rpk.npz"

    if save_on: 
        print(f'data will be saved to {filename}')
    else:
        print('data from this will NOT BE SAVED.')

    # save data out 
    nS = len(Ss)
    K50s = np.zeros(nS)
    K50_errs = np.zeros(nS)

    # get k50 and error for each S
    for i, S in enumerate(Ss):
        print(f'on S={S}')
        start = time.time()
        K50, K50_err, _ = lvf.find_K50_threshold_Bind_delta(S, C, delta, nruns, K_guess=Kguess, maxworkers=40)
        K50s[i], K50_errs[i] = K50, K50_err
        end = time.time()
        print(f"S={S} delta={delta} -> K50 = {K50:.4f}+-{K50_err:.4f}")
        print(f"    took {(end-start)/60} min ({end-start} sec)")

    # save the data! 
    np.savez_compressed(filename, Ss=Ss, K50s=K50s, K50_errs=K50_errs, delta=delta, rho=rho)
    print(f"\nData saved successfully to: {filename}")

    print(f'S={Ss}\n K50s={K50s}\n Kerr={K50_err}')
    # plot to see 
    plt.figure(figsize=(8,6))
    plt.errorbar(Ss, K50s, xerr=K50_err, fmt='o--', ms=10, lw=2, capsize=5)
    plt.xlabel('number of species S')
    plt.ylabel('K for which 50% of runs are stable')
    plt.title(f'K50(S), delta={delta}, rho={rho}, nruns={nruns}')

    plt.show()
    

if __name__ == "__main__":
    main()
