'''
For a set of delta values, find the K value at which 50% of runs are stable (using large Delta limit)
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
import concurrent.futures

import time 

def main():

    save_on = True     # flag to save the data at the end. 
    #save_on = False     # flag to (not( save the data at the end. 

    Cs = np.linspace(0.1, 1, 10)

    delta = 1000
    rho = 1
    L = 2
    nruns = 200

    S = 500

    Kguess = 1      # guess a K to make it go a bit faster 

    maxworks = 45


    # save data stuff here: 
    filestart = f'data/C_data/K50_ofC/S={S}/'
    output_dir = os.path.dirname(filestart)
    if output_dir:  
        os.makedirs(output_dir, exist_ok=True)

    filename = f"{filestart}K50byC_rho{rho:0.2f}_C{np.min(Cs)}-{np.max(Cs)}_{nruns}rpk.npz"

    if save_on: 
        print(f'data will be saved to {filename}')
    else:
        print('data from this will NOT BE SAVED.')

    # save data out 
    nCs = len(Cs)
    K50s = np.zeros(nCs)
    K50_errs = np.zeros(nCs)

    # get k50 and error for each S
    with concurrent.futures.ProcessPoolExecutor(max_workers=maxworks) as executor:
        for i, C in enumerate(Cs):
            print(f'on C={C}')
            start = time.time()
            K50, K50_err, _ = lvf.find_K50_threshold_rho_delta(
                S, C, rho, delta, nruns, executor=executor, K_guess=Kguess
            )
            '''for i, delta in enumerate(deltas):
            print(f'on delta={delta}')
            start = time.time()
            K50, K50_err, _ = lvf.find_K50_threshold_rho_delta(S, C, rho, delta, nruns, K_guess=Kguess, maxworkers=40)'''
            K50s[i], K50_errs[i] = K50, K50_err
            end = time.time()
            print(f"C={C}, rho={rho}, delta={delta} -> K50 = {K50:.4f}+-{K50_err:.4f}")
            print(f"    took {(end-start)/60} min ({end-start} sec)")

    # save the data! 
    np.savez_compressed(filename, Cs=Cs, K50s=K50s, K50_errs=K50_errs, S=S, rho=rho, delta=delta)
    print(f"\nData saved successfully to: {filename}")

    print(f'Cs \n{Cs}')
    print(f'K50s \n{K50s}')
    print(f'K50errss \n{K50_errs}')
    # plot to see 
    plt.figure(figsize=(8,6))
    plt.errorbar(Cs, K50s, xerr=K50_errs, fmt='o--', ms=10, lw=2, capsize=5)
    plt.xlabel('delta')
    plt.ylabel('K for which 50% of runs are stable')
    plt.title(f'K50(delta), S={S}, rho={rho}, nruns={nruns}')

    plt.show()
    

if __name__ == "__main__":
    main()
