'''
For a set of delta values, find the K value at which 50% of runs are stable (using large Delta limit)
'''
import numpy as np
import matplotlib.pyplot as plt
import lvlh_functions as lvf
import os
import time 

def main():

    save_on = True     # flag to save the data at the end. 
    #save_on = False     # flag to save the data at the end. 

    deltas = [0, 1, 10, 100]
    C = 1.0
    rho = -1./3.
    L = 2
    nruns = 200

    S = 50

    Kguess = 1      # guess a K to make it go a bit faster 



    # save data stuff here: 
    filestart = f'data/K50_data/K50_ofS/S={S}/'
    output_dir = os.path.dirname(filestart)
    if output_dir:  
        os.makedirs(output_dir, exist_ok=True)
    filename = f"{filestart}K50bydelta_rho{rho:0.2f}_delta{np.min(deltas)}-{np.max(deltas)}_{nruns}rpk.npz"

    if save_on: 
        print(f'data will be saved to {filename}')
    else:
        print('data from this will NOT BE SAVED.')

    # save data out 
    nS = len(deltas)
    K50s = np.zeros(nS)
    K50_errs = np.zeros(nS)

    # get k50 and error for each S
    for i, delta in enumerate(deltas):
        print(f'on delta={delta}')
        start = time.time()
        K50, K50_err, _ = lvf.find_K50_threshold_rho_delta(S, C, rho, delta, nruns, K_guess=Kguess)
        K50s[i], K50_errs[i] = K50, K50_err
        end = time.time()
        print(f"delta={delta}, rho={rho}, delta={delta} -> K50 = {K50:.4f}+-{K50_err:.4f}")
        print(f"    took {(end-start)/60} min ({end-start} sec)")

    # save the data! 
    np.savez_compressed(filename, deltas=deltas, K50s=K50s, K50_errs=K50_errs, S=S, rho=rho)
    print(f"\nData saved successfully to: {filename}")

    # plot to see 
    plt.figure(figsize=(8,6))
    plt.errorbar(deltas, K50s, xerr=K50_err, fmt='o--', ms=10, lw=2, capsize=5)
    plt.xlabel('delta')
    plt.ylabel('K for which 50% of runs are stable')
    plt.title(f'K50(delta), S={S}, rho={rho}, nruns={nruns}')

    plt.show()
    

if __name__ == "__main__":
    main()
