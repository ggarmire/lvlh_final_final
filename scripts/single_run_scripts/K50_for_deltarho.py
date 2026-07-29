import numpy as np
import concurrent.futures
import matplotlib.pyplot as plt
import time

import lvlh_functions as lvf

def main():
    # set parameters 
    S = 120
    rho = 0
    delta = 1000    
    nruns = 100
    maxworkers = 6

    C = 1
    start = time.time()

    K50, K50_err, K_history_fracs = lvf.find_K50_threshold_rho_delta(S, C, rho, delta, nruns)

    end = time.time()

    print(f"S={S}, rho={rho}, delta={delta} -> K50 = {K50:.4f}")
    print(f"took {(end-start)/60} min ({end-start} sec)")


    plt.figure(figsize=(8, 6))
    Ks = sorted(K_history_fracs.keys())
    fracs = [K_history_fracs[k] for k in Ks]

    plt.plot(Ks, fracs, 'o--', color='black', label='evaluated Ks')
    #plt.plot(K50, 0.5, 'x', color='red', markersize=12, label=f'Calculated $K_{{50}}$ = {K50:.3f}')
    plt.errorbar(K50, 0.5, xerr=K50_err, fmt='o', color='red', ms=12, lw=2, capsize=5, label='50% with bootstrap error')
    plt.axhline(0.5, color='black', linestyle='--', alpha=0.5)
    plt.legend()
    plt.xlabel('K')
    plt.ylabel('stable fraction')
    plt.title(f'finding K50 for rho={rho}, S={S}, delta={delta}, nruns={nruns}')

    plt.show()


if __name__ == "__main__":
    main()


    