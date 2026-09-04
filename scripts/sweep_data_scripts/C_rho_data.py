import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import numpy as np
import concurrent.futures
import lvlh_functions as lvf

def main():
    S = 500
    delta = 1000
    nruns = 200
    mw = 76
    
    Cs = np.linspace(0.2, 1.0, 5)
    #Cs = np.linspace(0.1, 1.0, 2)
    rhos = np.linspace(-0.2, 1.0, 7)
    #rhos = np.linspace(0, 1.0, 9)
    #rhos = np.linspace(0, 1.0, 2)
    
    K50s = np.zeros((len(Cs), len(rhos)))
    K50_errs = np.zeros((len(Cs), len(rhos)))
    
    print(f"Starting C vs rho K50 sweep. Total points: {len(Cs)*len(rhos)}")
    start = time.time()
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=mw) as executor:
        for i, C in enumerate(Cs):
            for j, rho in enumerate(rhos):
                if rho > -0.3:
                    K_guess = round(2 * (1 + 3 * rho)**(-0.5), 2)
                else: K_guess = 40
                
                K50, K50_err, _ = lvf.find_K50_threshold_rho_delta(
                    S, C, rho, delta, nruns, executor, K_guess=K_guess
                )
                
                K50s[i, j] = K50
                K50_errs[i, j] = K50_err
                print(f"C={C:.2f}, rho={rho:.2f} -> K50={K50:.3f}")

    print(f"Sweep completed in {(time.time() - start)/60:.2f} minutes.")
    
    os.makedirs('data', exist_ok=True)
    filename = f"data/Crho_K50_heatmap_S{S}_{nruns}rpk.npz"
    np.savez_compressed(filename, Cs=Cs, rhos=rhos, K50s=K50s, K50_errs=K50_errs, S=S, delta=delta)
    print(f"Data saved to {filename}")

if __name__ == "__main__":
    main()