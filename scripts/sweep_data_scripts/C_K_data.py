import os

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

def _worker_CK_point(args):
    """Evaluates the stable fraction for a specific C and K combination."""
    i_C, j_K, S, C, K, rho, delta, nruns = args
    
    stable_count = 0
    sigma = K * (S * C)**(-0.5)
    
    # Run the ensemble for this point
    for seed in range(nruns):
        B = lvf.B_rho(S, C, sigma, seed, L=2, rho=rho)
        R = lvf.R_star_2stage_delta(B, delta)
        if lvf.check_stable(B, R):
            stable_count += 1
            
    return i_C, j_K, stable_count / nruns

def main():
    S = 10
    rho = 0.0
    delta = 1000
    nruns = 200
    mw = 20
    
    Cs = np.linspace(0.1, 1.0, 10)  
    Ks = np.linspace(1, 3, 21) 
    
    nCs = len(Cs)
    nKs = len(Ks)
    stable_fracs = np.zeros((nCs, nKs))
    
    tasks = []
    for i_C, C in enumerate(Cs):
        for j_K, K in enumerate(Ks):
            tasks.append((i_C, j_K, S, C, K, rho, delta, nruns))
            
    print(f"Starting C vs K heatmap generation. Total points: {len(tasks)}")
    
    start = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=mw) as executor:
        results = executor.map(_worker_CK_point, tasks)
        for i_C, j_K, frac in results:
            stable_fracs[i_C, j_K] = frac
            
    print(f"Sweep completed in {(time.time() - start)/60:.2f} minutes.")
    
    # Save the data
    os.makedirs('data', exist_ok=True)
    filename = f"data/C_data/CK_heatmap_S{S}_rho{rho}_{nruns}rpk.npz"
    np.savez_compressed(filename, Cs=Cs, Ks=Ks, stable_fracs=stable_fracs, S=S, rho=rho, delta=delta)
    print(f"Data saved to {filename}")

if __name__ == "__main__":
    main()