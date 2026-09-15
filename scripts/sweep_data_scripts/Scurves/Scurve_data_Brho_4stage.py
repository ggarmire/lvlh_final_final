import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import matplotlib.pyplot as plt
import lvlh_functions as lvf


def main():
    # Sweep Parameters
    S = 25
    C = 1.0
    rho = 0
    L = 4
    delta = 1000

    nruns = 500
    Ks = np.linspace(2, 7, 30)

    # arguments for B and R: 
    extra_args = {'L': L, 'rho': rho} 
    R_args = {'delta': delta}

    filestart = f'data/S_curves/4stage/B_rho/B_L4_rho_{rho:0.2f}'
    output_dir = os.path.dirname(filestart)
    if output_dir:  
        os.makedirs(output_dir, exist_ok=True)
    
    # Execute the parallel sweep
    fracs, frac_errs = lvf.sweeps.generate_S_curve_4stage(
        matrix_function = lvf.B_rho, 
        S = S, 
        C = C, 
        B_args = extra_args, 
        Ks = Ks, 
        nruns = nruns, 
        filestart = filestart,
        R_args=R_args,
        maxworkers = 6
    )
    plt.fill_between(Ks, fracs-frac_errs, fracs+frac_errs, color='green', alpha = 0.3)
    plt.plot(Ks, fracs, '.-', color='black', lw = 1)
    plt.title(f'Stability by K, {S} species, 4 stages, rho = {rho}')
    plt.grid()
    plt.xlabel('K')
    plt.ylabel('fraction of runs stable')
    plt.show()

if __name__ == "__main__":
    main()