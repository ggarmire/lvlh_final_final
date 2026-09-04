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
    S = 1000
    C = 0.1
    L = 2
    delta = 1000

    nruns = 500
    Ks = np.linspace(0, 2, 50)

    # arguments for B and R: 
    extra_args = {'L': L} 
    R_args = {'delta': delta}

    filestart = f'data/C_data/S_curves/Bind_C{C}'
    output_dir = os.path.dirname(filestart)
    if output_dir:  
        os.makedirs(output_dir, exist_ok=True)
    
    # Execute the parallel sweep
    fracs, frac_errs = lvf.sweeps.generate_S_curve(
        matrix_function = lvf.B_ind, 
        S = S, 
        C = C, 
        B_args = extra_args, 
        Ks = Ks, 
        nruns = nruns, 
        filestart = filestart,
        R_args=R_args,
        maxworkers = 25
    )
    plt.fill_between(Ks, fracs-frac_errs, fracs+frac_errs, color='green', alpha = 0.3)
    plt.plot(Ks, fracs, '.-', color='black', lw = 1)
    plt.title(f'Stability by K, {S} species, 2 stages')
    plt.grid()
    plt.xlabel('K')
    plt.ylabel('fraction of runs stable')
    plt.show()

if __name__ == "__main__":
    main()