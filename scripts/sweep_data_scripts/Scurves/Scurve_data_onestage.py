import numpy as np
import matplotlib.pyplot as plt
import lvlh_functions as lvf
import os

def main():
    # Sweep Parameters
    S = 1000
    C = 1.0
    nruns = 200
    Ks = np.linspace(0.9, 1.1, 40)

    # no need for extra arguments in the 1 stage case!
    extra_args = {} 

    filestart = 'data/S_curves/one_stage/one_stage'
    output_dir = os.path.dirname(filestart)
    if output_dir: 
        os.makedirs(output_dir, exist_ok=True)
    
    # Execute the parallel sweep
    fracs, frac_errs = lvf.sweeps.generate_S_curve(
        matrix_function = lvf.A_onestage, 
        S = S, 
        C = C, 
        B_args = extra_args, 
        Ks = Ks, 
        nruns = nruns, 
        filestart = filestart,
        maxworkers = 5
    )
    plt.fill_between(Ks, fracs-frac_errs, fracs+frac_errs, color='green', alpha = 0.3)
    plt.plot(Ks, fracs, '.-', color='black', lw = 1)
    plt.title(f'Stability by K, {S} species, 1 stage')
    plt.grid()
    plt.xlabel('K')
    plt.ylabel('fraction of runs stable')
    plt.show()

if __name__ == "__main__":
    main()