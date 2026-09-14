import numpy as np
import concurrent.futures
import matplotlib.pyplot as plt
import time

import lvlh_functions as lvf

def main():

    seed = np.random.randint(0, 1000)
    #seed = 1
    print(f'seed={seed}')

    S = 250
    rho = 1
    C = 0.1

    Kset = 2

    sigma = Kset * (S*C)**(-0.5)
    K = sigma * (S*C)**0.5

    B = lvf.B_rho(S, C, sigma, seed, L=2, rho=rho)


    row_idx = np.arange(2*S)[:, None]
    col_idx = np.arange(2*S)[None, :]
    
    diag_block_mask = (row_idx // 2) == (col_idx // 2)

    off_block_entries = B[~diag_block_mask]

    var_entries = np.var(off_block_entries, ddof=1)

    print(f"Sample variance of off-block entries: {var_entries:.6f}")
    print(f"Theoretical variance (C*sigma^2):        {C*sigma**2:.6f}")




if __name__ == "__main__":
    main()


    