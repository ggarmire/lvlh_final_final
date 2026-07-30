import numpy as np
import lvlh_functions as lvf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os 

def main():
    rhos = [-1./3., 0, 0.5, 1]
    high_delta = 1000
    high_S = 100
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # get data for k50 vs S 
    k50S_dir = os.path.join(current_dir, '..', '..', 'data', 'K50_data', 'K50_ofS', f'delta={high_delta}')
    for irho, rho in enumerate(rhos):
        filepath = os.path.join(k50S_dir, f'K50byS_rho{rho:.2f}_S10-200_200rpk.npz')
        with np.load(filepath) as data: 
            K50s, K50_errs = data['K50s'], data['K50_errs']
