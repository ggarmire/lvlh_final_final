import numpy as np
import lvlh_functions as lvf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
import os 

def main():
    rhos = [0, 1]
    high_delta = 1000
    high_S = 50
    current_dir = os.path.dirname(os.path.abspath(__file__))

    K50maxbar = 10

    # read in data here: 


    # get data for k50 vs S 
    data_KofS = {}
    k50S_dir = os.path.join(current_dir, '..', '..', 'data', 'K50_data', 'K50_ofS', f'delta={high_delta}')
    for irho, rho in enumerate(rhos):
        filepath = os.path.join(k50S_dir, f'K50byS_rho{rho:.2f}_S10-200_200rpk.npz')
        with np.load(filepath) as data: 
            data_KofS[rho] = {'K50s': data['K50s'], 'K50_errs': data['K50_errs'], 'Ss': data['Ss']}

    # get data for k50 vs delta 
    data_Kofdelta = {}
    k50delta_dir = os.path.join(current_dir, '..', '..', 'data', 'K50_data', 'K50_ofdelta', f'S={high_S}')
    for irho, rho in enumerate(rhos):
            filepath = os.path.join(k50delta_dir, f'K50bydelta_rho{rho:.2f}_delta0-1000_200rpk.npz')
            with np.load(filepath) as data: 
                data_Kofdelta[rho] = {'K50s': data['K50s'], 'K50_errs': data['K50_errs'], 'deltas': data['deltas']}

    # get data for rho vs delta map
    rhodelta_map_dir = os.path.join(current_dir, '..', '..', 'data', 'K50_data', 'K50_deltarhomap')
    rhodelta_file = os.path.join(rhodelta_map_dir, 'K50_deltarhomap_s25_100rpk.npz')
    data_map = np.load(rhodelta_file)
    rd_rhos, rd_deltas, rd_K50s, rd_K50errs = data_map['rhos'], data_map['deltas'], data_map['K50s'], data_map['K50_errs']
    K50_maxmeasured = np.max(rd_K50s)
    print(f'max k50 here is {K50_maxmeasured}')
    K50maxplot = min(K50maxbar, K50_maxmeasured)


    # make plots here: 

    # plot setup 
    fig = plt.figure(figsize=(18, 7))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], hspace=0.35)
    ax1 = fig.add_subplot(gs[0, 0])  
    ax2 = fig.add_subplot(gs[1, 0])  
    ax3 = fig.add_subplot(gs[:, 1])


    # S vs K50
    for rho in rhos:
        d = data_KofS[rho]
        ax1.errorbar(d['Ss'], d['K50s'], yerr=d['K50_errs'], fmt='o--', capsize=3, label=fr'$\rho = {rho}$')
    ax1.set_title(fr'$K_{{50}}$ vs number of species S ($\delta = {high_delta}$)')
    ax1.set_xlabel('number of species (S)')
    ax1.set_ylabel(r'stability threshold ($K_{50}$)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # delta vs K50
    for rho in rhos:
        d = data_Kofdelta[rho]
        ax2.errorbar(d['deltas'], d['K50s'], yerr=d['K50_errs'], fmt='o--', capsize=3, label=fr'$\rho = {rho}$')
    ax2.set_title(fr'$K_{{50}}$ vs Delta ($S = {high_S}$)')
    ax2.set_xlabel(r'$\Delta$')
    ax2.set_ylabel(r'stability threshold ($K_{50}$)')
    ax2.legend()
    ax2.set_xscale('symlog')
    ax2.grid(True, alpha=0.3)

    # rho/delta map 
    x_indices = np.arange(len(rd_deltas))
    #mesh = ax3.pcolormesh(rd_deltas, rd_rhos, rd_K50s, cmap='viridis', shading='auto')
    mesh = ax3.pcolormesh(x_indices, rd_rhos, rd_K50s, cmap='rainbow', shading='auto', norm=LogNorm(vmax=K50maxbar))
    ax3.set_title(r'$K_{50}$ for $\rho$, $\Delta$ ($S = 25$)')
    ax3.set_xlabel(r'$\Delta$')
    ax3.set_ylabel(r'correlation $\rho$')
    #ax3.set_xscale('symlog')

    ax3.set_yticks(rd_rhos)
    ax3.set_yticklabels([f'{r:.2f}' for r in rd_rhos])
    ax3.set_xticks(x_indices)
    ax3.set_xticklabels([f'{d:g}' for d in rd_deltas])

    
    if K50_maxmeasured>K50maxbar: cbar = fig.colorbar(mesh, ax=ax3, extend='max')
    else: cbar = fig.colorbar(mesh, ax=ax3)
    cbar.set_label(r'$K_{50}$')
    ticks = cbar.get_ticks()
    tick_labels = [f'{tick:g}' for tick in ticks]
    if K50_maxmeasured>K50maxbar: tick_labels[-1] = rf'$\geq {K50maxplot}$'


    plt.tight_layout()
    plt.show()




if __name__ == "__main__": main()