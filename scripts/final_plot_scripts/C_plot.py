import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from brokenaxes import brokenaxes






def main():
    rhos = [-1./3., 0, 0.5, 1]
    rhocols = ['tab:blue', 'tab:green', 'tab:orange', 'tab:red']
    high_delta = 1000
    high_S = 500

    # load data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_KofC = {}
    k50S_dir = os.path.join(current_dir, '..', '..', 'data', 'C_data', 'K50_ofC', f'S={high_S}')

    data_frac = np.load("data/C_data/CK_heatmap_S100_rho0.0_200rpk.npz")
    
    for irho, rho in enumerate(rhos):
        filepath = os.path.join(k50S_dir, f'K50byC_rho{rho:.2f}_C0.1-1.0_200rpk.npz')
        with np.load(filepath) as data: 
            data_KofC[rho] = {'K50s': data['K50s'], 'K50_errs': data['K50_errs'], 'Cs': data['Cs']}


    Cs = data_KofC[rhos[0]]['Cs']
    n_C = len(Cs)
    K50_matrix = np.zeros((len(rhos), n_C))
    err_matrix = np.zeros((len(rhos), n_C))
    
    for i, rho in enumerate(rhos):
        K50_matrix[i, :] = data_KofC[rho]['K50s']
        err_matrix[i, :] = data_KofC[rho]['K50_errs']

    


    
    #### PLOTTING BELOW HERE #### 

    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 2, width_ratios=[0.8, 1], height_ratios=[3, 1], wspace=0.25, hspace=0)
    
    # panel a
    y_break = ((0.7, 2.3), (45.7, 47.5))
    ax_K50s = brokenaxes(ylims=y_break, subplot_spec=gs[0, 0], fig=fig, hspace=0.15, d=0.012, tilt=45, despine=False)
    ax_errs = fig.add_subplot(gs[1,0])
    ax_heat = fig.add_subplot(gs[:,1])


    # left side: K50s, errors and variation from mean 
    offsets = np.linspace(-0.015, 0.015, len(rhos)-1)

    for i, rho in enumerate(rhos):
        d = data_KofC[rho]
        Cs = d['Cs']    
        K50s = d['K50s']
        K50_errs = d['K50_errs']
        label = r'$\rho = -⅓$' if rho == -1./3. else fr'$\rho = {rho}$'
        ax_K50s.errorbar(Cs, K50s, yerr=K50_errs, fmt='o', capsize=3, markersize=7, label=label, color=rhocols[i])

        mean_K50_rho = np.mean(d['K50s'])
        rel_diff = (d['K50s'] - mean_K50_rho) / mean_K50_rho
        if rho>-0.2: ax_errs.errorbar(Cs + offsets[i-1], rel_diff, yerr=K50_errs, fmt='o', color=rhocols[i], capsize=2, markersize=5, alpha=0.8)
        ax_K50s.axhline(mean_K50_rho, xmin=0, xmax=1, color=rhocols[i], linestyle = '--', linewidth =1)


    #ax_K50s.set_xlabel(r'Connectance $C$', labelpad=20)
    ax_K50s.set_ylabel(r'stability threshold ($K_{50}$)')
    ax_K50s.legend(
        title=rf'$S$={high_S}, $\Delta$={high_delta} $S$={high_S}',
        title_fontproperties={'size':12}, 
        ncol=2,
        fancybox=False, 
        framealpha=0.3, 
        edgecolor='white',
        loc='upper left')
    ax_K50s.grid(True, alpha=0.3)
    ax_K50s.axs[-1].tick_params(labelbottom=False, bottom=False)
    ax_K50s.axs[0].text(-0.15, 1.05, '(a)', transform=ax_K50s.axs[0].transAxes, fontsize=16, va='bottom', ha='right')

    ax_errs.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax_errs.set_xlabel('Connectance $C$', fontsize=12)
    ax_errs.set_ylabel('Rel. Diff.\nfrom Mean', fontsize=10)
    ax_errs.grid(True, alpha=0.3, linestyle=':')
    
    ax_errs.set_xlim(ax_K50s.axs[-1].get_xlim())


    # right side: heatmap 

    C_grid1, K_grid1 = np.meshgrid(data_frac['Cs'], data_frac['Ks'], indexing='ij')
    c1 = ax_heat.pcolormesh(
        C_grid1, K_grid1, data_frac['stable_fracs'], 
        cmap='RdBu', vmin=0.0, vmax=1.0, shading='nearest'
    )
    
    cbar1 = fig.colorbar(c1, ax=ax_heat)
    cbar1.set_label('Fraction of Stable Runs', rotation=270, labelpad=20)
    ax_heat.set_xlabel('Connectance $C$', fontsize=12)
    ax_heat.set_ylabel('Complexity $K$', fontsize=12)
    ax_heat.legend(loc='upper right')
    ax_heat.text(-0.1, 1.02, '(b)', transform=ax_heat.transAxes, fontsize=16, va='bottom', ha='right')


    
    plt.tight_layout()
#    plt.savefig('figures/Heatmap_Comparison.pdf', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()