import numpy as np
import matplotlib.pyplot as plt
import os






def main():
    rhos = [0]
    rhocols = ['tab:blue', 'tab:green', 'tab:orange', 'tab:red']
    high_delta = 1000
    high_S = 100

    # load data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_KofC = {}
    k50S_dir = os.path.join(current_dir, '..', '..', 'data', 'C_data', 'K50_ofC', f'S={high_S}')

    data_frac = np.load("data/C_data/CK_heatmap_S100_rho0.0_200rpk.npz")
    
    for irho, rho in enumerate(rhos):
        filepath = os.path.join(k50S_dir, f'K50byC_rho{rho:.2f}_C0.1-1.0_200rpk.npz')
        with np.load(filepath) as data: 
            data_KofC[rho] = {'K50s': data['K50s'], 'K50_errs': data['K50_errs'], 'Cs': data['Cs']}

    
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    
    # panel a
    ax = axs[0]
    for i, rho in enumerate(rhos):
        d = data_KofC[rho]
        Cs = d['Cs']
        K50s = d['K50s']
        K50_errs = d['K50_errs']
        print(f'rho={rho}')
        ax.errorbar(Cs, K50s, yerr=d['K50_errs'], fmt='o:', capsize=3, markersize=7, label=(fr'$\rho = -⅓$') if rho==-1./3. else fr'$\rho = {rho}$', color=rhocols[i])
        ax.set_xlabel(r'Connectance $C$', labelpad=20)
    ax.set_ylabel(r'stability threshold ($K_{50}$)')
    ax.legend(
        title=rf'$\Delta$={high_delta} \n$S$={high_S}',
        title_fontproperties={'size':12}, 
        ncol=2,
        fancybox=False, 
        framealpha=0.3, 
        edgecolor='white')
    ax.grid(True, alpha=0.3)


    # panel b 
    ax = axs[1]
    C_grid1, K_grid1 = np.meshgrid(data_frac['Cs'], data_frac['Ks'], indexing='ij')
    c1 = ax.pcolormesh(
        C_grid1, K_grid1, data_frac['stable_fracs'], 
        cmap='RdBu', vmin=0.0, vmax=1.0, shading='nearest'
    )
    
    rho_val = data_frac['rho']
    K_theory = 2 * (1 + 3 * rho_val)**(-0.5)
    ax.axhline(K_theory, color='black', linestyle='--', linewidth=2, label=fr'Theory Limit ($\rho={rho_val}$)')
    
    cbar1 = fig.colorbar(c1, ax=ax)
    cbar1.set_label('Fraction of Stable Runs', rotation=270, labelpad=20)
    ax.set_xlabel('Connectance $C$', fontsize=12)
    ax.set_ylabel('Complexity $K$', fontsize=12)
    ax.set_title('Stability Fraction', fontsize=14)
    ax.legend(loc='upper right')
    ax.text(-0.1, 1.05, '(b)', transform=ax.transAxes, fontsize=16, va='bottom', ha='right')
    
    
    
    plt.tight_layout()
    plt.savefig('figures/Heatmap_Comparison.pdf', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()