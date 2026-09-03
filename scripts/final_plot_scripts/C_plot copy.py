import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

def main():
    rhos = [-1./3., 0, 0.5, 1]
    rhocols = ['tab:blue', 'tab:green', 'tab:orange', 'tab:red']
    high_delta = 1000
    high_S = 500

    # load data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_KofC = {}
    
    # NOTE: Ensure these paths match your environment
    k50S_dir = os.path.join(current_dir, '..', '..', 'data', 'C_data', 'K50_ofC', f'S={high_S}')
    data_frac = np.load("data/C_data/CK_heatmap_S100_rho0.0_200rpk.npz")
    
    for irho, rho in enumerate(rhos):
        filepath = os.path.join(k50S_dir, f'K50byC_rho{rho:.2f}_C0.1-1.0_200rpk.npz')
        with np.load(filepath) as data: 
            data_KofC[rho] = {'K50s': data['K50s'], 'K50_errs': data['K50_errs'], 'Cs': data['Cs']}

    # Calculate Data for the Bottom Strip (Relative difference from the mean for each C)
    Cs = data_KofC[rhos[0]]['Cs']
    n_C = len(Cs)
    K50_matrix = np.zeros((len(rhos), n_C))
    err_matrix = np.zeros((len(rhos), n_C))
    
    for i, rho in enumerate(rhos):
        K50_matrix[i, :] = data_KofC[rho]['K50s']
        err_matrix[i, :] = data_KofC[rho]['K50_errs']

    # Mean across all rhos for each distinct C
    mean_K50_per_C = np.mean(K50_matrix, axis=0)
    
    # ---------------------------------------------------------
    # Layout Setup using Nested GridSpec
    # ---------------------------------------------------------
    fig = plt.figure(figsize=(15, 7))
    
    # Master grid: 1 row, 2 columns
    gs_master = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.25)
    
    # Left side (Panel a) grid: 3 rows (Top break, Bottom break, Bottom strip)
    # Height ratios control how tall each section is relative to the others
    gs_left = gridspec.GridSpecFromSubplotSpec(
        3, 1, subplot_spec=gs_master[0], 
        height_ratios=[1.5, 3, 2], hspace=0.15
    )
    
    ax_top = fig.add_subplot(gs_left[0])
    ax_bot = fig.add_subplot(gs_left[1], sharex=ax_top)
    ax_strip = fig.add_subplot(gs_left[2], sharex=ax_top)
    
    # Right side (Panel b)
    ax_heat = fig.add_subplot(gs_master[1])

    # ---------------------------------------------------------
    # Panel (a): Top and Bottom Broken Axes
    # ---------------------------------------------------------
    for i, rho in enumerate(rhos):
        d = data_KofC[rho]
        label_str = r'$\rho = -⅓$' if rho == -1./3. else fr'$\rho = {rho}$'
        
        # Plot on both top and bottom axes
        ax_top.errorbar(d['Cs'], d['K50s'], yerr=d['K50_errs'], fmt='o:', capsize=3, markersize=7, color=rhocols[i])
        ax_bot.errorbar(d['Cs'], d['K50s'], yerr=d['K50_errs'], fmt='o:', capsize=3, markersize=7, label=label_str, color=rhocols[i])

    # Set the limits for the break (Adjust these slightly if you need more padding)
    ax_top.set_ylim(46, 55)
    ax_bot.set_ylim(0, 3)

    # Hide the spines between ax_top and ax_bot to create the "break" effect
    ax_top.spines['bottom'].set_visible(False)
    ax_bot.spines['top'].set_visible(False)
    ax_top.tick_params(labelbottom=False, bottom=False)

    # Draw the diagonal slash marks for the break
    d_slash = 0.015
    kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False, lw=1.5)
    ax_top.plot((-d_slash, +d_slash), (-d_slash, +d_slash), **kwargs)
    ax_top.plot((1 - d_slash, 1 + d_slash), (-d_slash, +d_slash), **kwargs)
    kwargs.update(transform=ax_bot.transAxes)
    ax_bot.plot((-d_slash, +d_slash), (1 - d_slash, 1 + d_slash), **kwargs)
    ax_bot.plot((1 - d_slash, 1 + d_slash), (1 - d_slash, 1 + d_slash), **kwargs)

    # Formatting for Panel (a) main plot
    ax_bot.set_ylabel(r'Stability threshold ($K_{50}$)', y=1.0, labelpad=15)
    ax_bot.legend(
        title=rf'$\Delta$={high_delta}'+'\n'+rf'$S$={high_S}',
        title_fontproperties={'size':12}, 
        ncol=2, fancybox=False, framealpha=0.3, edgecolor='white'
    )
    ax_top.grid(True, alpha=0.3)
    ax_bot.grid(True, alpha=0.3)
    
    # ---------------------------------------------------------
    # Panel (a) Bottom Strip: Relative Difference Staggered
    # ---------------------------------------------------------
    # Create tiny horizontal offsets to prevent error bars from overlapping
    offsets = np.linspace(-0.02, 0.02, len(rhos))
    
    for i, rho in enumerate(rhos):
        rel_diff = (K50_matrix[i] - mean_K50_per_C) / mean_K50_per_C
        # Approximate propagated error relative to the constant mean reference
        rel_err = err_matrix[i] / mean_K50_per_C 
        
        # Apply the stagger offset to the Cs array
        ax_strip.errorbar(Cs + offsets[i], rel_diff, yerr=rel_err, 
                          fmt='o', color=rhocols[i], capsize=2, markersize=5, alpha=0.8)

    ax_strip.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax_strip.set_xlabel(r'Connectance $C$', labelpad=10)
    ax_strip.set_ylabel('Rel. Diff.\nfrom Mean', fontsize=10)
    ax_strip.grid(True, alpha=0.3, linestyle=':')
    ax_top.text(-0.15, 1.05, '(a)', transform=ax_top.transAxes, fontsize=16, va='bottom', ha='right')

    # ---------------------------------------------------------
    # Panel (b): Heatmap
    # ---------------------------------------------------------
    C_grid1, K_grid1 = np.meshgrid(data_frac['Cs'], data_frac['Ks'], indexing='ij')
    c1 = ax_heat.pcolormesh(
        C_grid1, K_grid1, data_frac['stable_fracs'], 
        cmap='RdBu', vmin=0.0, vmax=1.0, shading='nearest'
    )
    
    rho_val = data_frac['rho']
    K_theory = 2 * (1 + 3 * rho_val)**(-0.5)
    cbar1 = fig.colorbar(c1, ax=ax_heat)
    cbar1.set_label('Fraction of Stable Runs', rotation=270, labelpad=20)
    ax_heat.set_xlabel('Connectance $C$', fontsize=12)
    ax_heat.set_ylabel('Complexity $K$', fontsize=12)
    ax_heat.legend(loc='upper right')
    ax_heat.text(-0.1, 1.05, '(b)', transform=ax_heat.transAxes, fontsize=16, va='bottom', ha='right')
    
    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/Heatmap_Comparison.pdf', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()