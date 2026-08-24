import numpy as np
import lvlh_functions as lvf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
import os 
from brokenaxes import brokenaxes


def K50_asym_fit(x, y0, A, B):
    return y0 + (A-y0)*x/(B+x)

def main():
    rhos = [-1./3., 0, 0.5, 1]
    rhocols = ['tab:blue', 'tab:green', 'tab:orange', 'tab:red']
    high_delta = 1000
    high_S = 1000
    current_dir = os.path.dirname(os.path.abspath(__file__))

    K50maxbar = 10
    
    # get data for k50 vs S 
    data_KofS = {}
    k50S_dir = os.path.join(current_dir, '..', '..', 'data', 'K50_data', 'K50_ofS', f'delta={high_delta}')
    for irho, rho in enumerate(rhos):
        filepath = os.path.join(k50S_dir, f'K50byS_rho{rho:.2f}_S10-1000_200rpk.npz')
        with np.load(filepath) as data: 
            data_KofS[rho] = {'K50s': data['K50s'], 'K50_errs': data['K50_errs'], 'Ss': data['Ss']}

    filepath_ind = os.path.join(k50S_dir, f'K50byS_Bind_S10-1000_100rpk.npz')
    with np.load(filepath_ind) as data: 
        data_KofS['ind'] = {'K50s': data['K50s'], 'K50_errs': data['K50_errs'], 'Ss': data['Ss']}

    # get data for k50 vs delta 
    data_Kofdelta = {}
    k50delta_dir = os.path.join(current_dir, '..', '..', 'data', 'K50_data', 'K50_ofdelta', f'S={high_S}')
    for irho, rho in enumerate(rhos):
            filepath = os.path.join(k50delta_dir, f'K50bydelta_rho{rho:.2f}_delta0.01-1000.0_100rpk.npz')
            with np.load(filepath) as data: 
                data_Kofdelta[rho] = {'K50s': data['K50s'], 'K50_errs': data['K50_errs'], 'deltas': data['deltas']}

    filepath_ind = os.path.join(k50delta_dir, f'K50bydelta_Bind_delta0.01-1000.0_100rpk.npz')
    with np.load(filepath_ind) as data: 
        data_Kofdelta['ind'] = {'K50s': data['K50s'], 'K50_errs': data['K50_errs'], 'deltas': data['deltas']}

    # get data for rho vs delta map
    rhodelta_map_dir = os.path.join(current_dir, '..', '..', 'data', 'K50_data', 'K50_deltarhomap')
    rhodelta_file = os.path.join(rhodelta_map_dir, 'S500_100rpk.npz')
    data_map = np.load(rhodelta_file)
    rd_rhos, rd_deltas, rd_K50s, rd_K50errs = data_map['rhos'], data_map['deltas'], data_map['K50s'], data_map['K50_errs']
    K50_maxmeasured = np.max(rd_K50s)
    print(f'max k50 here is {K50_maxmeasured}')
    K50maxplot = min(K50maxbar, K50_maxmeasured)


    # make plots here: 

    # plot setup 
    fig = plt.figure(figsize=(18, 9))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], hspace=0.35)

    y_break = ((0, 5), (45, 55))
    ax1 = brokenaxes(ylims=y_break, subplot_spec=gs[0, 0], fig=fig, hspace=0.2, d=0.0045, tilt=20, despine=False)
    #ax1 = fig.add_subplot(gs[0, 0])  
    ax2 = fig.add_subplot(gs[1, 0])  
    ax3 = fig.add_subplot(gs[:, 1])


    # S vs K50
    for i, rho in enumerate(rhos):
        d = data_KofS[rho]
        Ss = d['Ss']
        K50s = d['K50s']
        K50_errs = d['K50_errs']
        # fit it real quick
        print(f'rho={rho}')
        ax1.errorbar(d['Ss'], d['K50s'], yerr=d['K50_errs'], fmt='o:', capsize=3, markersize=7, label=(fr'$\rho = -⅓$') if rho==-1./3. else fr'$\rho = {rho}$', color=rhocols[i])
    di = data_KofS['ind']
    ax1.errorbar(di['Ss'], di['K50s'], yerr=di['K50_errs'], fmt='x:', markerfacecolor='None', capsize=5, markersize=6, label=r'$B_{ind}$', color='black')
    #ax1.set_title(fr'$K_{{50}}$ vs number of species S ($\delta = {high_delta}$)', fontsize=16)
    ax1.set_xlabel('number of species (S)', labelpad=20)
    ax1.set_ylabel(r'stability threshold ($K_{50}$)')
    ax1.legend(
        title=r'$\Delta$=1000',
        title_fontproperties={'size':12}, 
        ncol=2,
        fancybox=False, 
        framealpha=0.3, 
        edgecolor='white')
    ax1.set_xlim((0, 1050))
    ax1.grid(True, alpha=0.3)

    # delta vs K50
    for i, rho in enumerate(rhos):
        d = data_Kofdelta[rho]
        deltas = d['deltas']
        K50s = d['K50s']
        K50_errs = d['K50_errs']
        print(deltas, K50s)
        ax2.errorbar(deltas, K50s, yerr=K50_errs, fmt='o:', capsize=3, markersize=7, label=(fr'$\rho = -⅓$') if rho==-1./3. else fr'$\rho = {rho}$', color=rhocols[i])
    di = data_Kofdelta['ind']
    ax2.errorbar(di['deltas'], di['K50s'], yerr=di['K50_errs'], fmt='x:', markerfacecolor='None', capsize=3, markersize=5, label=r'$B_{ind}$', color='black')
    ax2.set_xlabel(r'$\Delta$')
    ax2.set_ylabel(r'stability threshold ($K_{50}$)', labelpad=20)
    ax2.legend(
            title=r'$S$=1000',
            title_fontproperties={'size':12}, 
            ncol=2,
            fancybox=False, 
            framealpha=0.3, 
            edgecolor='white')
    ax2.set_xlim((-0.1, 1300))
    ax2.set_ylim((0, 55))
    ax2.set_xscale('symlog')
    ax2.set_yscale('symlog')
    ax2.grid(True, alpha=0.3)



    # rho/delta map 
    x_indices = np.arange(len(rd_deltas))
    cmap = plt.get_cmap('rainbow').with_extremes(over='purple')
    mesh = ax3.pcolormesh(x_indices, rd_rhos, rd_K50s, cmap=cmap, shading='auto', norm=LogNorm(vmax=7))
    #ax3.set_title(r'$K_{50}$ for $\rho$, $\Delta$ ($S = 25$)', fontsize=16)
    ax3.set_xlabel(r'$\Delta$')
    ax3.set_ylabel(r'correlation $\rho$')
    #ax3.set_xscale('symlog')

    ax3.set_yticks(rd_rhos[::2])
    ax3.set_yticklabels([f'{r:.2f}' for r in rd_rhos[::2]])
    ax3.set_xticks(x_indices[::3])
    ax3.set_xticklabels([f'{d:g}' for d in rd_deltas[::3]])
    
    '''if K50_maxmeasured>K50maxbar: cbar = fig.colorbar(mesh, ax=ax3, extend='max')
    else: cbar = fig.colorbar(mesh, ax=ax3)
    cbar.set_label(r'stability threshold ($K_{50}$)', fontsize=14)'''

    cbar = fig.colorbar(mesh, ax=ax3, extend='max', extendrect=True)
    cbar.set_label(r'stability threshold ($K_{50}$)', fontsize=12, labelpad=20)
    ticks=[0.5, 1, 2, 3, 4, 5, 6]
    #ticks = np.arange(1, 7) 
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f'{t}' for t in ticks])
    #tick_labels = [f'{t}' for t in ticks]
    #tick_labels[-1] = ''
    cbar.ax.plot(
        [1.0, 1.15], [1.025, 1.025],  
        color='black', 
        linewidth=0.8,                
        transform=cbar.ax.transAxes, 
        clip_on=False                 
    )
    cbar.ax.text(
        2.2, 1.025,      
        r'$K_{50}>$20', 
        transform=cbar.ax.transAxes, 
        color='black',     
        ha='center', 
        va='center'
    )
    text_box_style = dict(boxstyle='square,pad=0.5', facecolor='white', alpha=0.5, edgecolor='none')
    ax3.text(0.95, 0.97, r'$K_{50}$ for $S=$500 species'+'\n'+ r'$100$ $B_{\rho}$ realizations per point',
                transform=ax3.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right', bbox=text_box_style
    )

    # label panels here 
    axes = [ax1.axs[0], ax2, ax3]
    labels = ['(a)', '(b)', '(c)']
    for ax, label in zip(axes, labels):
        ax.text(-0.05, 1.0, label, transform=ax.transAxes, fontsize=16, va='bottom', ha='right')


    print(f'k50 highs: {rd_K50s[0, -2]}')

    #plt.tight_layout()

    plt.savefig('figures/delta_fig.pdf', dpi=300, bbox_inches='tight')


    plt.show()





if __name__ == "__main__": main()