import numpy as np
import lvlh_functions as lvf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os 
from brokenaxes import brokenaxes

def onerun_eigs_rho(rho, S, C, Bseed, delta):
    ''' 
    get the eigenvalues for 1 run with C1, Brho; to then plot
    '''
    K = 2 * (1+3*rho)**(-0.5)
    sigma = K* (S*C)**(-0.5)      #C=1 here 
    B = lvf.B_rho(S, C, sigma, Bseed, L=2, rho=0)
    R = lvf.R_star_2stage_delta(B, delta)
    Jac = lvf.Jacobian(B, R)
    Jeigs = np.linalg.eigvals(Jac)
    return Jeigs 

def onerun_eigs_Bind(K, S, C, Bseed, delta):
    ''' 
    get the eigenvalues for 1 run with C1, Bind; to then plot
    '''
    sigma = K* (S*C)**(-0.5)      #C=1 here 
    B = lvf.B_ind(S, C, sigma, L=2, seed=Bseed)
    R = lvf.R_star_2stage_delta(B, delta)
    Jac = lvf.Jacobian(B, R)
    Jeigs = np.linalg.eigvals(Jac)
    return Jeigs 

def get_Scurve_data(dir, filename):
    data = np.load(os.path.join(dir, filename))
    Ks, fracs, frac_errs = lvf.pad_S_curve(data['Ks'], data['fracs'], data['frac_errs'], 0, 3)      # continue curve across whole region of K
    K50 = data['K50']
    K50_err = data['K50_err']
    return Ks, fracs, frac_errs, K50, K50_err
        
def S_curve_panel(ax, Ks_onestage, fracs_onestage, frac_errs_onestage, K50_onestage, K50_err_onestage,
                Ks_list, fracs_list, frac_errs_list, K50_list, K50_err_list, 
                K50_threshold, xlims, legend_title, first_panel=False):
    S_cols = ['lightskyblue', 'dodgerblue', 'blue']
    labels = ['S=1000', 'S=100', 'S=25']
    Scol1 = 'grey'
    ax.fill_between(Ks_onestage, fracs_onestage-frac_errs_onestage, fracs_onestage+frac_errs_onestage, alpha=0.3, color=Scol1)
    ax.plot(Ks_onestage, fracs_onestage, '-', alpha=0.5, lw = 5, color=Scol1, label='one stage')
    ax.vlines(x=K50_threshold, ymin=0, ymax=1, color='black', linestyle='--', lw=1.5, label = r'Large S, $\Delta$ limit')
    ax.errorbar(K50_onestage, 0.5, xerr=K50_err_onestage, fmt='x', alpha=0.7, ms = 8, color=Scol1)

    nlist = len(Ks_list)
    for i in range(nlist):
        Ks = Ks_list[i]; fracs=fracs_list[i]; frac_errs=frac_errs_list[i]; K50 = K50_list[i]; K50_err = K50_err_list[i]
        ax.plot(Ks, fracs, '-', alpha=1, lw = 1, color=S_cols[i], label=labels[i])
        ax.fill_between(Ks, fracs-frac_errs, fracs+frac_errs, alpha=0.3, color=S_cols[i])
        ax.errorbar(K50, 0.5, xerr=K50_err, fmt='x', alpha=1, ms = 10, color=S_cols[i])

    ax.set_xlim(xlims)
    ax.set_ylim(-0.01, 1.01)
    ax.set_xlabel(r'Complexity $K$', fontsize=10)
    ax.minorticks_on()
    ax.grid(True, which='major', linestyle='--', alpha=0.7)
    ax.grid(True, which='minor', linestyle='--', alpha=0.3)
    

    handles, labels = ax.get_legend_handles_labels()
    #order = [1, 2, 3, 4, 0, 5] 
    if first_panel:
        leg = ax.legend(
            #[handles[idx] for idx in order], 
            #[labels[idx] for idx in order], 
            handles, 
            labels, 
            title=legend_title, 
            title_fontproperties={'size':11}, 
            loc='upper right', 
            fancybox=False, 
            framealpha=0.3, 
            edgecolor='white'
        )
        leg.get_texts()[-1].set_fontsize(8)
        ax.set_ylabel(r'fraction of runs stable for given $K$', fontsize=10)    
    else:
        leg = ax.legend(
            #[handles[idx] for idx in order], 
            #[labels[idx] for idx in order], 
            [],
            [],
            title=legend_title, 
            title_fontproperties={'size':11}, 
            loc='upper right', 
            fancybox=False, 
            framealpha=0.3, 
            edgecolor='white'
                )

    leg.get_frame().set_linewidth(0.5)
    
    
    #else:
    #    leg = ax.legend(title=legend_title, title_fontproperties={'size':11}, loc='upper right', fancybox=False, framealpha=0.3, edgecolor='white')
    #    leg.get_frame().set_linewidth(0.5)
    
def eig_panel(ax, Jeigs, text, ylim=(-2.1, 2.1), first_panel=False):
    box_props = dict(boxstyle='square', facecolor='white', alpha=0.6, edgecolor='None')
    colors = np.where(np.real(Jeigs) > 0, 'red', 'green')
    ax.axhline(0, color='black', linewidth=1, linestyle='--')
    ax.axvline(0, color='black', linewidth=1, linestyle='--')
    ax.scatter(np.real(Jeigs), np.imag(Jeigs), c=colors, s=3)
    ax.set_xlabel(r'Re($\lambda_{J}$)', labelpad=20)
    if ylim is not None:
        ax.set_ylim(ylim)
    target_ax = ax.axs[0]
        
    break_left = ax.axs[0].get_xlim()[1]
    break_right = ax.axs[1].get_xlim()[0]
    
    ax.axs[0].text(break_left, 0, '/', fontsize=14, weight= 300, ha='center', va='center', zorder=10)
    ax.axs[1].text(break_right, 0, '/', fontsize=14, ha='center', va='center', zorder=10)
    #ax.text(0.05, 0.97, text, transform=ax.transAxes, fontsize=9, verticalalignment='top', bbox=box_props)
        
    target_ax.text(0.05, 0.97, text, transform=target_ax.transAxes, fontsize=9, verticalalignment='top', bbox=box_props)

    if first_panel:
        ax.set_ylabel(r'Im($\lambda_{J}$)')





def main():

    # set parameters for eigenvalue plots 
    S = 200 # change later 
    C = 1
    delta = 1000

    # eigenvalues for rho=1, rho=0
    K1 = 2 * (1+3*1)**(-0.5)
    K0 = 2 * (1)**(-0.5)
    Kind = 1

    Jeigs_1 = onerun_eigs_rho(1, S, C, 1, delta)
    Jeigs_0 = onerun_eigs_rho(0, S, C, 2, delta)

    # eigenvalues for Bind case 
    Jeigs_ind = onerun_eigs_Bind(1, S, C, 4, delta)

    # load in S curve data 
    current_dir = os.path.dirname(os.path.abspath(__file__))
    rho_dir = os.path.join(current_dir, '..', '..', 'data', 'S_curves', 'B_rho')
    ind_dir = os.path.join(current_dir, '..', '..', 'data', 'S_curves', 'B_ind')
    onestage_dir = os.path.join(current_dir, '..', '..', 'data', 'S_curves', 'one_stage')

    Ks_onestage, fracs_onestage, errs_onestage, K50_onestage, K50_err_onestage = get_Scurve_data(onestage_dir, 'one_stage_S1000_500rpk.npz')

    rho1_files = ['B_rho1.00_S25_500rpk.npz', 'B_rho1.00_S100_500rpk.npz', 'B_rho1.00_S1000_500rpk.npz']
    Ks_list_rho1, fracs_list_rho1, errs_list_rho1, K50_list_rho1, K50_err_list_rho1 = zip(*[get_Scurve_data(rho_dir, f) for f in rho1_files])
    rho0_files = ['B_rho0.00_S25_500rpk.npz', 'B_rho0.00_S100_500rpk.npz'] #, 'B_rho0.00_S1000_500rpk.npz']
    Ks_list_rho0, fracs_list_rho0, errs_list_rho0, K50_list_rho0, K50_err_list_rho0 = zip(*[get_Scurve_data(rho_dir, f) for f in rho0_files])
    Bind_files = ['B_ind_S25_500rpk.npz', 'B_ind_S100_500rpk.npz', 'B_ind_S1000_500rpk.npz']
    Ks_list_Bind, fracs_list_Bind, errs_list_Bind, K50_list_Bind, K50_err_list_Bind = zip(*[get_Scurve_data(ind_dir, f) for f in Bind_files])
    




    # set up plots
    box_props = dict(boxstyle='square', facecolor='white', alpha=0, edgecolor='None')
    rho1text = r'2 stages, $B_{\rho}$, $\rho=1$'+f' \nS = {S}, C = {C}, K = {K1:.1f}'
    rho0text = r'2 stages, $B_{\rho}$, $\rho=0$'+f' \nS = {S}, C = {C}, K = {K0:.1f}'
    indtext = r'2 stages, $B_{ind}$'+f' \nS = {S}, C = {C}, K = {Kind:.1f}'
    eigcol = 'olivedrab'
    Scols = ['lightskyblue', 'dodgerblue', 'blue']
    Scol1 = 'grey'
    axfontsize = 10

    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 3, height_ratios=[6,2.5], width_ratios=[2, 2, 2], hspace=0.3, wspace=0.3)
    s1 = fig.add_subplot(gs[0, 0])
    s0 = fig.add_subplot(gs[0, 1])
    sind = fig.add_subplot(gs[0, 2])

    # for broken axis: 
    xlims_broken1 = ((-2010, -1998), (-6, 2))
    xlims_broken0 = ((-2015, -1998), (-6, 2))
    xlims_brokenind = ((-2010, -1998), (-6, 2))
    eig1 = brokenaxes(xlims=xlims_broken1, subplot_spec=gs[1, 0], fig=fig, wspace=0.1, d=0.0045, tilt=70)
    eig0 = brokenaxes(xlims=xlims_broken0, subplot_spec=gs[1, 1], fig=fig,  wspace=0.1, d=0.0045, tilt=70)
    eigind = brokenaxes(xlims=xlims_brokenind, subplot_spec=gs[1, 2], fig=fig,  wspace=0.1, d=0.0045, tilt=70)


    # S curve plots 
    
    # panel 1: 
    S_curve_panel(s1, Ks_onestage, fracs_onestage, errs_onestage, K50_onestage, K50_err_onestage, Ks_list_rho1, fracs_list_rho1, errs_list_rho1, K50_list_rho1, K50_err_list_rho1, 
                  K1, (0.7, 1.5), r'2 stages, $\rho=1$', first_panel=True)
    S_curve_panel(s0, Ks_onestage, fracs_onestage, errs_onestage, K50_onestage, K50_err_onestage, Ks_list_rho0, fracs_list_rho0, errs_list_rho0, K50_list_rho0, K50_err_list_rho0,
                  K0, (0.7, 3), r'2 stages, $\rho=0$')
    S_curve_panel(sind, Ks_onestage, fracs_onestage, errs_onestage, K50_onestage, K50_err_onestage, Ks_list_Bind, fracs_list_Bind, errs_list_Bind, K50_list_Bind, K50_err_list_Bind,
                  K1, (0.7, 1.5), r'2 stages, $B_{ind}$')

    # eigenvalue plots 
    e1text = r'2 stages, $B_{\rho}$, $\rho=1$'+f' \nS = {S}, C = {C}, K = {K1:.1f}'
    e0text = r'2 stages, $B_{\rho}$, $\rho=0$'+f' \nS = {S}, C = {C}, K = {K0:.1f}'
    eindtext = r'2 stages, $B_{ind}$'+f' \nS = {S}, C = {C}, K = {Kind:.3f}'

    


    eig_panel(eig1, Jeigs_1, e1text, (-2.1, 2.1), first_panel=True)
    eig_panel(eig0, Jeigs_0, e0text)
    eig_panel(eigind, Jeigs_ind, eindtext)
    #eig_panel(eig0, Jeigs_0, e0text, (-20, 1))
    #eig_panel(eigind, Jeigs_ind, eindtext, (-20, 1))

    #plt.tight_layout()
    plt.savefig(os.path.join(current_dir, '..', '..', 'figures', 'Scurves.pdf'), dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()



