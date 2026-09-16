import numpy as np
import lvlh_functions as lvf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os 
from brokenaxes import brokenaxes

def onerun_eigs_rho_3(rho, S, C, Bseed, delta):
    ''' 
    get the eigenvalues for 1 run with C1, Brho; to then plot
    '''
    K = 2 * (1+3*rho)**(-0.5)
    sigma = K* (S*C)**(-0.5)      #C=1 here 
    L = 3
    B = lvf.B_rho(S, C, sigma, Bseed, L, rho)
    R = lvf.R_star_3stage_delta(B, delta)
    Jac = lvf.Jacobian(B, R)
    Jeigs = np.linalg.eigvals(Jac)
    return Jeigs 

def onerun_eigs_Bind_3(K, S, C, Bseed, delta):
    ''' 
    get the eigenvalues for 1 run with C1, Bind; to then plot
    '''
    sigma = K* (S*C)**(-0.5)      #C=1 here 
    B = lvf.B_ind(S, C, sigma, L=3, seed=Bseed)
    R = lvf.R_star_3stage_delta(B, delta)
    Jac = lvf.Jacobian(B, R)
    Jeigs = np.linalg.eigvals(Jac)
    return Jeigs 

def onerun_eigs_rho_4(rho, S, C, Bseed, delta):
    ''' 
    get the eigenvalues for 1 run with C1, Brho; to then plot
    '''
    K = 2 * (1+3*rho)**(-0.5)
    sigma = K* (S*C)**(-0.5)      #C=1 here 
    L = 4
    B = lvf.B_rho(S, C, sigma, Bseed, L, rho)
    R = lvf.R_star_3stage_delta(B, delta)
    Jac = lvf.Jacobian(B, R)
    Jeigs = np.linalg.eigvals(Jac)
    return Jeigs 

def onerun_eigs_Bind_4(K, S, C, Bseed, delta):
    ''' 
    get the eigenvalues for 1 run with C1, Bind; to then plot
    '''
    sigma = K* (S*C)**(-0.5)      #C=1 here 
    B = lvf.B_ind(S, C, sigma, L=4, seed=Bseed)
    R = lvf.R_star_3stage_delta(B, delta)
    Jac = lvf.Jacobian(B, R)
    Jeigs = np.linalg.eigvals(Jac)
    return Jeigs 

def get_Scurve_data(dir, filename):
    data = np.load(os.path.join(dir, filename))
    Ks, fracs, frac_errs = lvf.pad_S_curve(data['Ks'], data['fracs'], data['frac_errs'], 0, 6)      # continue curve across whole region of K
    K50 = data['K50']
    K50_err = data['K50_err']
    return Ks, fracs, frac_errs, K50, K50_err
        
def S_curve_panel(ax, Ks_onestage, fracs_onestage, frac_errs_onestage, K50_onestage, K50_err_onestage,
                Ks_list, fracs_list, frac_errs_list, K50_list, K50_err_list, 
                K50_threshold, xlims, legend_title, first_panel=False):
    S_cols = ['lightskyblue', 'dodgerblue', 'blue']
    labels = [ 'S=25', 'S=100','S=1000']
    Scol1 = 'grey'
    ax.fill_between(Ks_onestage, fracs_onestage-frac_errs_onestage, fracs_onestage+frac_errs_onestage, alpha=0.3, color=Scol1)
    ax.plot(Ks_onestage, fracs_onestage, '-', alpha=0.5, lw = 5, color=Scol1, label='one stage')
    ax.vlines(x=K50_threshold, ymin=0, ymax=1, color='black', linestyle='--', lw=1.5, label = r'Large S, $\Delta$ limit')
    ax.errorbar(K50_onestage, 0.5, xerr=K50_err_onestage, fmt='x', alpha=0.7, ms = 8, color=Scol1)
    ax.plot([], [], 'x', color='black', ms=8, label='50% of \nruns stable')
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
    order = [0, 1, 5, 4, 3, 2]
    if first_panel:
        leg = ax.legend(
            [handles[idx] for idx in order], 
            [labels[idx] for idx in order], 
            #handles, 
            #labels, 
            title=legend_title, 
            title_fontproperties={'size':11}, 
            fontsize=9,
            loc='upper right', 
            fancybox=False, 
            framealpha=0.3, 
            edgecolor='white'
        )
        leg.get_texts()[-1].set_fontsize(8)
        ax.set_ylabel(r'fraction of runs stable for given $K$', fontsize=10)    
    else:
        leg = ax.legend(
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
    K1 = 1
    K0_3 = 3
    K0_4 = 4
    Kind = 1

    Jeigs_1_3 = onerun_eigs_rho_3(K1, S, C, 1, delta)
    Jeigs_0_3 = onerun_eigs_rho_3(K0_3, S, C, 2, delta)
    Jeigs_1_4 = onerun_eigs_rho_4(K1, S, C, 1, delta)
    Jeigs_0_4 = onerun_eigs_rho_4(K0_4, S, C, 2, delta)

    # eigenvalues for Bind case 
    Jeigs_ind_3 = onerun_eigs_Bind_3(Kind, S, C, 4, delta)
    Jeigs_ind_4 = onerun_eigs_Bind_4(Kind, S, C, 4, delta)

    # load in S curve data: 3 stage 
    current_dir = os.path.dirname(os.path.abspath(__file__))
    rho_dir_3 = os.path.join(current_dir, '..', '..', 'data', 'S_curves', '3stage', 'B_rho')
    ind_dir_3 = os.path.join(current_dir, '..', '..', 'data', 'S_curves', '3stage', 'B_ind')
    rho_dir_4 = os.path.join(current_dir, '..', '..', 'data', 'S_curves', '4stage', 'B_rho')
    ind_dir_4 = os.path.join(current_dir, '..', '..', 'data', 'S_curves', '4stage', 'B_ind')
        
    onestage_dir = os.path.join(current_dir, '..', '..', 'data', 'S_curves', 'one_stage')

    Ks_onestage, fracs_onestage, errs_onestage, K50_onestage, K50_err_onestage = get_Scurve_data(onestage_dir, 'one_stage_S1000_500rpk.npz')

    rho1_files_3 = ['B_L3_rho_1.00_S25_500rpk.npz', 'B_L3_rho_1.00_S100_500rpk.npz', 'B_L3_rho_1.00_S100_500rpk.npz']
    rho0_files_3 = ['B_L3_rho_0.00_S25_500rpk.npz', 'B_L3_rho_0.00_S100_500rpk.npz', 'B_L3_rho_0.00_S100_500rpk.npz']
    Bind_files_3 = ['B_L3_ind_S25_500rpk.npz', 'B_L3_ind_S100_500rpk.npz', 'B_L3_ind_S1000_200rpk.npz']

    rho1_files_4 = ['B_L4_rho_1.00_S25_500rpk.npz', 'B_L4_rho_1.00_S100_200rpk.npz', 'B_L4_rho_1.00_S100_200rpk.npz']
    rho0_files_4 = ['B_L4_rho_0.00_S25_500rpk.npz', 'B_L4_rho_0.00_S100_200rpk.npz', 'B_L4_rho_0.00_S100_200rpk.npz']
    Bind_files_4 = ['B_L4_ind_S25_500rpk.npz', 'B_L4_ind_S100_500rpk.npz', 'B_L4_ind_S100_500rpk.npz']

    Ks_list_rho1_3, fracs_list_rho1_3, errs_list_rho1_3, K50_list_rho1_3, K50_err_list_rho1_3 = zip(*[get_Scurve_data(rho_dir_3, f) for f in rho1_files_3])
    Ks_list_rho0_3, fracs_list_rho0_3, errs_list_rho0_3, K50_list_rho0_3, K50_err_list_rho0_3 = zip(*[get_Scurve_data(rho_dir_3, f) for f in rho0_files_3])
    Ks_list_Bind_3, fracs_list_Bind_3, errs_list_Bind_3, K50_list_Bind_3, K50_err_list_Bind_3 = zip(*[get_Scurve_data(ind_dir_3, f) for f in Bind_files_3])

    Ks_list_rho1_4, fracs_list_rho1_4, errs_list_rho1_4, K50_list_rho1_4, K50_err_list_rho1_4 = zip(*[get_Scurve_data(rho_dir_4, f) for f in rho1_files_4])
    Ks_list_rho0_4, fracs_list_rho0_4, errs_list_rho0_4, K50_list_rho0_4, K50_err_list_rho0_4 = zip(*[get_Scurve_data(rho_dir_4, f) for f in rho0_files_4])
    Ks_list_Bind_4, fracs_list_Bind_4, errs_list_Bind_4, K50_list_Bind_4, K50_err_list_Bind_4 = zip(*[get_Scurve_data(ind_dir_4, f) for f in Bind_files_4])
        

    # set up plots
    box_props = dict(boxstyle='square', facecolor='white', alpha=0, edgecolor='None')
    rho1text = r'3, $B_{\rho}$, $\rho=1$'+f' \nS = {S}, C = {C}, K = {K1:.1f}'
    rho0text = r'3 stages, $B_{\rho}$, $\rho=0$'+f' \nS = {S}, C = {C}, K = {K0_3:.1f}'
    indtext = r'3 stages, $B_{ind}$'+f' \nS = {S}, C = {C}, K = {Kind:.1f}'
    eigcol = 'olivedrab'
    Scols = ['lightskyblue', 'dodgerblue', 'blue']
    Scol1 = 'grey'
    axfontsize = 10


    fig = plt.figure(figsize=(15, 20))
    gs = gridspec.GridSpec(4, 3, height_ratios=[6,2.5, 6,2.5], width_ratios=[2, 2, 2], hspace=0.3, wspace=0.3)
    s1_3 = fig.add_subplot(gs[0, 0])
    s0_3 = fig.add_subplot(gs[0, 1])
    sind_3 = fig.add_subplot(gs[0, 2])

    s1_4 = fig.add_subplot(gs[2, 0])
    s0_4 = fig.add_subplot(gs[2, 1])
    sind_4 = fig.add_subplot(gs[2, 2])

    # for broken axis: 
    xlims_broken1_3 = ((-2017, -1998), (-6, 2))
    xlims_broken0_3 = ((-2015, -1998), (-6, 2))
    xlims_brokenind_3 = ((-2010, -1998), (-6, 2))
    eig1_3 = brokenaxes(xlims=xlims_broken1_3, subplot_spec=gs[1, 0], fig=fig, wspace=0.06, d=0.005, tilt=70)
    eig0_3 = brokenaxes(xlims=xlims_broken0_3, subplot_spec=gs[1, 1], fig=fig,  wspace=0.06, d=0.005, tilt=70)
    eigind_3 = brokenaxes(xlims=xlims_brokenind_3, subplot_spec=gs[1, 2], fig=fig,  wspace=0.06, d=0.005, tilt=70)


    xlims_broken1_4 = ((-2017, -1998), (-6, 2))
    xlims_broken0_4 = ((-2015, -1998), (-6, 2))
    xlims_brokenind_4 = ((-2010, -1998), (-6, 2))
    
    eig1_4 = brokenaxes(xlims=xlims_broken1_4, subplot_spec=gs[3, 0], fig=fig, wspace=0.06, d=0.005, tilt=70)
    eig0_4 = brokenaxes(xlims=xlims_broken0_4, subplot_spec=gs[3, 1], fig=fig,  wspace=0.06, d=0.005, tilt=70)
    eigind_4 = brokenaxes(xlims=xlims_brokenind_4, subplot_spec=gs[3, 2], fig=fig,  wspace=0.06, d=0.005, tilt=70)

    # S curve plots 
    
    # panel 1: 
    S_curve_panel(s1_3, Ks_onestage, fracs_onestage, errs_onestage, K50_onestage, K50_err_onestage, Ks_list_rho1_3, fracs_list_rho1_3, errs_list_rho1_3, K50_list_rho1_3, K50_err_list_rho1_3, 
                  K1, (0.7, 1.4), r'3 stages, $\rho=1$', first_panel=True)
    S_curve_panel(s0_3, Ks_onestage, fracs_onestage, errs_onestage, K50_onestage, K50_err_onestage, Ks_list_rho0_3, fracs_list_rho0_3, errs_list_rho0_3, K50_list_rho0_3, K50_err_list_rho0_3,
                  K0_3, (2, 5), r'3 stages, $\rho=0$')
    S_curve_panel(sind_3, Ks_onestage, fracs_onestage, errs_onestage, K50_onestage, K50_err_onestage, Ks_list_Bind_3, fracs_list_Bind_3, errs_list_Bind_3, K50_list_Bind_3, K50_err_list_Bind_3,
                  K1, (0.7, 1.4), r'3 stages, $B_{ind}$')

    S_curve_panel(s1_4, Ks_onestage, fracs_onestage, errs_onestage, K50_onestage, K50_err_onestage, Ks_list_rho1_4, fracs_list_rho1_4, errs_list_rho1_4, K50_list_rho1_4, K50_err_list_rho1_4, 
                      K1, (0.7, 1.4), r'4 stages, $\rho=1$', first_panel=False)
    S_curve_panel(s0_4, Ks_onestage, fracs_onestage, errs_onestage, K50_onestage, K50_err_onestage, Ks_list_rho0_4, fracs_list_rho0_4, errs_list_rho0_4, K50_list_rho0_4, K50_err_list_rho0_4,
                      K0_4, (2, 5), r'4 stages, $\rho=0$')
    S_curve_panel(sind_4, Ks_onestage, fracs_onestage, errs_onestage, K50_onestage, K50_err_onestage, Ks_list_Bind_4, fracs_list_Bind_4, errs_list_Bind_4, K50_list_Bind_4, K50_err_list_Bind_4,
                      K1, (0.7, 1.4), r'4 stages, $B_{ind}$')

    # eigenvalue plots 
    e1text_3 = r'2 stages, $B_{\rho}$, $\rho=1$'+f' \nS = {S}, C = {C}, K = {K1:.1f}'
    e0text_3 = r'2 stages, $B_{\rho}$, $\rho=0$'+f' \nS = {S}, C = {C}, K = {K0_3:.1f}'
    eindtext_3 = r'2 stages, $B_{ind}$'+f' \nS = {S}, C = {C}, K = {Kind:.1f}'

    e1text_4 = r'2 stages, $B_{\rho}$, $\rho=1$'+f' \nS = {S}, C = {C}, K = {K1:.1f}'
    e0text_4 = r'2 stages, $B_{\rho}$, $\rho=0$'+f' \nS = {S}, C = {C}, K = {K0_4:.1f}'
    eindtext_4 = r'2 stages, $B_{ind}$'+f' \nS = {S}, C = {C}, K = {Kind:.1f}'

    


    eig_panel(eig1_3, Jeigs_1_3, e1text_3, (-2.1, 2.1), first_panel=True)
    eig_panel(eig0_3, Jeigs_0_3, e0text_3)
    eig_panel(eigind_3, Jeigs_ind_3, eindtext_3)
    eig_panel(eig1_4, Jeigs_1_4, e1text_4, (-2.1, 2.1), first_panel=True)
    eig_panel(eig0_4, Jeigs_0_4, e0text_4)
    eig_panel(eigind_4, Jeigs_ind_4, eindtext_4)
    #eig_panel(eig0, Jeigs_0, e0text, (-20, 1))
    #eig_panel(eigind, Jeigs_ind, eindtext, (-20, 1))

    #plt.tight_layout()
    plt.savefig(os.path.join(current_dir, '..', '..', 'figures', 'Scurves_34stage.pdf'), dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()



