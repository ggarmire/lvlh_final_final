import numpy as np
import lvlh_functions as lvf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os 

# evolve and get Jacobian for a single run with 1 stage per species 
def single_run_single_stage(S, K, C, ts, Aseed, x0seed=None):
    if x0seed == None: x0seed = Aseed 
    x0 = lvf.init_cond(S, 1, 0.2, x0seed)

    sigma = K * (S*C)**(-0.5)
    A = lvf.A_onestage(S, C, sigma, seed=Aseed)
    r = lvf.r_star(A)

    result = lvf.evolve_system(A, r, x0, ts)
    Jac = lvf.Jacobian(A, r)
    
    return result, Jac

def main():
    # common parameters: 
    S = 500     # number of species 
    C = 1       # connectance
    ts = np.linspace(0, 100, 1000)

    # low K run: 
    K1 = 0.9
    Aseed1 = 1
    result1, Jac1 = single_run_single_stage(S, K1, C, ts, Aseed1)
    J1eig, _ = np.linalg.eig(Jac1)

    # high K run: 
    K2 = 1.1
    Aseed2 = 5
    result2, Jac2 = single_run_single_stage(S, K2, C, ts, Aseed2)
    J2eig, _ = np.linalg.eig(Jac2)

    # load S curve data in 
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, '..', '..', 'data', 'S_curves', 'one_stage')

    data_s25 = np.load(os.path.join(data_dir, 'one_stage_s25_500rpk.npz'))
    data_s100 = np.load(os.path.join(data_dir, 'one_stage_s100_500rpk.npz'))
    data_s1000 = np.load(os.path.join(data_dir, 'one_stage_s1000_500rpk.npz'))

    Ks_25, fracs_25 = lvf.pad_S_curve(data_s25['Ks'], data_s25['fracs'])
    Ks_100, fracs_100 = lvf.pad_S_curve(data_s100['Ks'], data_s100['fracs'])
    Ks_1000, fracs_1000 = lvf.pad_S_curve(data_s1000['Ks'], data_s1000['fracs'])
    K50_25 = data_s25['K50']
    K50_100 = data_s100['K50']
    K50_1000 = data_s1000['K50']


    # find 50% thresholds with error here

    # plotting 
    box_props = dict(boxstyle='square', facecolor='white', alpha=0.7, edgecolor='None')
    K1text = f'Single Stage \nS = {S}, C = {C}, K = {K1}'
    K2text = f'Single Stage \nS = {S}, C = {C}, K = {K2}'
    axfontsize = 12

    fig = plt.figure(figsize=(14, 7))
    gs = gridspec.GridSpec(2, 3, width_ratios=[2, 1.5, 2], hspace=0.3, wspace=0.4)

    res1 = fig.add_subplot(gs[0, 0])
    eig1 = fig.add_subplot(gs[0, 1])
    res2 = fig.add_subplot(gs[1, 0])
    eig2 = fig.add_subplot(gs[1, 1])
    ax_s = fig.add_subplot(gs[:, 2])

    res1.plot(ts, result1[:, 10:], alpha = 0.1, lw = 0.8, color='grey')
    res1.plot(ts, result1[:, :10], lw = 1.5)
    #res1.grid()
    res1.set_xlabel('time', fontsize=axfontsize)
    res1.set_ylabel('abundance', fontsize=axfontsize)
    #res1.set_title(f'K={K1}')
    res1.text(0.05, 0.97, K1text, transform=res1.transAxes, fontsize=9, verticalalignment='top', bbox=box_props)
    res1.set_xlim((0, 100))
    #res1.set_ylim(bottom=0)

    res2.plot(ts, result2[:, 10:], alpha = 0.1, lw = 0.8, color='grey')
    res2.plot(ts, result2[:, :10], lw = 1.5)
    #res2.grid()
    res2.set_xlabel('time', fontsize=axfontsize)
    res2.set_ylabel('abundance', fontsize=axfontsize)
    #res2.set_title(f'K={K2}')
    res2.text(0.05, 0.97, K2text, transform=res2.transAxes, fontsize=9, verticalalignment='top', bbox=box_props)
    res2.set_xlim((0, 100))
    res2.set_ylim(bottom=0)

    #eig1.grid()
    col_eig1 = np.where(np.real(J1eig) > 0,  'darkred', 'green')
    eig1.axhline(0, color='black', linewidth=1, linestyle='--')
    eig1.axvline(0, color='black', linewidth=1, linestyle='--')
    eig1.scatter(np.real(J1eig), np.imag(J1eig), s = 4, c=col_eig1) 
    eig1.set_xlabel(r'$\text{Re}(\lambda_J$)', fontsize=axfontsize)
    eig1.set_ylabel(r'$\text{Im}(\lambda_J$)', fontsize=axfontsize)
    #eig1.set_title('K=0.9')
    eig1.set_aspect('equal', adjustable='box')
    eig1.set_xlim([-2.4, 0.4])
    eig1.set_ylim([-1.4, 1.4])
    eig1.text(0.05, 0.97, K1text, transform=eig1.transAxes, fontsize=9, verticalalignment='top', bbox=box_props)
    
    #eig2.grid()
    col_eig2 = np.where(np.real(J2eig) > 0,  'firebrick', 'green')
    eig2.axhline(0, color='black', linewidth=1, linestyle='--')
    eig2.axvline(0, color='black', linewidth=1, linestyle='--') 
    eig2.scatter(np.real(J2eig), np.imag(J2eig), s = 4, c=col_eig2)
    eig2.set_xlabel(r'$\text{Re}(\lambda_J$)', fontsize=axfontsize)
    eig2.set_ylabel(r'$\text{Im}(\lambda_J$)', fontsize=axfontsize)
    #eig2.set_title('K=1.1')
    eig2.set_aspect('equal', adjustable='box')
    eig2.set_xlim([-2.4, 0.4])
    eig2.set_ylim([-1.4, 1.4])
    eig2.text(0.05, 0.97, K2text, transform=eig2.transAxes, fontsize=9, verticalalignment='top', bbox=box_props)

    
    ax_s.vlines(x=1, ymin=0, ymax=1, color='slategray', linestyle='--', lw=1.5, label='large S limit')
    #ax_s.hlines(y=0.5, xmin=0.5, xmax=1.5, color='black', linestyle='--', lw=1, alpha = 0.5)
    ax_s.fill_between(Ks_1000, fracs_1000-frac_errs_1000, fracs_1000+frac_errs_1000, alpha=0.5, color='blue')
    ax_s.plot(Ks_1000, fracs_1000, '-', alpha=1, lw = 1.5, color='blue', label='S=1000')
    
            
    ax_s.plot(Ks_100, fracs_100, '-', alpha=1, lw = 1.5, color='dodgerblue', label='S=100')
    ax_s.plot(Ks_25, fracs_25, '-', alpha=1, lw = 1.5, color='lightskyblue', label='S=25')
    ax_s.plot(K50_1000, 0.5, 'x', ms = 8, color='blue')
    ax_s.plot(K50_100, 0.5, 'x', ms = 8, color='dodgerblue')
    ax_s.plot(K50_25, 0.5, 'x', ms = 8, color='lightskyblue')

    ax_s.set_xlim(0.5, 1.5)
    ax_s.set_ylim(-0.01, 1.01)
    #ax_s.text(0.96, 0.5, 'large S stability threshold', color='black', rotation=90, va='top', fontsize=8)
    ax_s.set_xlabel(r'complexity $K$', fontsize=axfontsize)
    ax_s.set_ylabel(r'fraction of run stable for given $K$', fontsize=axfontsize)
    ax_s.plot([], [], 'x', color='black', ms=8, label='50% of \nruns stable')

    ax_s.minorticks_on()
    ax_s.grid(True, which='major', linestyle='--', alpha=0.7)
    ax_s.grid(True, which='minor', linestyle='--', alpha=0.3)

    leg = ax_s.legend(title='One Stage, C=1', title_fontproperties={'size':12}, loc = 'upper right', fancybox=False, framealpha=0.3, edgecolor='white')
    leg.get_frame().set_linewidth(0.5)
    legend_texts = leg.get_texts()
    legend_texts[-1].set_fontsize(9)
    # label panels here
    axes = [res1, eig1, res2, eig2, ax_s]
    labels = ['1a)', '2a)', '1b)', '2b)', '3)']
    labels = ['(a)', '(c)', '(b)', '(d)', '(e)']
    for ax, label in zip(axes, labels):
        ax.text(-0.1, 1.0, label, transform=ax.transAxes, fontsize=16, va='bottom', ha='right')

    plt.tight_layout()

    plt.savefig('figures/onestage_ev.png', dpi=300, bbox_inches='tight')
    plt.show()

    plt.show()



    
if __name__ == "__main__": main()