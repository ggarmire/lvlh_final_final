import os 


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
import lvlh_functions as lvf

def main():

    Ss = [50, 500]
    C = 1.0
    delta = 1000
    nruns = 200

    mw = 90

    
    filestart = f'figures/'
    output_dir = os.path.dirname(filestart)
    if output_dir:  
        os.makedirs(output_dir, exist_ok=True)
    filename = f'{filestart}ap_k50searchalgo_S{Ss[0]}+{Ss[1]}_nruns{nruns}.pdf'


    rhos = [0, 0.5, 1.0]
    colors = ['C0', 'C0', 'C0']
    panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    K_guesses = [0.8, 2, 1.9]

    fig, axs = plt.subplots(2, 3, figsize=(14, 12))

    with concurrent.futures.ProcessPoolExecutor(max_workers=mw) as executor:
        for row, S in enumerate(Ss):
            for col, rho in enumerate(rhos):
                ax = axs[row, col]
                idx = row * 3 + col
                K_guess = K_guesses[col]
                
                K50, K50_err, K_history = lvf.find_K50_threshold_rho_delta(
                    S, C, rho, delta, nruns, executor, K_guess=K_guess
                )

                tested_Ks = list(K_history.keys())
                stable_fracs = list(K_history.values())

                sorted_indices = np.argsort(tested_Ks)
                sorted_Ks = np.array(tested_Ks)[sorted_indices]
                sorted_fs = np.array(stable_fracs)[sorted_indices]
                
                for j in range(len(sorted_Ks) - 1):
                    if (sorted_fs[j] >= 0.5 >= sorted_fs[j+1]) or (sorted_fs[j] <= 0.5 <= sorted_fs[j+1]):
                        interp_Ks = [sorted_Ks[j], sorted_Ks[j+1]]
                        interp_fs = [sorted_fs[j], sorted_fs[j+1]]
                        break

                ax.plot(interp_Ks, interp_fs, color='C0', linestyle='--', 
                        linewidth=1.5, zorder=2, label='Interpolation Segment')

                ax.scatter(tested_Ks, stable_fracs, color=colors[col], alpha=0.5, 
                        s=100, label='Evaluated Points', zorder=2)
                ax.scatter(tested_Ks[0], stable_fracs[0], color='C0', 
                        marker='D', s=100, label='Initial Guess', zorder=2, alpha=0.5)
                
                for j in range(len(tested_Ks)):
                    ax.annotate(
                        str(j + 1), 
                        xy=(tested_Ks[j], stable_fracs[j]),
                        xytext=(0, -1), # Offset 6 points vertically so it floats on top
                        textcoords="offset points",
                        ha='center', va='center',
                        fontsize=10, fontweight='600', color='black',
                        zorder=3
                    )
                
                ax.errorbar(K50, 0.5, xerr=K50_err, fmt='D', color='black', ms=8, alpha=0.8, 
                            capsize=4, label=f'Calculated $K_{{50}}$', zorder=4)

                ax.axhline(0.5, color='black', linestyle='--', alpha=0.3)
                ax.set_xlabel('Complexity $K$', fontsize=12)
                #ax.set_title(fr'2 Stages, $\rho={rho}$', fontsize=14)
                ax.grid(True, linestyle='--', alpha=0.4)

                #box_props = dict(boxstyle='square', facecolor='white', alpha=0.8, edgecolor='lightgray')
                box_props = dict(boxstyle='square', facecolor='white', alpha=0.85, edgecolor='none')
                box_props = dict(boxstyle='square', facecolor='white', alpha=0.85, edgecolor='none')
                #param_text = f"$\\Delta$ = {delta}, $\\rho$ = {rho}\n$S$ = {S}, nruns = {nruns} \n$K_{50}$={K50}$\pm${K50_err}"
                #ax.text(0.05, 0.05, param_text, transform=ax.transAxes, 
                #        fontsize=10, verticalalignment='bottom', bbox=box_props, zorder=6)
                k50_text = f"$K_{{50}} = {K50:.3f} \\pm {K50_err:.3f}$"
                ax.text(0.05, 0.04, k50_text, transform=ax.transAxes, 
                        fontsize=14, color='black', verticalalignment='bottom', 
                        bbox=box_props, zorder=6)
                param_text = f"$\\rho$ = {rho}, $\\Delta$ = {delta} \n$S$ = {S}, nruns = {nruns}  "
                ax.text(0.05, 0.09, param_text, transform=ax.transAxes, 
                        fontsize=10, color='black', verticalalignment='bottom', 
                        bbox=box_props, zorder=7)

                
                if col == 0:
                    ax.set_ylabel('Fraction of Stable Runs', fontsize=12)
                    
                ax.legend(loc='upper right', fancybox=False, framealpha=0.8, edgecolor='white')
                
                ax.text(0, 1.02, panel_labels[idx], transform=ax.transAxes, 
                        fontsize=16, va='bottom', ha='right')

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()