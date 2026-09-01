import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import lvlh_functions as lvf

def generate_system(stages, K, S, C, seed, delta=1000):
    sigma = K * (S * C)**(-0.5)
    if stages == 1:
        B = lvf.A_onestage(S, C, sigma, seed)
        R = lvf.r_star(B)
    else:
        B = lvf.B_rho(S, C, sigma, seed, L=2, rho=0.5)
        R = lvf.R_star_2stage_delta(B, delta)
    J = lvf.Jacobian(B, R)
    return B, R, J

def main(): 
    S = 100
    C = 1
    delta = 2
    ts = np.linspace(0, 100, 1000)

    configs = [
        {'stages': 1, 'K': 0.8, 'title': '1 Stage, C=1, K=0.8'},
        {'stages': 1, 'K': 1.2, 'title': '1 Stage, C=1, K=1.3'},
        {'stages': 2, 'K': 1.2, 'title': r'2 Stages, C=1, $\Delta=1$, $\rho=0.5$, K=1.2'},
        {'stages': 2, 'K': 1.3, 'title': r'2 Stages, C=1, $\Delta=1$, $\rho=0.5$, K=1.3'}
    ]

    fig, axs = plt.subplots(2, 2, figsize=(14, 8))
    axs = axs.flatten()

    box_props = dict(boxstyle='square', facecolor='white', alpha=0.7, edgecolor='None')
    panel_labels = ['(a)', '(b)', '(c)', '(d)']

    for i, conf in enumerate(configs):
        ax = axs[i]
        
            
        

        B, R, J = generate_system(conf['stages'], conf['K'], S, C, seed=i+38, delta=delta)
        Jeigs = np.linalg.eigvals(J)
        lead_eig = max(Jeigs, key=np.real)

        eig_text = f"{np.real(lead_eig):.3f}"
        if np.imag(lead_eig) != 0:
            sign = r'$\pm$'
            eig_text += f" {sign} {abs(np.imag(lead_eig)):.3f}i"
        full_text = f"{conf['title']}\nLeading " + r"$\lambda$: " + eig_text
        
        n = len(B)
        x0 = lvf.init_cond(S, L=conf['stages'], spread=0.2, seed=1)
        
        sol = lvf.evolve_system(B, R, x0, ts)
        ax.plot(ts, sol[:,10:], alpha=0.2, linewidth=0.8, color='grey')
        ax.plot(ts, sol[:,:10], alpha=1, linewidth=1.2)
        #ax.set_title(conf['title'], fontsize=12)
        ax.set_xlabel('Time')
        ax.set_ylabel('Population Density')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_xlim(0,100)

        ax.text(0.05, 0.97, full_text, transform=ax.transAxes, 
                fontsize=12, verticalalignment='top', bbox=box_props)
        ax.text(-0.1, 1.0, panel_labels[i], transform=ax.transAxes, 
                fontsize=16, va='bottom', ha='right')


    plt.tight_layout()
    plt.savefig('figures/ap_systemevolution.pdf', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()





