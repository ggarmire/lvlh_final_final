import numpy as np
import matplotlib.pyplot as plt

def main():

    filename = "data/C_data/CK_heatmap_S10_rho0.0_200rpk.npz"
    data = np.load(filename)
    
    Cs = data['Cs']
    Ks = data['Ks']
    stable_fracs = data['stable_fracs']
    rho = data['rho']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create the heatmap
    # stable_fracs is shape (len(Cs), len(Ks)), so we transpose it for pcolormesh
    C_grid, K_grid = np.meshgrid(Cs, Ks, indexing='ij')
    c = ax.pcolormesh(C_grid, K_grid, stable_fracs, cmap='viridis_r', shading='nearest')
    
    # Add the theoretical stability threshold
    # For 2-stages, large S, delta->inf: K_theory = 2 * (1 + 3*rho)^(-0.5)
    K_theory = 2 * (1 + 3 * rho)**(-0.5)
    ax.axhline(K_theory, color='black', linestyle='--', linewidth=2, label=f'Theoretical Limit ($K={K_theory:.2f}$)')
    
    # Formatting
    cbar = fig.colorbar(c, ax=ax)
    cbar.set_label('Fraction of Stable Runs', rotation=270, labelpad=20, fontsize=12)
    
    ax.set_xlabel('Connectance (C)', fontsize=12)
    ax.set_ylabel('Complexity (K)', fontsize=12)
    ax.set_title(fr'Stability Heatmap ($\rho={rho}$, $S={data["S"]}$)', fontsize=14)
    ax.legend(loc='upper right', framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig('figures/CK_heatmap.pdf', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()