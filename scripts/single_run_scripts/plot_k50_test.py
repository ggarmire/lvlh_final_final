import numpy as np
import lvlh_functions as lvf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
import os 
from scipy.optimize import curve_fit
from brokenaxes import brokenaxes

def K50_asym_fit(x, y0, A, B):
    return y0 + (A-y0)*x/(B+x)




def main():
    #rhos = [-1./3., 0, 0.5, 1]
    rhos = [0, 1]
    asym_lines = [2, 2/((2.5)**0.5), 1]

    rhocols = ['tab:blue', 'tab:green', 'tab:orange', 'tab:red']
    high_delta = 1000
    high_S = 1000
    current_dir = os.path.dirname(os.path.abspath(__file__))

    K50maxbar = 10

    # read in data here: 


    # get data for k50 vs S 
    data_KofS = {}
    k50S_dir = os.path.join(current_dir, '..', '..', 'data', 'K50_data', 'K50_ofS', f'delta={high_delta}')
    for irho, rho in enumerate(rhos):
        filepath = os.path.join(k50S_dir, f'K50byS_rho{rho:.2f}_S10-1000_100rpk.npz')
        with np.load(filepath) as data: 
            data_KofS[rho] = {'K50s': data['K50s'], 'K50_errs': data['K50_errs'], 'Ss': data['Ss']}

    filepath_ind = os.path.join(k50S_dir, f'K50byS_Bind_S10-1000_100rpk.npz')
    with np.load(filepath_ind) as data: 
        data_KofS['ind'] = {'K50s': data['K50s'], 'K50_errs': data['K50_errs'], 'Ss': data['Ss']}
    
    
    


    # get data for k50 vs delta 
    data_Kofdelta = {}
    k50delta_dir = os.path.join(current_dir, '..', '..', 'data', 'K50_data', 'K50_ofdelta', f'S={high_S}')
    for irho, rho in enumerate(rhos):
            filepath = os.path.join(k50delta_dir, f'K50bydelta_rho{rho:.2f}_delta0.01-500.0_100rpk.npz')
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



    # S vs K50
    plt.hlines(y=asym_lines, xmin = 0, xmax=1100, color='grey', linestyle='--', lw=1)

    for i, rho in enumerate(rhos):
        d = data_KofS[rho]
        Ss = d['Ss']
        K50s = d['K50s']
        K50_errs = d['K50_errs']
        # fit it real quick
        print(f'rho={rho}')
        x_fit = np.linspace(10, 1200, 200)
        if rho>=0.3:
            p0 = [K50s[0], 2*(1+3*(rho))**(-0.5), 10]
        else: p0 = [K50s[0], 47, 10]
        popt, _ = curve_fit(K50_asym_fit, Ss, K50s, p0, maxfev=10000)
        y_fit = K50_asym_fit(x_fit, *popt)
        #plt.plot(x_fit, y_fit, '--', color=rhocols[i], linewidth=1, alpha = 0.7)
        plt.errorbar(d['Ss'], d['K50s'], yerr=d['K50_errs'], fmt='o--', capsize=3, label=(fr'$\rho = -⅓$') if rho==-1./3. else fr'$\rho = {rho}$', color=rhocols[i])
    di = data_KofS['ind']
    plt.errorbar(di['Ss'], di['K50s'], yerr=di['K50_errs'], fmt='x--', capsize=3, label='Bind', color='black')
    #plt.set_title(fr'$K_{{50}}$ vs number of species S ($\delta = {high_delta}$)', fontsize=16)
    plt.xlabel('number of species (S)', labelpad=20)
    plt.ylabel(r'stability threshold ($K_{50}$)')
    plt.legend()
    plt.xlim((0, 1050))
    plt.grid(True, alpha=0.3)


    # plot setup 
    fig = plt.figure(figsize=(18, 9))
    # delta vs K50
    plt.hlines(y=asym_lines, xmin = 0, xmax=10000, color='grey', linestyle='--', lw=1)   
    for i, rho in enumerate(rhos):
        d = data_Kofdelta[rho]
        deltas = d['deltas']
        K50s = d['K50s']
        K50_errs = d['K50_errs']
        print(deltas, K50s)
        # fit it real quick
        x_fit = np.logspace(np.log10(min(deltas)), np.log10(10000), 200)
        if rho >-1./3.: 
            p0 = [K50s[0], 2*(1+3*(rho))**(-0.5), 1]

        else: p0 = [0, 1, 1]
        popt, _ = curve_fit(K50_asym_fit, deltas, K50s, p0)
        print(f'fit for rho={rho} \n{popt}')
        y_fit = K50_asym_fit(x_fit, *popt)
        #plt.plot(x_fit, y_fit, '--', color=rhocols[i], linewidth=1, alpha = 0.7)
        plt.errorbar(deltas, K50s, yerr=K50_errs, fmt='o--', capsize=3, label=(fr'$\rho = -⅓$') if rho==-1./3. else fr'$\rho = {rho}$', color=rhocols[i])
    di = data_Kofdelta['ind']
    plt.errorbar(di['deltas'], di['K50s'], yerr=di['K50_errs'], fmt='x--', capsize=3, label='Bind', color='black')
    
    #plt.set_title(fr'$K_{{50}}$ vs Delta ($S = {high_S}$)', fontsize=16)
    plt.xlabel(r'$\Delta$')
    plt.ylabel(r'stability threshold ($K_{50}$)', labelpad=20)
    plt.legend()
    plt.xlim((-0.1, 1300))
    plt.ylim((0, 55))
    plt.xscale('symlog')
    #plt.yscale('symlog')
    plt.grid(True, alpha=0.3)

    plt.show()




if __name__ == "__main__": main()