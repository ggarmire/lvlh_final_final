import numpy as np
import lvlh_functions as lvf
'''
special functions used in analysis
'''
# check stability from an interaction matrix and a 
def check_stable(B, R):
    '''
    takes in interaction matrix B and growth vector/demographic matrix R, returns stable status as binary (1=yes)
    '''
    J = lvf.Jacobian(B, R)
    eig1 = np.max(np.real(np.linalg.eigvals(J)))
    return eig1<0 


# pad the S curve to min and max K values 
def pad_S_curve(Ks, fracs, frac_errs, kmin = 0.5, kmax = 1.5):
    '''
    This adds extension on to the S curves at high S, where I didnt compute the tails all the way to kmin and kmax.
    Ks = Ks used in S curve generation 
    fracs = stable fractions from S curve generation 
    kmin = where to extend S curve to on the left 
    kmax = where to extend S curve to on the right  
    '''
    kpad = np.copy(Ks); fpad = np.copy(fracs); ferrpad = np.copy(frac_errs)
    if kpad[0] > kmin:
        kpad = np.insert(kpad, 0, kmin)
        fpad = np.insert(fpad, 0, 1.0)
        ferrpad = np.insert(ferrpad, 0, 0.0)
    if kpad[-1] < kmax:
        kpad = np.append(kpad, kmax)
        fpad = np.append(fpad, 0.0)
        ferrpad = np.append(ferrpad, 0.0)
    return kpad, fpad, ferrpad


# bootstrapping function for uncertainty estimates 
def bootstrap_error(results, nboots=1000, seed=0):
    '''
    calculate standard error on a sample by bootstrapping.

    results = array containing the sample values.
    nboots = numbere of resampling iterations to perform

    returns stderr = standard error from bootstrapping.
    '''
    np.random.seed(seed)
    results_array = np.array(results)       # in case results is a list rather than an array 
    nruns = len(results_array)

    boot_indices = np.random.randint(0, nruns, size=(nboots, nruns))    # random indices to resample
    boot_samples = results_array[boot_indices]
    boot_means = np.mean(boot_samples, axis=1)
    stderr = np.std(boot_means, ddof = 1)
    return stderr 

def bootstrap_interp_error(K_low, K_high, stables_low, stables_high, nboots=1000, seed=0):
    '''
    modified bootstrapping method specifically for interpokation of the 50% stable fraction, calculate standard error on K50.

    stables_low/stables_high: array of T/F data for nruns 
    nboots = numbere of resampling iterations to perform
    
    returns K50err = standard error on the 50% thresholdfrom bootstrapping.
    '''
    np.random.seed(seed)
    stables_low = np.array(stables_low); stables_high = np.array(stables_high)      #reformat just in case 
    nruns = len(stables_low)
    # get random set of resampling indices for each boot iteration 
    boot_indices = np.random.randint(0, nruns, (nboots,nruns))
    # get f_low and f_high for each boot iteration
    f_low_boot = np.mean(stables_low[boot_indices], axis = 1)
    f_high_boot = np.mean(stables_high[boot_indices], axis = 1)
    # interpolate each low/high pair:
    K50_boots = np.full(nboots, (K_high + K_low) / 2.0)     # where f_low and f_high are equal for any of the boots, to avoid nans and 1/0 errors 
    safe_pairs = f_low_boot != f_high_boot
    K50_boots[safe_pairs] = K_high - (K_high-K_low) / (f_high_boot[safe_pairs]-f_low_boot[safe_pairs]) * (f_high_boot[safe_pairs] - 0.5)

    K50_stderr = np.std(K50_boots, ddof=1)
    return K50_stderr
