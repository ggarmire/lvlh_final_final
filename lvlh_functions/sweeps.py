import numpy as np
import os
import concurrent.futures
import lvlh_functions as lvf
import time 
from scipy.sparse.linalg import eigs
from scipy.interpolate import interp1d


# S curves 

def _run_sweep_task(task_args):
    '''
    runs one task for the sweep below. Returns stability status of the task. 
    '''
    matrix_function, S, C, K, seed, B_args, R_args = task_args
    sigma = K * (S*C)**(-0.5)
    B = matrix_function(S=S, C=C, sigma=sigma, seed=seed, **B_args)
    if matrix_function is lvf.A_onestage: R = lvf.r_star(B)
    else: R = lvf.R_star_2stage_delta(B, **(R_args or {}))
    return lvf.check_stable(B, R)

def generate_S_curve(matrix_function, S, C, B_args, Ks, nruns, filestart, R_args=None, maxworkers=None):
    '''
    Find the fraction of runs stable for each K in Ks.
    maxtrix_function = function used to generate the interaction matrix 
    S = number of species 
    C = connectance 
    B_args = additional args needed depending on the type of interaction matrix 
    Ks = array of K values at which to calculate the fraction of runs stable 
    nruns = number of runs tried at each K 
    filestart = 
    R_args = any necessary arguments for the demographic matrix (namely delta)
    maxworkers = max number of CPU cores to use in parallel. If None, will use all available cores.  
    '''
    if maxworkers == None: maxworkers = os.cpu_count()

    nKs = len(Ks)
    fracs = np.zeros(nKs)
    frac_errs = np.zeros(nKs)
    start = time.time()
    print(f"Starting sweep for {S} species, {nKs} Ks, {nruns} runs per K\nKmin = {np.min(Ks)}, Kmax = {np.max(Ks)}")
    print(f"using {maxworkers} CPU cores.")

    # set up tasks to run in parallel:
    tasks = []
    for i, K in enumerate(Ks): 
        for run in range(nruns):
            seed = run
            task_args = (matrix_function, S, C, K, seed, B_args, R_args)
            tasks.append((i, task_args))

    stable_outcomes = {i: [] for i in range(len(Ks))}
    total_tasks = len(tasks)
    completed = 0

    # run tasks parallel: 
    with concurrent.futures.ProcessPoolExecutor(max_workers=maxworkers) as executor:
        results = executor.map(_run_sweep_task, [t[1] for t in tasks], chunksize = 10)
        for (i, _), is_stable in zip(tasks, results):
            stable_outcomes[i].append(int(is_stable))
            completed += 1
            if completed % 50 == 0 or completed == total_tasks: 
                print(f"{completed}/{total_tasks} cases done ({(completed/total_tasks)*100:.1f}%)", end='\r')

    for i, K in enumerate(Ks):
        fracs[i] = np.mean(np.array(stable_outcomes[i]))
        frac_errs[i] = lvf.bootstrap_error(stable_outcomes[i], nboots=1000, seed=i)
        print(f"  K = {K:.2f}, fraction stable = {fracs[i]:.3f}+-{frac_errs[i]:.3f}")
    end = time.time()
    print(f'took {(end-start)/60} min, {(end-start)/total_tasks} sec per task.')

    filename = f"{filestart}_S{S}_{nruns}rpk.npz"
    np.savez_compressed(filename, Ks=Ks, fracs=fracs, frac_errs = frac_errs, S=S)
    print(f"\nData saved successfully to: {filename}")

    return fracs, frac_errs


# find 50% stable K threshold

def _worker_stable_check(task_args):
    '''
    to use with evaluate_oneK_stablefrac, check stable/unstable for each task.
    '''
    S, C, K, rho, delta, seed = task_args 
    sigma = K * (S*C)**(-0.5)
    B = lvf.B_rho(S, C, sigma, seed, L=2, rho=rho)
    R = lvf.R_star_2stage_delta(B, delta)
    # return true/false for stable 
    return lvf.check_stable(B, R)

def evaluate_oneK_stablefrac(K, S, C, rho, delta, nruns, maxworkers):
    # set up nruns tasks: find stable/unstable for each run
    tasks = [(S, C, K, rho, delta, seed) for seed in range(nruns)]
    # run tasks in parallel:
    with concurrent.futures.ProcessPoolExecutor(max_workers=maxworkers) as executor:
        stables = list(executor.map(_worker_stable_check, tasks, chunksize=10))
    # get actual stable frac 
    stable_count = sum(stables)
    stable_frac = stable_count / nruns
    # need to return stables for bootstrapping later 
    return stable_frac, stables   

def find_K50_threshold_rho_delta(S, C, rho, delta, nruns, K_guess=1, stepsize = 0.4, maxworkers=None):
    K_history = {}      # to store tested values of K
    def get_f(K):
        '''
        given the established system, find the fraction of runs stable at the given K value
        '''
        K_round = round(K,4)
        if K_round not in K_history: 
            K_history[K_round] = evaluate_oneK_stablefrac(K, S, C, rho, delta, nruns, maxworkers)
            print(f'K={K_round} has {K_history[K_round][0]*100}% runs stable. \r')
        f = K_history[K_round][0]
        return f
    
    K_current = K_guess 
    f_current = get_f(K_current)

    step = stepsize if f_current > 0.5 else -stepsize       # set direction in which to step K based on f
    # try different values of K until we get Ks for which f(K)>0.5 and f(K)<0.5
    while True: 
        step = stepsize if f_current > 0.5 else -stepsize
        # step K
        K_next = K_current + step 
        if K_next <=0: K_next = 0.0001      # prevent negative K 
        # find next fraction 
        f_next = get_f(K_next)
        # check if we have found a value on either side of 0.5
        if (f_current > 0.5 and f_next <= 0.5) or (f_current < 0.5 and f_next >= 0.5):
            break
        K_current = K_next
        f_current = f_next
        if f_current == 0.0 or f_current == 1.0:       # if we are far away, increase stepsize
            stepsize *= 2.0
    # get a window inside f = (0.4, 0.6) to interpolate for f=0.5
    K_low, K_high = min(K_current, K_next), max(K_current, K_next)
    while True:
        f_low = get_f(K_low)
        f_high = get_f(K_high)
        # check if we have constrained the window: 
        if (0.4 <= f_low <= 0.6) and (0.4 <= f_high <= 0.6):
            break
        # if the Ks are very close together, i.e. the line is almost vertical, stop.
        if abs(K_high - K_low) < 0.001:
            break
        # take midpoint of line and test 
        K_mid = (K_low + K_high) / 2.0
        f_mid = get_f(K_mid)
        # reset either upper or lower limit to K_mid
        if f_mid > 0.5:
            K_low = K_mid
        else:
            K_high = K_mid
    K_history_fracs = {k: v[0] for k, v in K_history.items()}   # all fractions for tested Ks, useful for plotting 

    # get interp point, get bootstrapped error. 
    K50 = K_high - (K_high-K_low)/(f_high-f_low)*(f_high-0.5)

    K_low_round = round(K_low,4); K_high_round = round(K_high,4)
    stables_low = K_history[K_low_round][1]
    stables_high = K_history[K_high_round][1]
    K50_err = lvf.bootstrap_interp_error(K_low, K_high, stables_low, stables_high)

    return K50, K50_err, K_history_fracs

def _worker_stable_check_Bind(task_args):
    '''
    to use with evaluate_oneK_stablefrac, check stable/unstable for each task.
    '''
    S, C, K, delta, seed = task_args 
    sigma = K * (S*C)**(-0.5)
    B = lvf.B_ind(S, C, sigma, 2, seed)
    R = lvf.R_star_2stage_delta(B, delta)
    # return true/false for stable 
    return lvf.check_stable(B, R)

def evaluate_oneK_stablefrac_Bind(K, S, C,  delta, nruns, maxworkers):
    # set up nruns tasks: find stable/unstable for each run
    tasks = [(S, C, K, delta, seed) for seed in range(nruns)]
    # run tasks in parallel:
    with concurrent.futures.ProcessPoolExecutor(max_workers=maxworkers) as executor:
        stables = list(executor.map(_worker_stable_check_Bind, tasks, chunksize=10))
    # get actual stable frac 
    stable_count = sum(stables)
    stable_frac = stable_count / nruns
    # need to return stables for bootstrapping later 
    return stable_frac, stables  

def find_K50_threshold_Bind_delta(S, C, delta, nruns, K_guess=1, stepsize = 0.4, maxworkers=None):
    K_history = {}      # to store tested values of K
    def get_f(K):
        '''
        given the established system, find the fraction of runs stable at the given K value
        '''
        K_round = round(K,4)
        if K_round not in K_history: 
            K_history[K_round] = evaluate_oneK_stablefrac_Bind(K, S, C, delta, nruns, maxworkers)
            print(f'K={K_round} has {K_history[K_round][0]*100}% runs stable. \r')
        f = K_history[K_round][0]
        return f
    
    K_current = K_guess 
    f_current = get_f(K_current)

    step = stepsize if f_current > 0.5 else -stepsize       # set direction in which to step K based on f
    # try different values of K until we get Ks for which f(K)>0.5 and f(K)<0.5
    while True: 
        step = stepsize if f_current > 0.5 else -stepsize
        # step K
        K_next = K_current + step 
        if K_next <=0: K_next = 0.0001      # prevent negative K 
        # find next fraction 
        f_next = get_f(K_next)
        # check if we have found a value on either side of 0.5
        if (f_current > 0.5 and f_next <= 0.5) or (f_current < 0.5 and f_next >= 0.5):
            break
        K_current = K_next
        f_current = f_next
        if f_current == 0.0 or f_current == 1.0:       # if we are far away, increase stepsize
            stepsize *= 2.0
    # get a window inside f = (0.4, 0.6) to interpolate for f=0.5
    K_low, K_high = min(K_current, K_next), max(K_current, K_next)
    while True:
        f_low = get_f(K_low)
        f_high = get_f(K_high)
        # check if we have constrained the window: 
        if (0.4 <= f_low <= 0.6) and (0.4 <= f_high <= 0.6):
            break
        # if the Ks are very close together, i.e. the line is almost vertical, stop.
        if abs(K_high - K_low) < 0.001:
            break
        # take midpoint of line and test 
        K_mid = (K_low + K_high) / 2.0
        f_mid = get_f(K_mid)
        # reset either upper or lower limit to K_mid
        if f_mid > 0.5:
            K_low = K_mid
        else:
            K_high = K_mid
    K_history_fracs = {k: v[0] for k, v in K_history.items()}   # all fractions for tested Ks, useful for plotting 

    # get interp point, get bootstrapped error. 
    K50 = K_high - (K_high-K_low)/(f_high-f_low)*(f_high-0.5)

    K_low_round = round(K_low,4); K_high_round = round(K_high,4)
    stables_low = K_history[K_low_round][1]
    stables_high = K_history[K_high_round][1]
    K50_err = lvf.bootstrap_interp_error(K_low, K_high, stables_low, stables_high)

    return K50, K50_err, K_history_fracs
    











    