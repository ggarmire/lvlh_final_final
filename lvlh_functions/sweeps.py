import numpy as np
import os
import concurrent.futures
import lvlh_functions as lvf
import time 
from scipy.sparse.linalg import eigs
from scipy.interpolate import interp1d
from contextlib import redirect_stdout


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
            if completed == 0 or (completed % 10 == 0) or completed == total_tasks: 
                print(f"{completed}/{total_tasks} cases done ({(completed/total_tasks)*100:.1f}%)", end='\r')

    for i, K in enumerate(Ks):
        fracs[i] = np.mean(np.array(stable_outcomes[i]))
        frac_errs[i] = lvf.bootstrap_error(stable_outcomes[i], nboots=1000, seed=i)
        print(f"  K = {K:.2f}, fraction stable = {fracs[i]:.3f}+-{frac_errs[i]:.3f}")
    end = time.time()
    print(f'took {(end-start)/60} min, {(end-start)/total_tasks} sec per task.')

    # get K50 
    for idx in range(len(fracs) - 1):
        f1, f2 = fracs[idx], fracs[idx+1]
        if (f1 >= 0.5 >= f2) or (f1 <= 0.5 <= f2):
            f_low = f1; f_high =f2
            K_low = Ks[idx]; K_high = Ks[idx+1]
            idx_low = idx
            K50 = K_high - (K_high-K_low)/(f_high-f_low)*(f_high-0.5)

    # get K50 error 
    K50_stderr = lvf.bootstrap_interp_error(K_low, K_high, stable_outcomes[idx_low], stable_outcomes[idx_low+1])



    filename = f"{filestart}_S{S}_{nruns}rpk.npz"
    #np.savez_compressed(filename, Ks=Ks, fracs=fracs, frac_errs = frac_errs, S=S)
    np.savez_compressed(filename, Ks=Ks, fracs=fracs, frac_errs = frac_errs, K50=K50, K50_err = K50_stderr, S=S)
    print(f"\nData ( Ks, fracs, frac_errs, K50, K50_err, S) saved successfully to: {filename}")

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
    if nruns < 200: chunksize = 2
    else: chunksize = 5
    # run tasks in parallel:
    with concurrent.futures.ProcessPoolExecutor(max_workers=maxworkers) as executor:
        stables = list(executor.map(_worker_stable_check, tasks, chunksize=chunksize))
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
    if nruns < 200: chunksize = 2
    else: chunksize = 5
    # run tasks in parallel:
    with concurrent.futures.ProcessPoolExecutor(max_workers=maxworkers) as executor:
        stables = list(executor.map(_worker_stable_check_Bind, tasks, chunksize=chunksize))
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

def _worker_map_point(args):
    irho, jdelta, rho, delta, S, C, nruns, Kguess = args
    start = time.time()

    with open(os.devnull, 'w') as f, redirect_stdout(f):
        # NOTE: Ensure evaluate_oneK_stablefrac runs SYNCHRONOUSLY inside this function
        K50, K50_err, _ = lvf.find_K50_threshold_rho_delta(
            S, C, rho, delta, nruns, K_guess=Kguess
        )
        
    runtime = time.time() - start
    return irho, jdelta, rho, delta, K50, K50_err, runtime

def generate_rhodelta_map(S, C, rhos, deltas, nruns, filestart=None, maxworkers=None):
    '''
    Find the 50% stable threshold for many values in a rho vs delta map.
    S = number of species 
    C = connectance 
    rhos = rho values to use
    deltas = delta values to use 
    nruns = number of runs tried at each K 
    filestart = 
    maxworkers = max number of CPU cores to use in parallel. If None, will use all available cores.  
    '''
    if maxworkers is None: 
        maxworkers = os.cpu_count()
    nrhos = len(rhos)
    ndeltas = len(deltas)

    K50s = np.zeros((nrhos, ndeltas)) 
    K50_errs = np.zeros((nrhos, ndeltas)) 
    runtimes = np.zeros((nrhos, ndeltas))

    print(f"Starting heatmap sweep for S={S}, {nrhos} rhos, {ndeltas} deltas.")
    print(f"Using {maxworkers} CPU cores.")
    start_total = time.time()

    # array of all tasks for parallelization 
    tasks = []
    for irho, rho in enumerate(rhos):
        Kguess = 2/((1+3*rho)**(-0.5)) if rho > -0.25 else 5.0
        for jdelta, delta in enumerate(deltas):
            tasks.append((irho, jdelta, rho, delta, S, C, nruns, Kguess))

    total_tasks = len(tasks)
    print(f'{total_tasks} tasks to complete.')
    completed = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=maxworkers) as executor:
        results = executor.map(_worker_map_point, tasks, chunksize=1)
        for irho, jdelta, rho, delta, K50, K50_err, rt in results:
            K50s[irho, jdelta] = K50
            K50_errs[irho, jdelta] = K50_err
            runtimes[irho, jdelta] = rt
            completed += 1
            print(f"[{completed:3d}/{total_tasks}] rho={rho:.2f}, delta={delta:.2e} -> K50={K50:.3f} +- {K50_err:.3f} ({rt:.1f}s)")

    if filestart:
        output_dir = os.path.dirname(filestart)
        if output_dir:  
            os.makedirs(output_dir, exist_ok=True)
        filename = f"{filestart}_S{S}_{nruns}rpk.npz"
        np.savez_compressed(
            filename, deltas=deltas, rhos=rhos, 
            K50s=K50s, K50_errs=K50_errs, runtimes=runtimes, 
            nruns=nruns, S=S, C=C
        )
        print(f"Data saved successfully to: {filename}")

    return K50s, K50_errs, runtimes
    











    