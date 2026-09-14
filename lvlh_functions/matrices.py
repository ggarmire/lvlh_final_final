import numpy as np

# initial condition for system evolution

def init_cond(S, L, spread, seed):
    '''
    Draws SxL random numbers between (1-spread) and (1+spread) 
    to act as the initial conditions for evolving the gLV system.

    S = number of species 
    L = number of stages per species 
    spread = spread of uniform distribution from which random variables are drawn. <= 1. 
    seed = random number generator seed 
    '''
    np.random.seed(seed)
    return np.random.uniform(1-spread, 1+spread, S*L)


# interaction matrices 

def A_onestage(S, C, sigma, seed):
    '''
    Generate interaction matrix for system with S species, one stage. Diagonals are -1.
    S = number of species (int)
    C = connectance (probability of nonzero entry in A)
    sigma = std of nonzero entries in A
    seed = random number seed used to generate A

    returns A: SxS matrix 
    '''
    np.random.seed(seed)
    A = np.random.normal(0, sigma, (S,S))       # fill A with random iid variables 
    zero_mask = np.random.random(size=(S, S)) > C       # randomly select entries in A to be zero with probability 1-C
    A[zero_mask] = 0.0      # mask selected 0s
    np.fill_diagonal(A, -1.0)       # fill diagonal with -1 for intraspecific competition
    return A

def B_rho(S, C, sigma, seed, L, rho):
    '''
    Generate an interaction matrix for S species, L stages. On-diagonal LxL blocks are -1. 
    S = number of species 
    C = connectance
    L = number of stages 
    sigma = std of nonzero entries in A
    rho = correlation between inter-species interactions of different stages 
    seed = random number seed used to generate A

    returns B: LSxLS matrix
    '''
    np.random.seed(seed)
    L2 = L*L
    B = np.zeros((L*S, L*S))
    cov_mat = sigma**2 * (rho * np.ones((L2, L2)) + (1 - rho) * np.eye(L2))

    for i in range(S):
        for j in range(S):
            block_row = slice(L*i, L*i+L)
            block_col = slice(L*j, L*j+L)

            if i == j: B[block_row, block_col] = -1
            else:
                val = np.random.rand(1)
                if val > C: B[block_row, block_col] = np.zeros((L,L))
                else:
                    x = np.random.multivariate_normal(mean=np.zeros(L2), cov=cov_mat)
                    B[block_row, block_col] = x.reshape(L,L)
    return B


def B_ind(S, C, sigma, L, seed):
    '''
    Generate interaction matrix for system with S species, L stages, uncorrelated intra-species inter-stage interactions.
    only true diagonal entries of Bind are set to 0. Thus, set the same way as an A matrix for an SxL system
    S = number of species (int)
    L = number of stages 
    C = connectance (probability of nonzero entry in A)
    sigma = std of nonzero entries in A
    seed = random number seed used to generate A
    returns A: SxS matrix 
    '''
    np.random.seed(seed)
    B = np.random.normal(0, sigma, (S*L,S*L))
    zero_mask = np.random.random(size=(S*L, S*L)) > C
    B[zero_mask] = 0.0
    np.fill_diagonal(B, -1.0)
    return B


# demographic matrix / growth vector 

def r_star(A):
    '''
    rowth rate vector for single-stage case, that will give x* = 1
    A = interaction matrix (one stage)
    '''
    S = A.shape[0]
    return -np.dot(A, np.ones(S))

def R_star_2stage_delta(B, delta): 
    '''
    this generates the R matrix for 2 stage system with x*=1. 
    B = interaction matrix 
    rseed sets random number generation 
    delta is the minimum value of any phi, gamma, or -mu. 
    '''
    S = int(B.shape[0]/2)
    One = np.ones(B.shape[0])
    B_rs = np.dot(B, One)
    deltas = np.full(S, delta)
    # child params first:
    fmins = np.maximum(0, -B_rs[0::2])        # minimum value each f could have 
    fs = fmins + deltas                 # now the minimum value is delta
    mucs = -B_rs[0::2] - fs
    # adult params:
    gmins = np.maximum(0, -B_rs[1::2])        # minimum value each g could have 
    gs = gmins + deltas                 # now the minimum value is delta
    muas = -B_rs[1::2] - gs
    # format R matrix with on-diagonal 2x2 blocks 
    R = np.zeros(B.shape)
    evens = np.arange(0, 2*S, 2)
    odds = np.arange(1, 2*S, 2)
    R[evens, evens] = mucs  
    R[evens, odds] = fs     
    R[odds, evens] = gs    
    R[odds, odds] = muas

    return R

def R_star_3stage_delta(B, delta): 
    '''
    this generates the R matrix for 3 stage system with x*=1. 
    B = interaction matrix (LSxLS)
    rseed sets random number generation 
    delta is the minimum value of any phi, gamma, or -mu. 
    '''
    S = int(B.shape[0]/3)
    One = np.ones(B.shape[0])
    B_rs = np.dot(B, One)
    deltas = np.full(S, delta)

    # indices of each stage 
    idx_stage1 = np.arange(0, 3 * S, 3)
    idx_stage2 = np.arange(1, 3 * S, 3)
    idx_stage3 = np.arange(2, 3 * S, 3)

    # stage 1:
    fmins = np.maximum(0, -B_rs[idx_stage1])
    phis = fmins + deltas
    mu1s = -B_rs[idx_stage1] - phis

    # stage 2
    g1mins = np.maximum(0, -B_rs[idx_stage2])
    gamma1s = g1mins + deltas
    mu2s = -B_rs[idx_stage2] - gamma1s

    # stage 3
    g2mins = np.maximum(0, -B_rs[idx_stage3])
    gamma2s = g2mins + deltas
    mu3s = -B_rs[idx_stage3] - gamma2s 

    # make R
    R = np.zeros(B.shape)
    R[idx_stage1, idx_stage1] = mu1s
    R[idx_stage2, idx_stage2] = mu2s
    R[idx_stage3, idx_stage3] = mu3s

    R[idx_stage1, idx_stage3] = phis     
    R[idx_stage2, idx_stage1] = gamma1s 
    R[idx_stage3, idx_stage2] = gamma2s  

    return R



