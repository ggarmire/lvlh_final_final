import numpy as np
import concurrent.futures
import matplotlib.pyplot as plt
import time

import lvlh_functions as lvf

def main():

    seed = np.random.randint(0, 1000)
    seed = 1
    print(f'seed={seed}')

    S = 200

    delta = 1000



    C = 1

    Kset = 1

    sigma = Kset * (S*C)**(-0.5)
    K = sigma * (S*C)**0.5

    B = lvf.B_ind(S, C, sigma, L=3, seed=seed)
    R = lvf.R_star_3stage_delta(B, delta)
    Jac = lvf.Jacobian(B, R)
    Jeigs = np.linalg.eigvals(Jac)

    # find Z from block matrix structure to compare eigenvalues:
    Z = B[1::2, 1::2] - B[::2, 1::2]
    Zish = B[1::2, 1::2] - B[::2, 1::2]
    diag_Z = -np.diag(R, k=1)[::2] - np.diag(R, k=-1)[::2]
    diag_Z_delta = np.full(S, -2*delta)
    np.fill_diagonal(Zish, diag_Z_delta)
    np.fill_diagonal(Z, diag_Z)

    Zeigs = np.linalg.eigvals(Z)
    Zisheigs = np.linalg.eigvals(Zish)

    # do W, X, Y:
    W = B[::2, ::2] + B[::2, 1::2]
    np.fill_diagonal(W, -2)

    X = B[::2, 1::2]
    Xdiag = 1 + np.diag(R, k=1)[::2]
    np.fill_diagonal(X, Xdiag)

    Y = B[1::2, ::2] + B[1::2, 1::2] - B[::2, ::2] - B[::2, 1::2]
    np.fill_diagonal(Y, -2)

 

    print(f"S={S},  delta={delta}, K={K}")

    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(14, 8))
    colors = np.where(np.real(Jeigs) > 0, 'red', 'green')
    ax1.scatter(np.real(Jeigs), np.imag(Jeigs),c=colors, s=10, label='eigenvalues of J')
    ax1.plot(np.real(Zeigs), np.imag(Zeigs), 'x', c='blue', ms=4, label='det(D)=0')
    ax1.legend()
    #ax1.scatter(np.real(Zisheigs), np.imag(Zisheigs), c='red', s=10)

    ax1.set_xlim((np.min(np.real(Jeigs))-2, -2*delta+3))
    ax1.grid('true')
    ax1.set_xlabel(r'Re($\lambda_{J}$)')
    ax1.set_ylabel(r'Im($\lambda_{J}$)')

    ax2.scatter(np.real(Jeigs), np.imag(Jeigs),c=colors, s=10)
    ax2.set_xlim((-2-2*K, -2+2*K))
    ax2.grid('true')
    ax2.set_xlabel(r'Re($\lambda_{J}$)')
    #ax2.set_ylabel(r'Im($\lambda_{J}$)')

    plt.figure()
    plt.scatter(np.real(Jeigs), np.imag(Jeigs),c=colors, s=10, label='eigenvalues of J')
    


    plt.show()



if __name__ == "__main__":
    main()


    