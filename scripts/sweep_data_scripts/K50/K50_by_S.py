'''
For a set of S values, find the K value at which 50% of runs are stable (using large Delta limit)
'''
import numpy as np
import matplotlib.pyplot as plt
import lvlh_functions as lvf
import os

def main():
    Ss = [100, 50, 25, 10]
    C = 1.0
    rho = 0
    L = 2
    delta = 1000

    nruns = 200

    # arguments for B and R: 
    extra_args = {'L': L} 
    R_args = {'delta': delta}


if __name__ == "__main__":
    main()
