"""
NAME: setup_psd_completion.py
AUTHOR: ANONYMOUS
AFFILIATION: ANONYMOUS
DATE MODIFIED: 10 June 2020
DESCRIPTION: Generates a PSD matrix completion problem with nonuniform sampling.
             Specifically, it samples the upper left block of the matrix 

             V V' + [I 0; 0 0] 

             densely and the remainder of the blocks sparsely.  The matrix V
             has shape n x m and is drawn from standard normal distribution.
             Plots the covergence.
"""

# Standard imports
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import pandas as pd

# Solves CD over PSD cone in a memory-efficient manner
import cdpsd

# Classes for storing the matrix variable
import matrix_variable

# Phase retrieval setup helper functions
import setup_psd_completion

if __name__ == '__main__':

    np.random.seed(2) # For consistency

    # Generate problem instance for some V
    f, gradf, opG, adjG, g, n, pstar = setup_psd_completion.get_data(V=np.random.randn(100, 3), d=10, p=0.1)

    # Initialize variables
    X = matrix_variable.MatrixVariableEmpty(n)
    y = -g

    # Options
    opts = dict(max_iters=1000000,
                max_matmults=50000,
                max_eigmin_iters=lambda kk: int((kk + 1)**0.25 * np.log(n)),
                eigmin_tol=lambda kk: 1.0e-1,
                disp_output=lambda kk: False,
                disp_head_foot=False,
                use_logging=True,
                flog_name='./logs/latest_log_demo_psd_completion_rand.csv')

    # Solve problem without greedy heuristic
    cdpsd.solve(f, gradf, opG, adjG, g, X, y, **opts)

    # Solve problem using greedy heuristic for different ranks
    ranks = [2, 3, 4, 5]
    for rr in ranks:
        X = matrix_variable.MatrixVariableEmpty(n)
        y = -g
        opts['use_greedy'] = lambda kk: kk % 100 == 0
        opts['greedy_tol'] = lambda kk: 1.0e-8
        opts['flog_name'] = './logs/latest_log_demo_psd_completion_rand_greedy' + str(rr) + '.csv'
        cdpsd.solve(f, gradf, opG, adjG, g, X, y, r=rr, **opts)

    # Load logs
    df = pd.read_csv('./logs/latest_log_demo_psd_completion_rand.csv')
    dfrank = {}
    for rr in ranks:
        dfrank[rr] = pd.read_csv('./logs/latest_log_demo_psd_completion_rand_greedy' + str(rr) + '.csv')

    # Plot results
    plt.figure()
    plt.semilogy(np.cumsum(df['eigmin_matmult']), df['obj'] - pstar, label='CD', color='black')
    for rr in ranks:
        plt.semilogy((np.cumsum(dfrank[rr]['eigmin_matmult']) + np.cumsum(dfrank[rr]['greedy_matmult']))[1:], dfrank[rr]['obj'][1:] - pstar, label='CD with greedy,' + '$r = {0}$'.format(rr))
        plt.scatter(np.sum(dfrank[rr]['eigmin_matmult'][:2]) + np.sum(dfrank[rr]['greedy_matmult'][:2]), dfrank[rr]['obj'][1] - pstar, marker='o')
    plt.xlim([-1000, 50000])
    plt.xlabel('matrix multiplications')
    plt.ylabel(r'$f(X_k) - p^\star$')
    plt.legend(loc='upper right')
    plt.grid()
    plt.tight_layout()
    plt.show()





