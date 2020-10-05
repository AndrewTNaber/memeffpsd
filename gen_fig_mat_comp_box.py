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
             It does this a large number of times and outputs boxplot.
"""

# Standard imports
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import pandas as pd
import sys

# Solves CD over PSD cone in a memory-efficient manner
import cdpsd

# Classes for storing the matrix variable
import matrix_variable

# Phase retrieval setup helper functions
import setup_psd_completion

def sample():

    # Generate problem instance for some V
    f, gradf, opG, adjG, g, n, pstar = setup_psd_completion.get_data(V=np.random.randn(100, 3), d=10, p=0.1)

    # Initialize variables
    X = matrix_variable.MatrixVariableEmpty(n)
    y = -g

    # Options
    opts = dict(max_iters=1000000,
                max_matmults=50000,
                eigmin_method='ARPACK',
                max_eigmin_iters=lambda kk: int((kk + 1)**0.25 * np.log(n)),
                eigmin_tol=lambda kk: 1.0e-1,
                output=['iteration', 'obj', 'dfeas', 'slack', 'time', 'eigmin_matmult'],
                use_logging=True,
                flog_name='./logs/latest_log_demo_psd_completion_rand_hist_meas.csv',
                disp_output=lambda kk:False,
                disp_head_foot=False)

    # Solve problem without greedy heuristic
    cdpsd.solve(f, gradf, opG, adjG, g, X, y, 1, **opts)

    # Solve problem using greedy heuristic for different ranks
    ranks = [2, 3, 4, 5]
    for rr in ranks:
        X = matrix_variable.MatrixVariableEmpty(n)
        y = -g
        opts['use_greedy'] = lambda kk: kk % 100 == 0
        opts['greedy_tol'] = lambda kk: 1.0e-8
        opts['flog_name'] = './logs/latest_log_demo_psd_completion_rand_hist_meas_greedy' + str(rr)
        cdpsd.solve(f, gradf, opG, adjG, g, X, y, rr, **opts)

    # Load logs
    df = pd.read_csv('./logs/latest_log_demo_psd_completion_rand_hist_meas.csv')
    dfrank = {}
    for rr in ranks:
        dfrank[rr] = pd.read_csv('./logs/latest_log_demo_psd_completion_rand_hist_meas_greedy' + str(rr))

    # Make measurements
    data_rank = {}
    for rr in ranks:
        bm_obj = dfrank[rr]['obj'].iloc[1] - pstar
        bm_matmults = np.sum(dfrank[rr]['eigmin_matmult'][:2]) + np.sum(dfrank[rr]['greedy_matmult'][:2])
        cd_matmults = np.cumsum(df['eigmin_matmult'])
        cd_obj = 0.0
        for ii in range(len(cd_matmults)):
            if cd_matmults.iloc[ii] >= bm_matmults:
                cd_obj = df['obj'].iloc[ii] - pstar
                break
        data_rank['bm_minus_cd' + str(rr)] = np.log10(bm_obj) - np.log10(cd_obj)
        data_rank['bm_minus_cdg' + str(rr)] = np.log10(bm_obj) - np.log10(dfrank[rr]['obj'].iloc[-1] - pstar)

    for key in data_rank.keys():
        print(key, data_rank[key])

    return data_rank

if __name__ == '__main__':

    num_samps = int(sys.argv[1])

    np.random.seed(0) # For consistency

    meas = []
    for ii in range(num_samps):
        print('RUNNING SAMPLE {0}'.format(ii))
        meas.append(sample())
    df = pd.DataFrame(meas)

    cmap = plt.get_cmap('tab10')

    labels = ['$r=2$', '$3$', '$4$', '$5$']

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(6, 3))

    # PBM - CD
    bplot1 = ax1.boxplot([df['bm_minus_cd' + str(ii + 1)] for ii in range(1, 5)],
                         sym='.',
                         medianprops=dict(color='k'),
                         vert=True,  # vertical box alignment
                         patch_artist=True,  # fill with color
                         labels=labels)  # will be used to label x-ticks

    # PBM - CD with greedy heuristic
    bplot2 = ax2.boxplot([df['bm_minus_cdg' + str(ii + 1)] for ii in range(1, 5)],
                         sym='.',
                         medianprops=dict(color='k'),
                         vert=True,  # vertical box alignment
                         patch_artist=True,  # fill with color
                         labels=labels)  # will be used to label x-ticks

    # fill with colors
    colors = [cmap.colors[ii - 1] for ii in range(1, 5)]
    for bplot in (bplot1, bplot2):
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)

    # adding horizontal grid lines
    ax1.yaxis.grid(True)
    ax1.set_xlabel('(a)')
    ax2.yaxis.grid(True)
    ax2.set_xlabel('(b)')
    plt.tight_layout()
    plt.show()



