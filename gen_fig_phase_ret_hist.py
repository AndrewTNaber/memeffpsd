"""
NAME: gen_fig_phase_ret_hist.py
AUTHOR: ANONYMOUS
AFFILIATION: ANONYMOUS
DATE MODIFIED: 10 June 2020
DESCRIPTION: Repeatedly loads a CIFAR-10 image and sets up a phase retrieval problem
             for it and then solves this problem using conic descent (with and
             without the greedy heuristic) and the conditional gradient method
             with an overestimate of the bound on the optimal solution.  It plots
             a histogram of the fraction of matrix multiplications of CG required
             by CD to hit CG's optimal value after 500 iterations.
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

# Implementation of conditional gradient
import cg

# Phase retrieval setup helper functions
import setup_phase_retrieval

def get_meas_from_image(img_number):

    # Load CIFAR10 image
    img_gray = setup_phase_retrieval.get_cifar10_image(img_number)

    # Generate phase retrieval problem instance
    f, gradf, opG, adjG, g, n = setup_phase_retrieval.get_data(img=img_gray, k=10, gamma=5.0e-5, SNR=20.0)

    # Initialize variables
    X = matrix_variable.MatrixVariableEmpty(n)
    y = -g

    # Options
    opts = dict(max_iters=500,
                eigmin_tol=lambda kk: 1.0e-1,
                disp_output=lambda kk: False,
                disp_head_foot=False,
                use_logging=True)

    # --------------------------------
    # SOLVE USING CONDITIONAL GRADIENT
    # --------------------------------

    opts['flog_name'] = './logs/latest_log_phase_ret_cg.csv'

    h = np.sum(g[:-1]) * 10 # 10x overestimate
    cg.solve(f, gradf, opG, adjG, g, X, y, h, **opts)
    print('  ...done with CG.')
    df_cg = pd.read_csv(opts['flog_name'])

    # -------------------------
    # SOLVE USING CONIC DESCENT
    # -------------------------

    opts['flog_name'] = './logs/latest_log_phase_ret_cd.csv'

    # Reset X and y
    X.update(0.0, 0.0, np.zeros(n))
    y = -g

    # Solve
    cdpsd.solve(f, gradf, opG, adjG, g, X, y, **opts)
    print('  ...done with CD.')
    df_cd = pd.read_csv(opts['flog_name'])

    # --------------------
    # RETURN RELEVANT DATA
    # --------------------
    cg_obj = df_cg['obj'].iloc[-1]
    cg_num_matmults = np.sum(df_cg['eigmin_matmult'])

    cd_num_matmults = 0
    cd_num_iters = 0
    for kk in range(500):
        if df_cd['obj'].iloc[kk] <= cg_obj:
            cd_num_iters = kk
            cd_num_matmults = np.sum(df_cd['eigmin_matmult'].iloc[:kk])
            break

    print('  fraction: {0}'.format(cd_num_matmults / cg_num_matmults))

    greedy_num_matmults = 0
    greedy_num_iters = 0

    return cd_num_iters, greedy_num_iters, cd_num_matmults, greedy_num_matmults, cg_num_matmults


if __name__ == '__main__':

    num_samps = int(sys.argv[1])

    np.random.seed(0) # For consistency

    meas = []
    for ii in range(num_samps):
        print('RUNNING SAMPLE {0}'.format(ii))
        cdi, cdgi, cdm, cdgm, cgm = get_meas_from_image(ii)
        meas.append({'CD iters':cdi, 
                     'CD + greedy iters':cdgi,
                     'CD matmults':cdm,
                     'CD + greedy matmults':cdgm,
                     'CG matmults':cgm})

    df = pd.DataFrame(meas)

    # ---------------
    # Display results
    # ---------------
    plt.figure()
    plt.hist(df['CD matmults'] / df['CG matmults'], color='lightgray', ec='black')
    plt.xlabel('fraction of CG matrix multiplications')
    plt.ylabel('frequency')
    plt.tight_layout()
    plt.show()







