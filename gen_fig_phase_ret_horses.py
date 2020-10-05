"""
NAME: gen_fig_phase_ret_horses.py
AUTHOR: ANONYMOUS
AFFILIATION: ANONYMOUS
DATE MODIFIED: 10 June 2020
DESCRIPTION: Loads a CIFAR-10 image and sets up a phase retrieval problem
             for it and then solves this problem using conic descent (with and
             without the greedy heuristic) and the conditional gradient method
             with an overestimate of the bound on the optimal solution.  It
             breaks after 1500 matrix multiplications and displays the
             reconstructed images.
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

# Implementation of conditional gradient
import cg

# Phase retrieval setup helper functions
import setup_phase_retrieval

if __name__ == '__main__':

    # Load CIFAR10 image (7 is the horse!)
    img_gray = setup_phase_retrieval.get_cifar10_image(7)

    # Display the original image
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(img_gray, cmap='gray')
    plt.axis('off')
    plt.title('original image')

    # Generate problem instance
    np.random.seed(0) # For consistency
    f, gradf, opG, adjG, g, n = setup_phase_retrieval.get_data(img=img_gray, k=10, gamma=5.0e-5, SNR=20.0)

    # Initialize variables
    X = matrix_variable.MatrixVariableSketch(n, 3)
    y = -g

    # Options
    opts = dict(max_iters=100,
                eigmin_tol=lambda kk: 1.0e-1,
                max_matmults=1500,
                disp_output=lambda kk: False,
                disp_head_foot=False)

    # --------------------------------
    # SOLVE USING CONDITIONAL GRADIENT
    # --------------------------------

    h = np.sum(g[:-1]) * 10 # 10x overestimate
    cg.solve(f, gradf, opG, adjG, g, X, y, h, **opts)

    # Recover from sketch and resolve sign ambiguity
    U, d, err = X.get_approx_evd()
    print('CONDITIONAL GRADIENT:')
    print('  Sketch recovery error:', err)
    img_recovered = (d[0]**0.5 * U[:, 0]).reshape(32, 32)
    if la.norm(img_gray - img_recovered) > la.norm(img_gray + img_recovered):
        img_recovered *= -1
    print('  Relative error:       ', la.norm(img_gray - img_recovered) / la.norm(img_gray))

    # Display recovered image
    plt.subplot(2, 2, 2)
    plt.imshow(img_recovered, cmap='gray')
    plt.axis('off')
    plt.title('CG')

    # -------------------------
    # SOLVE USING CONIC DESCENT
    # -------------------------

    # Reset X and y
    X.update(0.0, 0.0, np.zeros(n))
    y = -g

    # Solve
    cdpsd.solve(f, gradf, opG, adjG, g, X, y, r=3, **opts)

    # Recover from sketch and resolve sign ambiguity
    U, d, err = X.get_approx_evd()
    print('CONIC DESCENT:')
    print('  Sketch recovery error:', err)
    img_recovered = (d[0]**0.5 * U[:, 0]).reshape(32, 32)
    if la.norm(img_gray - img_recovered) > la.norm(img_gray + img_recovered):
        img_recovered *= -1
    print('  Relative error:       ', la.norm(img_gray - img_recovered) / la.norm(img_gray))

    # Display recovered image
    plt.subplot(2, 2, 3)
    plt.imshow(img_recovered, cmap='gray')
    plt.axis('off')
    plt.title('CD')

    # --------------------------------------------
    # SOLVE USING CONIC DESCENT (GREEDY HEURISTIC)
    # --------------------------------------------

    opts['use_greedy'] = lambda kk: kk % 100 == 0
    opts['greedy_tol'] = lambda kk: 1.0e-5
    opts['max_greedy_iters'] = lambda kk: 100

    # Reset X and y
    X.update(0.0, 0.0, np.zeros(n))
    y = -g

    # Solve
    cdpsd.solve(f, gradf, opG, adjG, g, X, y, r=3, **opts)

    # Recover from sketch and resolve sign ambiguity
    U, d, err = X.get_approx_evd()
    print('CONIC DESCENT (WITH GREEDY STEPS):')
    print('  Sketch recovery error:', err)
    img_recovered = (d[0]**0.5 * U[:, 0]).reshape(32, 32)
    if la.norm(img_gray - img_recovered) > la.norm(img_gray + img_recovered):
        img_recovered *= -1
    print('  Relative error:       ', la.norm(img_gray - img_recovered) / la.norm(img_gray))

    # Display recovered image
    plt.subplot(2, 2, 4)
    plt.imshow(img_recovered, cmap='gray')
    plt.axis('off')
    plt.title('CD + greedy')

    # Show the result
    plt.show()
