

# Standard imports
import time
import numpy as np
import scipy.linalg as la
import scipy.optimize as opt
import matplotlib.pyplot as plt

# Functions for computing the approximate minimum eigenvalue and associated
# (normalized) eigenvector
import min_eigvec

# For prettily printing the output (will be removed in the future)
import pandas as pd
pd.options.display.float_format = '{:12,.3e}'.format

_PHI = 1.61803398875 # Golden ratio for use in _ls_bisection

def _ls_bisection(f, lb=0.0, ub=1.0, tol=1.0e-8):
    """Minimizes f on [lb, ub] using golden section search"""
    
    a = lb
    b = ub
    c = b - (b - a) / _PHI
    d = a + (b - a) / _PHI
    while abs(c - d) > tol:
        if f(c) < f(d):
            b = d
        else:
            a = c
        c = b - (b - a) / _PHI
        d = a + (b - a) / _PHI
    return (b + a) / 2

def solve(f, gradf, opG, adjG, g, X, y, h, **kwargs):
    """
    Solves the problem

        min. f(opG(X) - g)
        s.t. X is positive semidefinite
             tr(X) <= h

    using conditional gradient.
    """

    n = X.size()

    # --------------------------------------
    # Check consistency of the problem input
    # --------------------------------------

    # Dimensions

    # Adjoint

    # Gradient

    # -----------------------------------------------
    # Set problem parameters using defaults and input
    # -----------------------------------------------

    # Maximum number of iterations
    if 'max_iters' in kwargs:
        max_iters = kwargs['max_iters']
    else:
        max_iters = 5000

    # Method to find approximate minimum eigenvalue (and associated normalized eigenvector)
    if 'eigmin_method' in kwargs:
        if kwargs['eigmin_method'] == 'ARPACK':
            eigmin = min_eigvec.min_eigvec_ARPACK
        elif kwargs['eigmin_method'] == 'shift_invert':
            eigmin = min_eigvec.min_eigvec_shift_invert
        elif kwargs['eigmin_method'] == 'Lanczos':
            eigmin = min_eigvec.min_eigvec_Lanczos
        elif kwargs['eigmin_method'] == 'power':
            eigmin = min_eigvec.min_eigvec_power
        else:
            raise ValueError('invalid eigmin_method')
    else:
        eigmin = min_eigvec.min_eigvec_ARPACK

    # Method returning the maximum number of iterations for eigmin
    if 'max_eigmin_iters' in kwargs:
        max_eigmin_iters = kwargs['max_eigmin_iters']
    else:
        max_eigmin_iters = lambda kk: 200

    # Method returning eigmin tolerance for termination
    if 'eigmin_tol' in kwargs:
        eigmin_tol = kwargs['eigmin_tol']
    else:
        eigmin_tol = lambda kk: 1.0e-4

    # Method returning when to display to stdout
    if 'disp_output' in kwargs:
        disp_output = kwargs['disp_output']
    else:
        disp_output = lambda kk: kk % (max_iters // 10) == 0

    # What measurements to display to stdout
    if 'output' in kwargs:
        output = kwargs['output']
    else:
        output = ['iteration', 'obj', 'gap']

    # Bisection line search tolerance
    if 'ls_tol' in kwargs:
        ls_tol = kwargs['ls_tol']
    else:
        ls_tol = 1.0e-8

    # Maximum number of matrix multiplications
    if 'max_matmults' in kwargs:
        max_matmults = kwargs['max_matmults']
    else:
        max_matmults = np.inf

    # Whether or not to log the iterations
    if 'use_logging' in kwargs:
        use_logging = kwargs['use_logging']
    else:
        use_logging = False

    # File name for the log
    if 'flog_name' in kwargs:
        flog_name = kwargs['flog_name']
    else:
        flog_name = './latest_log_cg.csv'

    # -----------------------
    # Initialize measurements
    # -----------------------
    
    meas = {'iteration':0,
            'obj':0.0,
            'gap':0.0,
            'eigmin_matmult':0,
            'eigmin_time':0.0,
            'eigmin_maxiter':0,
            'eigmin_tol':0.0,
            'time':0.0}
    
    # -------------
    # Start logging
    # -------------

    if use_logging:
        flog = open(flog_name, 'w')
        flog.write(','.join(meas) + '\n')

    # --------------
    # Main iteration
    # --------------

    start_time = time.clock()
    total_matmults = 0

    for kk in range(max_iters):

        # --------------------
        # Conditional gradient
        # --------------------

        meas['iteration'] = kk
        meas['obj'] = f(y)

        # Approximate minimum eigenvalue (and associated normalized eigenvector)
        t0 = time.clock()
        q_min, lambda_min, matmultcnt = eigmin(adjG(gradf(y)), max_iters=max_eigmin_iters(kk), tol=eigmin_tol(kk))
        t1 = time.clock()
        
        meas['eigmin_matmult'] = matmultcnt
        meas['eigmin_time'] = t1 - t0
        meas['eigmin_maxiter'] = max_eigmin_iters(kk)
        meas['eigmin_tol'] = eigmin_tol(kk)

        total_matmults += matmultcnt

        if lambda_min <= 0.0:
            z = h * opG(q_min)
        else:
            z = np.zeros_like(y)

        meas['gap'] = np.vdot(gradf(y), y - z)

        # Bisection line search
        fls = lambda s: f(s * (y + g) + (1 - s) * z - g)
        s = _ls_bisection(fls, tol=ls_tol)

        # Update X and y
        if lambda_min <= 0.0:
            X.update(s, 1 - s, h**0.5 * q_min)
        else:
            X.update(s, 0.0, np.zeros(n))
        y = s * (y + g) + (1 - s) * z - g

        # Update log
        meas['time'] = time.clock() - start_time
        if use_logging:
            flog.write(','.join(str(meas[key]) for key in meas) + '\n')

        # Break if total matrix multiplications exceeds maximum
        if total_matmults > max_matmults:
            break

        # --------------
        # Display update
        # --------------

        if disp_output(kk):
            df = pd.DataFrame([meas])
            line = df[output].to_string(index=False)
            print(line.split('\n')[1]) if kk != 0 else print(line)

    # Close log file
    if use_logging:
        flog.close()

    return f(y)