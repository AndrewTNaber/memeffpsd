# NAME: cdpsd.py
# AUTHOR: ANONYMOUS
# AFFILIATION: ANONYMOUS
# DATE MODIFIED: 10 June 2020
# DESCRIPTION: The main solve function for applying conic descent to problems
#              over the positive semidefinite cone in a memory-efficient manner.
# TO DO:
# 1. Require that eigmin(A) return lambda_min = q_min' * A * q_min
# 2. scipy.optimize has a method to check gradients for you!
# 3. Don't use pandas to do the pretty printing

# Standard imports
import time
import numpy as np
import scipy.linalg as la
import scipy.optimize as opt

# Functions for computing the approximate minimum eigenvalue and associated
# (normalized) eigenvector
import min_eigvec

# For prettily printing the output (will be removed in the future)
import pandas as pd
pd.options.display.float_format = '{:12,.3e}'.format

_PHI = 1.61803398875 # Golden ratio for use in _ls_bisection

def _ls_bisection(f, lb=0.0, ub=1.0, tol=1.0e-8):
    """Minimizes f on [lb, ub] using the golden section search."""

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

def _check_prob_dimensions(f, gradf, opG, adjG, g, X, y, r):
    """Checks the dimensions of the provided problem."""

    m, = y.shape
    n = X.size()

    u = np.random.randn(m)
    V = np.random.randn(n, r)

    if not np.isscalar(f(u)):
        raise ValueError('f does not return a scalar')
    if gradf(u).shape != (m,):
        raise ValueError('gradf does not return a vector of size m')
    if opG(V).shape != (m,):
        raise ValueError('opG does not return a vector of size m')
    if adjG(u).shape != (n, n):
        raise ValueError('adjG does not return a matrix of size (n, n)')
    if g.shape != (m,):
        raise ValueError('g is not a vector of size m')
    if r > n:
        raise ValueError('r must be less than n')

    return m, n

def _check_prob_adjoint(opG, adjG, m, n, r):
    """Checks the adjointness of opG and adjG."""

    for _ in range(5):
        u = np.random.randn(m)
        V = np.random.randn(n, r)
        t1 = opG(V) @ u
        t2 = np.trace(V.T @ (adjG(u) @ V))
        if np.abs(t1 - t2) > np.abs(t1) * 1.0e-12:
            raise ValueError('adjG is (likely) not the adjoint of opG')

def _check_prob_gradient(f, gradf, m):
    """Checks the correctness of f and gradf."""

    for _ in range(5):
        u = np.random.randn(m)
        du = 1.0e-8 * np.random.randn(m)
        t1 = f(u + du)
        t2 = f(u) + gradf(u) @ du
        if np.abs(t1 - t2) > np.abs(t1) * 1.0e-12:
            raise ValueError('gradf is (likely) not the gradient of f')

def solve(f, gradf, opG, adjG, g, X, y, r=1, **kwargs):
    """
    Solves the problem

        min. f(opG(X) - g)
        s.t. X is n x n positive semidefinite

    in a memory efficient manner using conic descent.

    Parameters
    ----------
    f : callable
        The objective function to be minimized

            ``f(y) -> float``

        where y is a 1-D array with shape (m,).
    gradf : callable
            The gradient of f

                ``gradf(y) -> array, shape (m,)``

            where y is a 1-D array with shape (m,).
    opG : callable
          The linear operator acting on the square root of symmetric n x n
          matrices

              ``opG(U) -> array, shape (m,)``

          where X = UU' is the underlying symmetric n x n matrix on which opG
          is being called.  Should be implemented as efficiently as possible by
          the user.
    adjG : callable
           The adjoint of the linear operator

                ``adjG(y) -> scipy.sparse.linalg.LinearOperator``

            or

                ``adjG(y) -> scipy.sparse matrix (CSC or CSR format)

            Should be implemented as efficiently as possible by the user.
    X : matrix_variable.MatrixVariable
        The object tracking properties (size, trace, ...) of the matrix
        variable.
    y : array, shape (m,)
        The auxiliary variable used to drive the iterations.  It is the user's
        responsibility to ensure that it equals opG(X) - g because this cannot
        be verified internally!
    r : int
        The rank of the greedy step.  Should be much less than n.

    Keyword Arguments
    -----------------
    max_iters : int
                Maximum number of conic descent iterations.
                Default: 5000
    tol : float
          Termination tolerance as measured in distance (w.r.t. spectral norm)
          of the gradient from PSD cone.
          Default: 1.0e-6
    eigmin_method : string
                    Method to find approximate minimum eigenvalue and
                    associated normalized eigenvector. Must be one of
                    'ARPACK', 'Lanczos', 'shift_invert', or 'power'.
                    Default: 'ARPACK'
    max_eigmin_iters : callable

                            ``max_eigmin_iters(kk) -> int``

                       Method returning the maximum number of iterations for
                       the approximate minimum eigenvalue computation at
                       iteration kk of conic descent.
                       Default: lambda kk: 200
    eigmin_tol : callable

                        `` eigmin_tol(kk) -> int``

                 Method returning eigmin tolerance for termination of
                 approximate minimum eigenvalue computation at iteration kk
                 of conic descent.
                 Default: lambda kk: 1.0e-4
    disp_output : callable

                  ``disp_output(kk) -> bool``

                  Method returning whether to display output at iteration
                  k of conic descent.
                  Default: lambda kk: kk % (max_iters // 10) == 0
    disp_head_foot: bool
                    Whether to display the header and footer.
                    Default: True
    output : list
             What measurements to display.  Can be any of
                1.  'iteration' - Conic descent iteration
                2.  'obj' - Objective value
                3.  'slack' - Complementary slackness violation
                4.  'dfeas' - Dual feasibility violation
                5.  'eigmin_matmult' - Eigmin matrix multiplications count
                6.  'eigmin_time' - Time for eigmin computation
                7.  'eigmin_maxiter' - Max eigmin iterations
                8.  'eigmin_tol' - Eigmin tolerance
                9.  'eigmin_success' - Was a descent direction found?
                10. 'use_greedy' - Whether to run greedy step
                11. 'greedy_status' - opt.OptimizeResult.status
                12. 'greedy_message' - opt.OptimizeResult.message
                13. 'greedy_fun' - opt.OptimizeResult.fun
                14. 'greedy_jac' - 2-norm of opt.OptimizeResult.jac
                15. 'greedy_nfev' - opt.OptimizeResult.nfev
                16. 'greedy_njev' - opt.OptimizeResult.njev
                17. 'greedy_nit' - opt.OptimizeResult.nit
                18. 'greedy_matmult' - Greedy matrix multiplications count
                19. 'greedy_time' - Time for greedy step
                20. 'greedy_maxiter' - Max greedy iterations
                21. 'greedy_tol' - Greedy tolerance
                22. 'greedy_success' - Was greedy step successful?
                23. 'time' - Total time for this iteration of conic descent
             Default: ['iteration', 'obj', 'dfeas', 'slack']
    use_backtrack : bool
                    Whether to use backtracking line search instead of exact
                    line search.
                    Default: False
    alpha : float
            Backtracking line search acceptable decrease parameter.
            Default: 0.3
    beta : float
           Backtracking line search refinement parameter
           Default: 0.8
    ls_tol : float
             Bisection line search tolerance.
             Default: 1.0e-8
    use_greedy : callable

                    ``use_greedy(kk) -> bool``

                 Method returning when to run the greedy step.
                 Default: lambda kk: False
    greedy_tol : callable

                    ``greedy_tol(kk) -> float``

                 Method returning the tolerance for the greedy step (in 2-norm
                 of the gradient).
                 Default: lambda kk: 1.0e-6
    max_greedy_iters : callable

                            ``max_greedy_iters(kk) -> int``

                       Method returning the maximum number of iterations for
                       greedy step.
                       Default: lambda kk: 500
    max_matmults : int
                   Approximate maximum number of matrix multiplications.
                   Default: np.inf (pseudo maximum integer)
    use_logging : bool
                  Whether or not to log the iterations in a text file.
                  Default: False
    flog_name : string
                File name for the log.
                Default: './logs/latest_cdpsd_log.csv'

    Returns
    -------
    The final objective value.

    Notes
    -----
    (1) The input X and y will be modified during the course of the algorithm
    to reflect the current iterate in a memory-efficient manner.

    (2) Basic checking for the dimension consistency of the data as well as the
    adjoint of opG and gradient of f are provided.

    References
    ----------
    See the paper "Conic Descent and its Applications to Memory-efficient
    Optimization over the Positive Semidefinite Cone" by ANONYMOUS (to appear).

    Examples
    --------
    See the README file.

    """

    # --------------------------------------
    # Check consistency of the problem input
    # --------------------------------------

    m, n = _check_prob_dimensions(f, gradf, opG, adjG, g, X, y, r)
    _check_prob_adjoint(opG, adjG, m, n, r)
    _check_prob_gradient(f, gradf, m)

    # -----------------------------------------------
    # Set problem parameters using defaults and input
    # -----------------------------------------------

    # Maximum number of iterations
    if 'max_iters' in kwargs:
        max_iters = kwargs['max_iters']
    else:
        max_iters = 5000

    # Tolerance as measured by distance (w.r.t. spectral norm) of gradient
    # to PSD cone
    if 'tol' in kwargs:
        tol = kwargs['tol']
    else:
        tol = 1.0e-6

    # Method to find approximate minimum eigenvalue (and associated normalized 
    # eigenvector)
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

    # Whether to display the header and footer
    if 'disp_head_foot' in kwargs:
        disp_head_foot = kwargs['disp_head_foot']
    else:
        disp_head_foot = True

    # What measurements to display to stdout
    if 'output' in kwargs:
        output = kwargs['output']
    else:
        output = ['iteration', 'obj', 'dfeas', 'slack']

    # Whether to use backtracking line search instead of exact line search
    if 'use_backtrack' in kwargs:
        use_backtrack = kwargs['use_backtrack']
    else:
        use_backtrack = False

    # Acceptable decrease parameter for backtracking line search parameters
    alpha = kwargs['alpha'] if 'alpha' in kwargs else 0.3

    # Refinement parameter for backtracking line search parameters
    beta = kwargs['beta'] if 'beta' in kwargs else 0.8

    # Bisection line search tolerance
    if 'ls_tol' in kwargs:
        ls_tol = kwargs['ls_tol']
    else:
        ls_tol = 1.0e-8

    # Method returning when to run the greedy step
    if 'use_greedy' in kwargs:
        use_greedy = kwargs['use_greedy']
    else:
        use_greedy = lambda kk: False

    # Method returning the tolerance for the greedy step (in 2-norm of the
    # gradient)
    if 'greedy_tol' in kwargs:
        greedy_tol = kwargs['greedy_tol']
    else:
        greedy_tol = lambda kk: 1.0e-6

    # Method returning the maximum number of iterations for greedy step
    if 'max_greedy_iters' in kwargs:
        max_greedy_iters = kwargs['max_greedy_iters']
    else:
        max_greedy_iters = lambda kk: 500

    # Maximum number of matrix multiplications
    if 'max_matmults' in kwargs:
        max_matmults = kwargs['max_matmults']
    else:
        max_matmults = np.inf # pseudo max int

    # Whether or not to log the iterations
    if 'use_logging' in kwargs:
        use_logging = kwargs['use_logging']
    else:
        use_logging = False

    # File name for the log
    if 'flog_name' in kwargs:
        flog_name = kwargs['flog_name']
    else:
        flog_name = './logs/latest_cdpsd_log.csv'

    # -----------------------
    # Initialize measurements
    # -----------------------
    
    meas = {'iteration':0,
            'obj':0.0,
            'slack':0.0,
            'dfeas':0.0,
            'eigmin_matmult':0,
            'eigmin_time':0.0,
            'eigmin_maxiter':0,
            'eigmin_tol':0.0,
            'eigmin_success':0,
            'use_greedy':0,
            'greedy_status':0,
            'greedy_message':'',
            'greedy_fun':0.0,
            'greedy_jac':0.0, # 2-norm of opt.OptimizeResult.jac
            'greedy_nfev':0,
            'greedy_njev':0,
            'greedy_nit':0,
            'greedy_matmult':0,
            'greedy_time':0.0,
            'greedy_maxiter':0,
            'greedy_tol':0.0,
            'greedy_success':0,
            'time':0.0}
    
    # ------------------------------
    # Start logging & display header
    # ------------------------------

    if use_logging:
        flog = open(flog_name, 'w')
        flog.write(','.join(meas) + '\n')

    if disp_head_foot:
        print('------------------------------------------------')
        print('Conic Descent over PSD Cone (Memory-Efficient)')
        print('------------------------------------------------')

    # --------------
    # Main iteration
    # --------------

    start_time = time.clock()
    total_matmults = 0
    x0_cached = np.random.randn(n, r) # For initializing the greedy step

    for kk in range(max_iters):

        meas['iteration'] = kk

        # -------------
        # CONIC DESCENT
        # -------------

        # Bisection line search for rescaling
        slo = 0.0
        shi = 1.1
        fls = lambda s: f(s * y + (s - 1.0) * g)
        while fls(shi) < fls(1.0):
            shi *= 2.0
        s = _ls_bisection(fls, lb=slo, ub=shi, tol=ls_tol)

        # Update X and y
        X.update(s, 0.0, np.zeros(n))
        y = s * y + (s - 1.0) * g

        meas['obj'] = f(y)
        meas['slack'] = np.vdot(gradf(y), (y + g))

        # Approximate minimum eigenvalue and associated normalized eigenvector
        t0 = time.clock()
        q_min, lambda_min, matmultcnt = eigmin(adjG(gradf(y)), max_iters=max_eigmin_iters(kk), tol=eigmin_tol(kk))
        t1 = time.clock()
        
        meas['dfeas'] = lambda_min
        meas['eigmin_matmult'] = matmultcnt
        meas['eigmin_time'] = t1 - t0
        meas['eigmin_maxiter'] = max_eigmin_iters(kk)
        meas['eigmin_tol'] = eigmin_tol(kk)

        total_matmults += matmultcnt

        # # Check stopping criteria
        # if lambda_min > -tol:
        #     break

        if lambda_min < 0.0:

            meas['eigmin_success'] = True

            dy = opG(q_min)
            if use_backtrack: # Backtracking line search
                fy = f(y)
                t = 1.0
                while f(y + t * dy) > fy + alpha * t * lambda_min:
                    t *= beta
            else: # Exact line search
                tlo = 0.0
                thi = 1.1
                fls = lambda t: f(y + t * dy)
                while fls(thi) < fls(1.0):
                    thi *= 2.0
                t = _ls_bisection(fls, lb=tlo, ub=thi, tol=ls_tol)

            # Update X and y
            X.update(1.0, t, q_min)
            y += t * dy

        else:
            meas['eigmin_success'] = False

        # -----------
        # GREEDY STEP
        # -----------

        meas['use_greedy'] = use_greedy(kk)
        
        if use_greedy(kk):

            # Use conjugate gradient to compute greedy step
            def f_greedy(x):
                return f((y + g) * x[0]**2 + opG(x[1:].reshape((n, -1))) - g)
            def gradf_greedy(x):
                temp = np.zeros_like(x)
                gf = gradf((y + g) * x[0]**2 + opG(x[1:].reshape((n, -1))) - g)
                temp[0] = 2.0 * x[0] * np.vdot(gf, (y + g))
                temp[1:] = 2.0 * (adjG(gf) @ x[1:].reshape((n, -1))).ravel()
                return temp
            x0_cached[:, np.argmin(la.norm(x0_cached, axis=0))] = q_min.reshape(-1)
            x0 = np.concatenate((np.array([1.0]), x0_cached.reshape(-1)))
            t0 = time.clock()
            res = opt.minimize(f_greedy, x0, method='CG', jac=gradf_greedy,
                               options={'maxiter':max_greedy_iters(kk),
                                        'gtol':greedy_tol(kk),
                                        'norm':2})
            t1 = time.clock()

            meas['greedy_status'] = res.status
            meas['greedy_message'] = res.message
            meas['greedy_fun'] = res.fun
            meas['greedy_jac'] = la.norm(res.jac)
            meas['greedy_nfev'] = res.nfev
            meas['greedy_njev'] = res.njev
            meas['greedy_nit'] = res.nit
            meas['greedy_matmult'] = res.njev * r # each gradf_greedy call uses r matrix-vector multiplications
            meas['greedy_time'] = t1 - t0
            meas['greedy_maxiter'] = max_greedy_iters(kk)
            meas['greedy_tol'] = greedy_tol(kk)

            total_matmults += res.njev * r

            # Update X and y
            if res.fun < f(y):
                meas['greedy_success'] = True
                X.update(res.x[0]**2, 1.0, res.x[1:].reshape((n, -1)))
                y = (y + g) * res.x[0]**2 - g + opG(res.x[1:].reshape((n, -1)))
            else:
                meas['greedy_success'] = False

            x0_cached = res.x[1:].reshape((n, -1))

        else:

            meas['greedy_status'] = 0
            meas['greedy_message'] = ''
            meas['greedy_fun'] = 0.0
            meas['greedy_jac'] = 0.0
            meas['greedy_nfev'] = 0
            meas['greedy_njev'] = 0
            meas['greedy_nit'] = 0
            meas['greedy_matmult'] = 0
            meas['greedy_time'] = 0.0
            meas['greedy_maxiter'] = 0
            meas['greedy_tol'] = 0.0
            meas['greedy_success'] = False

        # Update log
        meas['time'] = time.clock() - start_time
        if use_logging:
            flog.write(','.join(str(meas[key]) for key in meas) + '\n')

        # Break if total matrix multiplications exceeds maximum allowable
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

    # Display footer
    if disp_head_foot:
        print('------------------------------------------------')

    return f(y)
