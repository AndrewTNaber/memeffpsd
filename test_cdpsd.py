# NAME: test_cdpsd.py
# AUTHOR: ANONYMOUS
# AFFILIATION: ANONYMOUS
# DATE MODIFIED: 10 June 2020
# DESCRIPTION: Tests the basic functionality of cdpsd.solve on a random problem
#              instance and compares it with an accurate answer from SCS.

# Standard imports
import numpy as np
import scipy.linalg as la
import cvxpy as cp
import matplotlib.pyplot as plt
import pandas as pd

# Solves CD over PSD cone in a memory-efficient manner
import cdpsd

# Classes for storing the matrix variable
import matrix_variable

if __name__ == '__main__':

    np.random.seed(0) # For consistency of results

    # -------------------
    # Basic problem setup
    # -------------------

    m = 100
    n = 10

    def f(x):
        return 0.5 * la.norm(x)**2
    def gradf(x):
        return x

    G = np.random.randn(m, n, n)
    G = 0.5 * (G + np.transpose(G, (0, 2, 1))) # Symmetrize
    G /= la.norm(G, axis=(1, 2)).reshape(m, 1, 1) # Normalize
    def opG(U):
        if U.ndim == 1:
            return U @ G @ U
        else:
            return np.trace(U.T @ G @ U, axis1=1, axis2=2)
    def adjG(z):
        return np.sum(G * z.reshape(m, 1, 1), axis=0)

    g = opG(np.random.randn(n, 3) / (n * 3)**0.25) + 0.1 * np.random.randn(m)

    # -------------------------
    # Solve problem using CDPSD
    # -------------------------

    X = matrix_variable.MatrixVariableEmpty(n)
    y = -g
    opts = dict(use_logging=True,
                flog_name='./logs/latest_test_cdpsd_log.csv',
                use_backtrack=True)
    print('CDPSD optimal value:', cdpsd.solve(f, gradf, opG, adjG, g, X, y, **opts))

    # -------------------------
    # Solve problem using CVXPY
    # -------------------------

    X_cvx = cp.Variable((n, n), PSD=True)
    y_cvx = cp.Variable(m)
    objective = cp.Minimize(0.5 * cp.sum_squares(y_cvx))
    constraints = [y_cvx[ii] == cp.trace(G[ii] @ X_cvx) - g[ii] for ii in range(m)]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS, eps=1.0e-6)
    pstar = problem.value
    print('CVXPY optimal value:', pstar)

    # -------------------
    # Display convergence
    # -------------------

    df = pd.read_csv('./logs/latest_test_cdpsd_log.csv')
    plt.loglog(np.cumsum(df['eigmin_matmult']), df['obj'] - pstar)
    plt.xlabel('matrix multiplications')
    plt.ylabel(r'$f(X_k) - p^\star$')
    plt.grid()
    plt.tight_layout()
    plt.show()















