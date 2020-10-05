"""
NAME: setup_psd_completion.py
AUTHOR: ANONYMOUS
AFFILIATION: ANONYMOUS
DATE MODIFIED: 10 June 2020
DESCRIPTION: Helper functions to set up the data for PSD matrix completion
             problems.
"""

# Standard imports
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import cvxpy as cp

def get_data(V, d, p=0.1):
    """Returns the functions and data necessary to run a PSD completion demo using CG or CD."""

    # Problem dimensions
    n = V.shape[0]

    # Choose measurement indices (WARNING: This is an O(n^2) operation! Must be modified for larger n.)
    II = []
    JJ = []
    for ii in range(n):
        for jj in range(ii, n):
            if ii <= d - 1 and jj <= d - 1:
                II.append(ii)
                JJ.append(jj)
            elif np.random.rand() <= p:
                II.append(ii)
                JJ.append(jj)
    m = len(II) # number of measurements
    II = np.array(II)
    JJ = np.array(JJ)

    def opA(U):
        """Evaluates the measurement operator on U U'."""

        if U.ndim == 1:
            return U[II] * U[JJ]
        else:
            return np.sum(U[II] * U[JJ], axis=1)

    def adjA(z):
        """Evaluates the adjoint of the measurement operator."""

        temp = sp.csr_matrix((z, (II, JJ)), shape=(n, n))

        return 0.5 * (temp + temp.T)
    
    # Measurements
    b = opA(V) + (np.random.randn(m) / np.sqrt(m)) * la.norm(opA(V)) / 10

    # Objective and its gradient
    f = lambda z: 0.5 * la.norm(z)**2 / m
    gradf = lambda z: z / m

    # Solve problem using cvxpy
    X = cp.Variable((n, n), PSD=True)
    y = cp.Variable(m)
    obj = cp.Minimize(0.5 * cp.sum_squares(y) / m)
    constr = []
    for kk in range(m):
        ii, jj = II[kk], JJ[kk]
        constr += [X[ii, jj] - b[kk] == y[kk]]
    prob = cp.Problem(obj, constr)
    prob.solve(eps=1e-12)

    return f, gradf, opA, adjA, b, n, prob.value

















