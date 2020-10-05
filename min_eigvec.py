# NAME: min_eigvec.py
# AUTHOR: ANONYMOUS
# AFFILIATION: ANONYMOUS
# DATE MODIFIED: 10 June 2020
# DESCRIPTION: Functions for computing the minimum eigenvalue/vector using
#              ARPACK, Lanczos, shift invert, or power iteration.
#              Implementations of the power iteration and Lanczos methods
#              are motivated by those in the code for "Scalable Semidefinite
#              Programming" by Yurtsever et al.  Tests for all functions can
#              be run from running as script.
# TO DO:
# 1. Set the default tolerances more intelligently

# Standard imports
import numpy as np
import scipy.linalg as la
from scipy.sparse import linalg as spla

def _twonormest(A, max_iters=100, tol=1.0e-6):
    """Estimates the 2-norm of linear operator A using power method."""

    cnt = 0

    n = A.shape[1]

    x = np.random.randn(n)
    x /= la.norm(x)

    g_new = 1.0
    g_old = 0.0

    # Main iteration
    for _ in range(max_iters):
        g_old = g_new
        Ax = A @ x
        x = A @ Ax
        g_new = la.norm(x) / la.norm(Ax)
        x /= la.norm(x)
        cnt += 2
        if (g_new - g_old) <= g_new * tol:
            return g_new, cnt
    
    return g_new, cnt

def min_eigvec_power(A, max_iters=30, tol=1.0e-6, shift=None):
    """Finds approximate minimum eigenvalue/vector using power method."""

    n = A.shape[1]

    # Estimate spectral norm of A
    shift, matmultcnt = _twonormest(A, tol=0.1)
    shift *= 2.0

    # Main iteration
    q = np.random.randn(n) # Random initialization
    q /= la.norm(q)
    Aq = A @ q
    matmultcnt += 1
    d = np.vdot(q, Aq)
    for _ in range(max_iters):
        q = shift * q - Aq
        q /= la.norm(q)
        Aq = A @ q
        matmultcnt += 1
        d = np.vdot(q, Aq)
        if la.norm(Aq - d * q) <= abs(d) * tol:
            break

    return q.reshape((-1, 1)), d, matmultcnt

def min_eigvec_Lanczos(A, max_iters=30, tol=1.0e-6, shift=None):
    """Finds approximate minimum eigenvalue/vector using Lanczos method."""

    n = A.shape[1]

    max_iters = min(max_iters, n - 1)

    # Random initialization
    Q = np.zeros((n, max_iters + 1))
    Q[:, 0] = np.random.randn(n)
    Q[:, 0] /= la.norm(Q[:, 0])

    # Diagonal and off-diagonal elements
    alpha = np.zeros(max_iters)
    beta = np.zeros(max_iters)

    # Lanczos iteration
    matmultcnt = 0
    for ii in range(max_iters):
        Q[:, ii + 1] = A @ Q[:, ii]
        matmultcnt += 1
        alpha[ii] = np.vdot(Q[:, ii], Q[:, ii + 1])
        if ii == 0:
            Q[:, 1] -= alpha[0] * Q[:, 0]
        else:
            Q[:, ii + 1] -= alpha[ii] * Q[:, ii] + beta[ii - 1] * Q[:, ii - 1]
        beta[ii] = la.norm(Q[:, ii + 1])
        if abs(beta[ii]) < np.sqrt(n) * np.spacing(1.0):
            break
        Q[:, ii + 1] /= beta[ii]

    # Compute approximate eigenvalues
    if ii == 0:
        return Q[:, :1], alpha[0], matmultcnt
    else:
        d, q = la.eigh_tridiagonal(alpha[:ii + 1], beta[:ii], select='i', select_range=(0, 0))
        return Q[:, :ii + 1] @ q, d[0], matmultcnt

def min_eigvec_shift_invert(A, max_iters=100, tol=1.0e-6, shift=None):
    """Finds approximate eigenvalue/vector using shift and invert method."""

    n = A.shape[1]

    matmultcnt = 0

    if shift is None:
        rho, matmultcnt = _twonormest(A)
        shift = -rho

    # Set up shifted linear operator (with added count of matrix multiplications)
    def mult_shift_with_cnt(v):
        nonlocal matmultcnt
        matmultcnt += 1
        return A @ v - shift * v
    slo = spla.LinearOperator(A.shape, matvec=mult_shift_with_cnt, dtype=np.float_)

    # Random initialization
    u = np.random.randn(n, 1)

    # Main iteration
    for ii in range(max_iters):
        v = u / la.norm(u)
        Av = A @ v
        matmultcnt += 1
        lambda_ = np.vdot(v, Av)
        if la.norm(Av - lambda_ * v) <= abs(lambda_) * tol:
            return v, lambda_, matmultcnt
        u = spla.cg(slo, v, tol=tol / 2)[0]
        d = np.vdot(u, v)
        if la.norm(u - d * v) <= abs(d) * np.spacing(1.0):
            break

    v = u / la.norm(u)
    lambda_ = np.vdot(v, A @ v)
    matmultcnt += 1

    return v, lambda_, matmultcnt

def min_eigvec_ARPACK(A, max_iters=100, tol=1.0e-6, shift=None):
    """Convenience wrapper for spla.eigsh which calls ARPACK."""

    # Hack to get the number of matrix-vector multiplications used by ARPACK
    count = 0
    def mult_with_count(v):
        nonlocal count
        count += 1
        return A @ v
    A_with_count = spla.LinearOperator(A.shape, matvec=mult_with_count, dtype=np.float_)

    # Call ARPACK
    d, q = spla.eigsh(A_with_count, k=1, which='SA', maxiter=max_iters, tol=tol)

    return q, d[0], count

def _test__twonormest():
    print('Checking _twonormest... ', end='')
    flag = True
    A = np.random.randn(10, 10)
    A = 0.5 * (A + A.T)
    est, _ = _twonormest(A)
    if abs(est - la.svdvals(A)[0]) / abs(la.svdvals(A)[0]) > 1.0e-3:
        flag &= False
    print('PASS') if flag else print('FAIL')

def _test_min_eigvec_power():
    print('Checking min_eigvec_power... ', end='')
    flag = True
    A = np.random.randn(10, 10)
    A = 0.5 * (A + A.T)
    q, d, _ = min_eigvec_power(A, max_iters=1000, tol=1.0e-3)
    if abs(d - la.eigvalsh(A)[0]) / abs(la.eigvalsh(A)[0]) > 1.0e-3:
        flag &= False
    print('PASS') if flag else print('FAIL')

def _test_min_eigvec_Lanczos():
    print('Checking min_eigvec_Lanczos... ', end='')
    flag = True
    A = np.random.randn(10, 10)
    A = 0.5 * (A + A.T)
    q, d, _ = min_eigvec_Lanczos(A, max_iters=1000, tol=1.0e-3)
    if abs(d - la.eigvalsh(A)[0]) / abs(la.eigvalsh(A)[0]) > 1.0e-3:
        flag &= False
    print('PASS') if flag else print('FAIL')

def _test_min_eigvec_shift_invert():
    print('Checking min_eigvec_shift_invert... ', end='')
    flag = True
    A = np.random.randn(10, 10)
    A = 0.5 * (A + A.T)
    q, d, _ = min_eigvec_shift_invert(A, max_iters=1000, tol=1.0e-3)
    if abs(d - la.eigvalsh(A)[0]) / abs(la.eigvalsh(A)[0]) > 1.0e-3:
        flag &= False
    print('PASS') if flag else print('FAIL')

def _test_min_eigvec_ARPACK():
    print('Checking min_eigvec_ARPACK... ', end='')
    flag = True
    A = np.random.randn(10, 10)
    A = 0.5 * (A + A.T)
    q, d, count = min_eigvec_ARPACK(A, max_iters=1000, tol=1.0e-3)
    if abs(d - la.eigvalsh(A)[0]) / abs(la.eigvalsh(A)[0]) > 1.0e-3:
        flag &= False
    print('PASS') if flag else print('FAIL')

def _run_all_tests():
    _test__twonormest()
    _test_min_eigvec_power()
    _test_min_eigvec_Lanczos()
    _test_min_eigvec_shift_invert()
    _test_min_eigvec_ARPACK()

if __name__ == '__main__':
    _run_all_tests()
