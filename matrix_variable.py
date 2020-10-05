# NAME: matrix_variable.py
# AUTHOR: ANONYMOUS
# AFFILIATION: ANONYMOUS
# DATE MODIFIED: 10 June 2020
# DESCRIPTION: Classes for storing the matrix variable as part of cdpsd.solve.
#              The Nystrom sketch implementation was motivated from the code
#              for the paper "Scalable Semidefinite Programming" by Yurtsever
#              et al. Tests for all classes can be run from running as script.
# TO DO:
# 1. Finish documentation.
# 2. How does la.cholesky work if the matrix isn't symmetric?
# 3. Consider an extension to rectangular matrices:
#       AbstractMatrixVariable
#           L MatrixVariable
#               L MatrixVariableEmpty
#               L MatrixVariableFull
#               L MatrixVariableSketch
#           L SymmetricMatrixVariable
#               L SymmetricMatrixVariableEmpty
#               L SymmetricMatrixVariableFull
#               L SymmetricMatrixVariableSketch
# 4. Consider extension to complex (Hermitian) matrices

# Standard imports
from abc import ABC, abstractmethod
import numpy as np
import scipy.linalg as la

class MatrixVariable(ABC):
    """Abstract matrix variable"""

    @abstractmethod
    def size(self):
        """Returns size of matrix"""
        pass

    @abstractmethod
    def tr(self):
        """Returns trace of matrix"""
        pass

    @abstractmethod
    def update(self, s, t, U):
        """X <- s * X + t * U * U'"""
        pass

class MatrixVariableEmpty(MatrixVariable):
    """Empty matrix variable"""

    def __init__(self, n, tr=0.0):
        self._n = n
        self._tr = tr

    def size(self):
        return self._n

    def tr(self):
        return self._tr

    def update(self, s, t, U):
        self._tr = s * self._tr + t * la.norm(U)**2

class MatrixVariableFull(MatrixVariable):
    """Full matrix variable"""

    def __init__(self, n, X=None):
        self._n = n
        self._X = X if X is not None else np.zeros((n, n))
        self._tr = np.trace(X) if X is not None else 0.0

    def size(self):
        return self._n

    def tr(self):
        return self._tr

    def update(self, s, t, U):
        if U.ndim == 1:
            self._X = s * self._X + t * np.outer(U, U)
        else:
            self._X = s * self._X + t * U @ U.T
        self._tr = np.trace(self._X)

    def get_matrix(self):
        """Returns full matrix"""
        return self._X

class MatrixVariableSketch(MatrixVariable):
    """Nystrom sketch"""

    def __init__(self, n, r):
        self._n = n
        self._r = r
        self._Omega = np.random.randn(n, r)
        self._Y = np.zeros((n, r))
        self._tr = 0.0

    def size(self):
        return self._n

    def tr(self):
        return self._tr

    def rk(self):
        """Returns rank used by the sketch"""
        return self._r

    def update(self, s, t, U):
        if U.ndim == 1:
            self._Y = s * self._Y + t * U.reshape((-1, 1)) @ (U.reshape((-1, 1)).T @ self._Omega)
        else:
            self._Y = s * self._Y + t * U @ (U.T @ self._Omega)
        self._tr = s * self._tr + t * la.norm(U)**2

    def get_approx_evd(self):
        """Returns approximate eigenvalue decomposition of matrix"""
        shift = np.sqrt(self._n) * np.spacing(la.norm(self._Y, 2))
        Y_shifted = self._Y + shift * self._Omega
        L = la.cholesky(self._Omega.T @ Y_shifted, lower=True)
        _, sigma, VT = la.svd(la.lstsq(L, Y_shifted.T)[0], full_matrices=False)
        err = self._tr - np.sum(np.maximum(0.0, sigma**2 - shift))
        return VT.T, np.maximum(0.0, sigma**2 - shift), err

def _test_MatrixVariableEmpty():
    print("Checking class MatrixVariableEmpty... ", end='')
    flag = True
    n = 5
    U = np.arange(n).reshape((n, 1))
    t = 1.1
    X = MatrixVariableEmpty(n)
    if X.size() != n:
        flag &= False
    try:
        X.update(1.0, t, U)
    except:
        flag &= False
    print("PASS") if flag else print("FAIL")

def _test_MatrixVariableFull():
    print("Checking class MatrixVariableFull... ", end='')
    flag = True
    n = 5
    U = np.arange(n).reshape((n, 1))
    t = 1.1
    X_true = t * U @ U.T
    X = MatrixVariableFull(n)
    X.update(1.0, t, U)
    if X.size() != n:
        flag &= False
    if not np.allclose(X.get_matrix(), X_true):
        flag &= False
    U = np.arange(2 * n).reshape((n, 2))
    X_true = t * U @ U.T
    X = MatrixVariableFull(n)
    X.update(1.0, t, U[:, 0:1])
    X.update(1.0, t, U[:, 1:2])
    if not np.allclose(X.get_matrix(), X_true):
        flag &= False
    if not np.allclose(X.get_matrix(), X.get_matrix().T):
        flag &= False
    X = MatrixVariableFull(n, np.zeros((n, n)))
    X.update(1.0, t, U[:, 0])
    X.update(1.0, t, U[:, 1])
    if not np.allclose(X.get_matrix(), X_true):
        flag &= False
    if not np.allclose(X.get_matrix(), X.get_matrix().T):
        flag &= False
    X_input = np.random.randn(n, n)
    X = MatrixVariableFull(n, X_input)
    if not np.allclose(X.get_matrix(), X_input):
        flag &= False
    X_true = X_input + t * U @ U.T
    X.update(1.0, t, U)
    if not np.allclose(X.get_matrix(), X_true):
        flag &= False
    print("PASS") if flag else print("FAIL")

def _test_MatrixVariableSketch():
    print("Checking class MatrixVariableSketch... ", end='')
    flag = True
    n = 100
    r = 5
    U = np.random.randn(n, 1)
    t = 1.1
    X_true = t * U @ U.T
    X = MatrixVariableSketch(n, 1)
    X.update(1.0, t, U)
    if X.size() != n:
        flag &= False
    Q, d, err = X.get_approx_evd()
    if not np.allclose(Q @ np.diag(d) @ Q.T, X_true):
        flag &= False
    U = np.random.randn(n, 2)
    X_true = t * U @ U.T
    X = MatrixVariableSketch(n, r)
    X.update(1.0, t, U[:, 0:1])
    X.update(1.0, t, U[:, 1:2])
    Q, d, err = X.get_approx_evd()
    if not np.allclose(Q @ np.diag(d) @ Q.T, X_true):
        flag &= False
    X = MatrixVariableSketch(n, r)
    X.update(1.0, t, U[:, 0])
    X.update(1.0, t, U[:, 1])
    Q, d, err = X.get_approx_evd()
    if not np.allclose(Q @ np.diag(d) @ Q.T, X_true):
        flag &= False
    print("PASS") if flag else print("FAIL")

def _run_all_tests():
    _test_MatrixVariableEmpty()
    _test_MatrixVariableFull()
    _test_MatrixVariableSketch()

if __name__ == '__main__':
    _run_all_tests()
