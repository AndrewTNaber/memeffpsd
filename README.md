# Conic Descent and its Application to Memory-efficient Optimization over the Positive Semidefinite Cone

This repository is the official implementation of our paper.

## Requirements

The code and examples were developed using Python 3.7.1 with the packages:
1. numpy (1.15.4)
2. scipy (1.1.0)
3. matplotlib (3.0.2)
4. pandas (0.23.4)
5. cvxpy (1.0.24)

The first four are standard parts of most scientific Python distributions.  We *strongly* recommend [Anaconda](https://anaconda.org).

For cvxpy, please refer to the [official documentation](https://www.cvxpy.org/install/index.html) for installation guidance.  However, cvxpy is only required for the examples to compute accurate reference optimal values.  It is *not* needed for the core implementation of our algorithm contained in cdpsd.py, matrix_variable.py, and min_eigvec.py.

## Basic Usage

The following command
```
$ python test_cdpsd.py
```
will set up a toy problem, solve it using both our algorithm and the Splitting Conic Solver (SCS) included with cvxpy, and display a plot demonstrating convergence.  The code in test_cdpsd.py is a basic usage example.

The core of our algorithm is contained in three files:
1. cdpsd.py contains the solve function which solves problem (5) in our paper.
2. matrix_variable.py contains classes for storing the matrix variable.  There are currently three choices: MatrixVariableEmpty (which does not store the matrix), MatrixVariableFull (which stores the full matrix for debugging), and MatrixVariableSketch (which stores a memory-efficient Nystrom sketch).
3. min_eigvec.py contains functions for computing minimum eigenvalues and eigenvectors.

In its most basic usage, you will have already defined the following functions in accordance with the version of problem (5) you want to solve with dimensions n and m as defined in the paper:
1. `f`, the objective loss acting on an m vector
2. `gradf`, the gradient of the objective loss acting on an m vector
3. `opG`, the linear operator G acting on the square root of a PSD n x n matrix
4. `adjG`, the adjoint of the linear operator G acting on an m vector

Note that `f` and `gradf` and `opG` and `adjG` will be checked numerically when cdpsd.solve is called because it's pretty easy to mess them up!  For explicit examples of these functions in phase retrieval or matrix completion problems, see the example sections below and their source.  It is very important that `opG` and `adjG` be set up to run as efficiently as possible.  The final part of the problem data is the m vector `g`.  Here are the definitions used in test_cdpsd.py (but now in the Python command prompt):
```
>>> import numpy as np
>>> import scipy.linalg as la
>>> m = 100
>>> n = 10
>>> def f(x):
        return 0.5 * la.norm(x)**2
>>> def gradf(x):
        return x
>>> G = np.random.randn(m, n, n)
>>> G = 0.5 * (G + np.transpose(G, (0, 2, 1))) # Symmetrize
>>> G /= la.norm(G, axis=(1, 2)).reshape(m, 1, 1) # Normalize
>>> def opG(U):
        if U.ndim == 1:
            return U @ G @ U
        else:
            return np.trace(U.T @ G @ U, axis1=1, axis2=2)
>>> def adjG(z):
        return np.sum(G * z.reshape(m, 1, 1), axis=0)
>>> g = opG(np.random.randn(n, 3) / (n * 3)**0.25) + 0.1 * np.random.randn(m)
```

After you have defined the problem data (`f`, `gradf`, `opG`, `adjG`, and `g`) you must instantiate a MatrixVariable instance `X` from matrix_variable.py, and set up the corresponding auxiliary variable `y`.  Both `X` and `y` will be modified during the course of the CD iterations to reflect the current iterate.  Here's a basic Python command line example of this which sets up a Nystrom sketch of the n x n zero matrix with rank parameter 5 and initializes the auxiliary variable accordingly (assuming prior definitions of the problem data):
```
>>> import cdpsd, matrix_variable
>>> X = matrix_variable.MatrixVariableSketch(n, 5)
>>> y = -g
>>> cdpsd.solve(f, gradf, opG, adjG, g, X, y)
```

You should see something that looks like this:
```
------------------------------------------------
Conic Descent over PSD Cone (Memory-Efficient)
------------------------------------------------
iteration          obj        dfeas        slack
        0    1.767e+01   -8.115e+00    1.654e-16
      500    3.559e-01   -1.313e-02    8.216e-08
     1000    3.433e-01   -6.687e-03   -2.277e-07
     1500    3.383e-01   -3.901e-03   -2.548e-07
     2000    3.355e-01   -3.307e-03    9.009e-08
     2500    3.337e-01   -2.928e-03   -8.393e-08
     3000    3.325e-01   -2.593e-03    1.756e-08
     3500    3.316e-01   -1.703e-03   -1.977e-07
     4000    3.308e-01   -1.804e-03    2.131e-07
     4500    3.302e-01   -1.705e-03   -3.500e-08
------------------------------------------------
```

You can recover a (thin) approximate eigenvalue decomposition of the final iterate from the Nystrom sketch using
```
>>> evecs, evals, _ = X.get_approx_evd()
```

Detailed help about the usage and optional keyword arguments of cdpsd.solve can be obtained by running
```
>>> help(cdpsd.solve)
```
at the Python command line.

## Replicating figures from the paper

### Phase Retrieval Problems
The file setup_phase_retrieval.py provides helper functions to set up phase retrieval problems generated from the CIFAR-10 dataset.  To generate a histogram like that in Figure 1, run the command
```
$ python gen_fig_phase_ret_hist.py 50
```
Beware: 50 samples takes a little while, so you might want to try something like 5 at first.  To generate four images of the horse in Figure 1, run the command
```
$ python gen_fig_phase_ret_horses.py
```

### Matrix Completion Problems
The file setup_matrix_completion.py provides helper function to set up PSD matrix completion problems.  To generate a greedy convergence plot like that in Figure 2, run the command
```
$ python gen_fig_mat_comp_plot.py
```
To generate box plots like those of Figure 2, run the command
```
$ python gen_fig_mat_comp_box.py 50
```
Again, 50 samples takes a little while, so you might want to try something like 5 at first.
