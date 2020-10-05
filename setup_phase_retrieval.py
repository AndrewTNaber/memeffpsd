"""
NAME: setup_phase_retrieval.py
AUTHOR: ANONYMOUS
AFFILIATION: ANONYMOUS
DATE MODIFIED: 10 June 2020
DESCRIPTION: Helper functions to set up the data for phase retrieval problems.
"""

# Standard imports
import numpy as np
import scipy.linalg as la
from scipy.sparse import linalg as spla
from scipy.fftpack import dctn, idctn

# For unpacking CIFAR-10 dataset
import pickle

def unpickle(fname):
    """Unpickles the CIFAR-10 dataset in fname."""

    with open(fname, 'rb') as f:
        batch_dict = pickle.load(f, encoding='bytes')

    return batch_dict

def get_cifar10_image(ind):
    """Gets grayscale image at index ind from CIFAR-10 dataset1."""

    cifar_batch1_dict = unpickle('./data/data_batch_1')

    # Pick out the image and convert it to grayscale
    img = cifar_batch1_dict[b'data'][ind]
    img_red = img[:1024].reshape(32, 32)
    img_green = img[1024:2048].reshape(32, 32)
    img_blue = img[2048:].reshape(32, 32)
    img_gray = (0.2989 * img_red + 0.5870 * img_green + 0.1140 * img_blue) / 255

    return img_gray

def get_data(img, k, gamma=0.0, SNR=20.0):
    """Returns the functions and data necessary to run a phase retrieval demo using CG or CD."""

    if img.ndim != 2 or img.shape[0] != img.shape[1]:
        raise ValueError('image must be square')

    # Unravel image
    x_original = img.ravel()

    # Problem dimensions
    n_img = img.shape[0] # image dimension
    n = n_img**2 # signal size
    m = k * n # number of measurements

    # Phase masks
    S = 2 * (np.random.randint(2, size=(n, k)) - 0.5)

    def opA(U):
        """Evaluates the measurement operator on U U'."""

        temp = np.zeros(m)
        if U.ndim == 1:
            for kk in range(k):
                temp[kk * n:(kk + 1) * n] += dctn((S[:, kk] * U).reshape(n_img, n_img), norm='ortho').reshape(n)**2
        else:
            for jj in range(U.shape[1]):
                for kk in range(k):
                    temp[kk * n:(kk + 1) * n] += dctn((S[:, kk] * U[:, jj]).reshape(n_img, n_img), norm='ortho').reshape(n)**2
        return temp

    def adjA(z):
        """Evaluates the adjoint of the measurement operator."""

        def mult_efficient(v):
            """Efficient matrix-vector multiplication for the adjoint of the measurement operator."""

            temp = np.zeros(n)
            for kk in range(k):
                temp += S[:, kk] * idctn((z[kk * n:(kk + 1) * n] * dctn((S[:, kk] * v.reshape(-1)).reshape(n_img, n_img), norm='ortho').reshape(n)).reshape(n_img, n_img), norm='ortho').reshape(n)
            return temp

        return spla.LinearOperator((n, n), matvec=mult_efficient, dtype=np.float_)
    
    # Measurements
    epsilon = np.random.randn(m) * la.norm(opA(x_original)) / (np.sqrt(m) * 10**(SNR / 20))  # Gaussian noise
    b = opA(x_original) + epsilon

    # Overall linear operator and its adjoint
    opG = lambda U: np.concatenate((opA(U), np.full(1, la.norm(U)**2)))
    adjG = lambda z: adjA(z[:m]) + z[m] * spla.LinearOperator((n, n), matvec=lambda v: v, dtype=np.float_)

    # Overall offset
    g = np.concatenate((b, np.zeros(1)))

    # Objective and its gradient
    f = lambda z: 0.5 * la.norm(z[:m])**2 / m + gamma * z[m]
    gradf = lambda z: np.concatenate((z[:m] / m, np.full(1, gamma)))

    return f, gradf, opG, adjG, g, n

