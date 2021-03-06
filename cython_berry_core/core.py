# AUTOGENERATED! DO NOT EDIT! File to edit: 00_core.ipynb (unless otherwise specified).

__all__ = ['npmaxabs', 'simulated_shares', 'simulated_mean_shares', 'berry_core']

# Cell

import numpy as np
import numpy.linalg as npla
from typing import Tuple

def npmaxabs(arr: np.ndarray) -> float:
    return np.max(np.abs(arr))


def simulated_shares(utils: np.array) -> np.array:
    """
    return simulated shares for given simulated utilities

    :param np.array utils: array `(nproducts, ndraws)`

    :return: simulated shares `(nproducts, ndraws)`
    """
    shares = np.exp(utils)
    denom = 1.0 + np.sum(shares, 0)
    shares = shares / denom
    return shares


def simulated_mean_shares(utils: np.array) -> np.array:
    """
    return simulated mean shares for given simulated utilities

    :param np.array utils: array `(nproducts, ndraws)`

    :return: np.array simulated mean shares: array `(nproducts)`
    """
    return np.mean(simulated_shares(utils), 1)

def berry_core(shares: np.array, mean_u: np.array, X2: np.array,
               Sigma: np.array, tol: float = 1e-9,
               maxiter: int = 10000, ndraws: int = 10000, verbose: bool=False) -> Tuple[np.array, bool]:
    """
    contraction to invert for product effects :math:`\\xi` from market shares

    :param np.array shares: `nproducts` vector of observed market shares

    :param np.array mean_u: `(nproducts)` vector of mean utilities

    :param np.array X2: `(nproducts, nx2)` matrix of nonlinear covariates

    :param np.array Sigma: `(nx2, nx2)` variance-covariance matrix of random coefficients on `X2`, \
    or `(nx2)` if diagonal

    :param float tol: tolerance

    :param int maxiter: max iterations

    :param int ndraws: number of draws for simulation

    :params bool verbose: print stuff if `True`

    :return: `(nproducts)` vector of :math:`\\xi` values, and return code 0 if OK
    """
    nproducts, nx2 = X2.shape
    assert shares.size == nproducts, "should have as many shares as rows in X2"
    assert mean_u.size == nproducts, "should have as many mean utilities as rows in X2"

    if Sigma.ndim == 1 and Sigma.size == nx2:
        assert np.min(Sigma) >= 0.0, "berry_core: all elements of the diagonal Sigma should be positive or 0"
        Xsig = X2 * np.sqrt(Sigma)
    elif Sigma.ndim == 2 and Sigma.shape == (nx2, nx2):
        L = npla.cholesky(Sigma)
        Xsig = X2 @ L
    else:
        print_stars("berry_core: Sigma should be (nx2, nx2) or (nx2)")
        sys.exit()

    sum_shares = shares.sum()
    market_zero_share = 1.0 - sum_shares

    xi = np.log(shares / market_zero_share) - mean_u
    max_err = np.Inf
    retcode = 0
    iter = 0
    eps = np.random.normal(size=(nx2, ndraws))
    while max_err > tol:
        utils = (Xsig @ eps) + (mean_u + xi).reshape((-1, 1))
        shares_sim = simulated_mean_shares(utils)
        err_shares = shares - shares_sim
        max_err = npmaxabs(err_shares)
        if verbose and iter % 100 == 0:
            print(f"berry_core: error {max_err} after {iter} iterations")
        iter += 1
        if iter > maxiter:
            retcode = 1
            break
        xi += (np.log(shares) - np.log(shares_sim))
    if verbose:
        print_stars(f"berry_core: error {max_err} after {iter} iterations")
    return xi, retcode