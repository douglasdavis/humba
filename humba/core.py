import numba
import numpy as np
import math
from typing import Optional, Tuple


@numba.jit(nopython=True)
def jit_histogram_w(x, nbins, xmin, xmax, weights):
    norm = 1.0 / (xmax - xmin)
    count = np.zeros((nbins + 2))
    sumw2 = np.zeros((nbins + 2))
    for i in range(x.shape[0]):
        if x[i] < xmin:
            count[0] += weights[i]
            sumw2[0] += weights[i] * weights[i]
        elif x[i] > xmax:
            count[nbins + 1] += weights[i]
            sumw2[nbins + 1] += weights[i]
        else:
            binid = int((x[i] - xmin) * nbins * norm)
            count[binid + 1] += weights[i]
            sumw2[binid + 1] += weights[i]
    return count, np.sqrt(sumw2)


@numba.jit(nopython=True)
def jit_histogram(x, nbins, xmin, xmax):
    norm = 1.0 / (xmax - xmin)
    count = np.zeros((nbins + 2))
    for i in range(x.shape[0]):
        if x[i] < xmin:
            count[0] += 1
        elif x[i] > xmax:
            count[nbins + 1] += 1
        else:
            binid = int((x[i] - xmin) * nbins * norm)
            count[binid + 1] += 1
    return count


def histogram(
    x: np.ndarray,
    bins: int = 10,
    range: Tuple[float,float] = (0, 10),
    weights: Optional[np.ndarray] = None,
    uoflow: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Calculate the histogram for the data ``x``

    Parameters
    ----------
    x
        data to histogram
    nbins
        int
    range
        axis range
    weights
        array of weights for ``x``
    uoflow
        include over and underflow content in first and last bins
    """
    if weights is not None:
        res, err = jit_histogram_w(x, bins, range[0], range[1], weights)
        if uoflow:
            res[1] += res[0]
            res[-2] += res[-1]
            err[1] = math.sqrt(err[1] ** 2 + err[0] ** 2)
            err[-2] = math.sqrt(err[-2] ** 2 + err[-1] ** 2)
        return res[1:-1], err[1:-1]
    else:
        res = jit_histogram(x, bins, range[0], range[1])
        if uoflow:
            res[1] += res[0]
            res[-2] += res[-1]
        return res[1:-1]
