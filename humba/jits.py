"""
module housing just-in-time compiled functions
"""

import numba as nb
import numpy as np


@nb.jit(
    nb.types.UniTuple(nb.float32[:], 2)(
        nb.float32[:], nb.float32[:], nb.int64, nb.float64, nb.float64, nb.boolean
    ),
    nopython=True,
)
def _hfloat32_weighted(
    x: np.ndarray, weights: np.ndarray, nbins: int, xmin: float, xmax: float, flow: bool
):
    """ detail jit compiled function for 32 bit floating point weighted histograms """
    norm = 1.0 / (xmax - xmin)
    count = np.zeros((nbins + 2), dtype=np.float32)
    sumw2 = np.zeros((nbins + 2), dtype=np.float32)
    for i in range(x.shape[0]):
        binid = 0
        if x[i] < xmin:
            pass
        elif x[i] > xmax:
            binid = nbins + 1
        else:
            binid = int((x[i] - xmin) * nbins * norm) + 1
        weight = weights[i]
        count[binid] += weight
        sumw2[binid] += weight ** 2
    if flow:
        count[-2] += count[-1]
        sumw2[-2] += sumw2[-1]
        count[1] += count[0]
        sumw2[1] += sumw2[0]
    count, sumw2 = count[1:-1], sumw2[1:-1]
    return count, np.sqrt(sumw2)


@nb.jit(
    nb.float32[:](nb.float32[:], nb.int64, nb.float64, nb.float64, nb.boolean),
    nopython=True,
)
def _hfloat32(x: np.ndarray, nbins: int, xmin: float, xmax: float, flow: bool):
    """ detail jit compiled function for 32 bit floating point histograms """
    norm = 1.0 / (xmax - xmin)
    count = np.zeros((nbins + 2), dtype=np.float32)
    for i in range(x.shape[0]):
        binid = 0
        if x[i] < xmin:
            pass
        elif x[i] > xmax:
            binid = nbins + 1
        else:
            binid = int((x[i] - xmin) * nbins * norm) + 1
        count[binid] += 1
    if flow:
        count[-2] += count[-1]
        count[1] += count[0]
    return count[1:-1]


@nb.jit(
    nb.types.UniTuple(nb.float64[:], 2)(
        nb.float64[:], nb.float64[:], nb.int64, nb.float64, nb.float64, nb.boolean
    ),
    nopython=True,
)
def _hfloat64_weighted(x, weights, nbins, xmin, xmax, flow):
    """ detail funtion for 64 bit floating point weighted histograms """
    norm = 1.0 / (xmax - xmin)
    count = np.zeros((nbins + 2), dtype=np.float64)
    sumw2 = np.zeros((nbins + 2), dtype=np.float64)
    for i in range(x.shape[0]):
        binid = 0
        if x[i] < xmin:
            pass
        elif x[i] > xmax:
            binid = nbins + 1
        else:
            binid = int((x[i] - xmin) * nbins * norm) + 1
        weight = weights[i]
        count[binid] += weight
        sumw2[binid] += weight ** 2
    if flow:
        count[-2] += count[-1]
        sumw2[-2] += sumw2[-1]
        count[1] += count[0]
        sumw2[1] += sumw2[0]
    count, sumw2 = count[1:-1], sumw2[1:-1]
    return count, np.sqrt(sumw2)


@nb.jit(
    nb.float64[:](nb.float64[:], nb.int64, nb.float64, nb.float64, nb.boolean),
    nopython=True,
)
def _hfloat64(x: np.ndarray, nbins: int, xmin: float, xmax: float, flow: bool):
    """ detail jit compiled function for 64 bit floating point histograms """
    norm = 1.0 / (xmax - xmin)
    count = np.zeros((nbins + 2), dtype=np.float64)
    for i in range(x.shape[0]):
        binid = 0
        if x[i] < xmin:
            pass
        elif x[i] > xmax:
            binid = nbins + 1
        else:
            binid = int((x[i] - xmin) * nbins * norm) + 1
        count[binid] += 1
    if flow:
        count[-2] += count[-1]
        count[1] += count[0]
    return count[1:-1]


@nb.jit(
    nb.types.UniTuple(nb.float32[:, :], 2)(
        nb.float32[:], nb.float32[:, :], nb.int64, nb.float64, nb.float64, nb.boolean
    ),
    nopython=True,
    parallel=True,
)
def _hfloat32_multiweights(
    x: np.ndarray, weights: np.ndarray, nbins: int, xmin: float, xmax: float, flow: bool
):
    """ detail jit compiled function for 32 bit multiweighted histograms """
    norm = 1.0 / (xmax - xmin)
    count = np.zeros((nbins + 2, weights.shape[1]), dtype=np.float32)
    sumw2 = np.zeros((nbins + 2, weights.shape[1]), dtype=np.float32)
    for i in range(x.shape[0]):
        binid = 0
        if x[i] < xmin:
            pass
        elif x[i] > xmax:
            binid = nbins + 1
        else:
            binid = int((x[i] - xmin) * nbins * norm) + 1
        for j in range(weights.shape[1]):
            weight = weights[i][j]
            count[binid][j] += weight
            sumw2[binid][j] += weight * weight
    if flow:
        count[-2, :] += count[-1, :]
        sumw2[-2, :] += sumw2[-1, :]
        count[1, :] += count[0, :]
        sumw2[1, :] += sumw2[0, :]

    count = count[1:-1, :]
    sumw2 = sumw2[1:-1, :]
    return count, np.sqrt(sumw2)


@nb.jit(
    nb.types.UniTuple(nb.float64[:, :], 2)(
        nb.float64[:], nb.float64[:, :], nb.int64, nb.float64, nb.float64, nb.boolean
    ),
    nopython=True,
    parallel=True,
)
def _hfloat64_multiweights(
    x: np.ndarray, weights: np.ndarray, nbins: int, xmin: float, xmax: float, flow: bool
):
    """ detail jit compiled function for 64 bit multiweighted histograms """
    norm = 1.0 / (xmax - xmin)
    count = np.zeros((nbins + 2, weights.shape[1]), dtype=np.float64)
    sumw2 = np.zeros((nbins + 2, weights.shape[1]), dtype=np.float64)
    for i in range(x.shape[0]):
        binid = 0
        if x[i] < xmin:
            pass
        elif x[i] > xmax:
            binid = nbins + 1
        else:
            binid = int((x[i] - xmin) * nbins * norm) + 1
        for j in range(weights.shape[1]):
            weight = weights[i][j]
            count[binid][j] += weight
            sumw2[binid][j] += weight * weight
    if flow:
        count[-2, :] += count[-1, :]
        sumw2[-2, :] += sumw2[-1, :]
        count[1, :] += count[0, :]
        sumw2[1, :] += sumw2[0, :]

    count = count[1:-1, :]
    sumw2 = sumw2[1:-1, :]
    return count, np.sqrt(sumw2)
