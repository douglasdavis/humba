import numpy as np
from typing import Optional, Tuple

import humba.jits as hj


def histogram(
    x: np.ndarray,
    bins: int = 10,
    range: Tuple[float, float] = (0, 10),
    weights: Optional[np.ndarray] = None,
    flow: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Calculate the histogram for the data ``x``.

    Parameters
    ----------
    x : np.ndarray
        data to histogram
    bins : np.ndarray
        number of bins
    range : (float, float)
        axis range
    weights : np.ndarray, optional
        array of weights for ``x``
    flow : bool
        include over and underflow content in first and last bins

    Returns
    -------
    count : np.ndarray
        The values of the histogram
    error : np.ndarray, optional
        The poission uncertainty on the bin heights
    edges : np.ndarray
        The bin edges

    Notes
    -----
    If the dtype of the ``weights`` is not the same as ``x``, then it
    is converted to the dtype of ``x``.

    Examples
    --------
    >>> import numpy as np
    >>> from humba import histogram
    >>> x = np.random.randn(100000)
    >>> w = np.random.uniform(0.4, 0.5, x.shape[0])
    >>> hist1, _, edges = humba.histogram(x, bins=50, range=(-5, 5))
    >>> hist2, _, edges = humba.histogram(x, bins=50, range=(-5, 5), flow=True)
    >>> hist3, error, edges = histogram(x, bins=50, range=(-5, 5), weights=w)
    >>> hist4, error, edges = histogram(x, bins=50, range=(-3, 3), weights=w, flow=True)

    """
    edges = np.linspace(range[0], range[1], bins + 1)
    if weights is not None:
        assert x.shape == weights.shape, "x and weights must have identical shape"
        if x.dtype == np.float64:
            hfunc = hj._float64_weighted
        elif x.dtype == np.float32:
            hfunc = hj._float32_weighted
        else:
            raise TypeError("dtype of input must be float32 or float64")
        res, err = hfunc(x, weights.astype(x.dtype), bins, range[0], range[1], flow)
        return (res, err, edges)
    else:
        if x.dtype == np.float64:
            hfunc = hj._float64
        elif x.dtype == np.float32:
            hfunc = hj._float32
        else:
            raise TypeError("dtype of input must be float32 or float64")
        res = hfunc(x, bins, range[0], range[1], flow)
        return (res, None, edges)


def histogram_mw(
    x: np.ndarray,
    weights: np.ndarray,
    bins: int = 10,
    range: Tuple[float, float] = (0, 10),
    flow: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Histogram the same data but with multiple weight variations.

    Paramters
    ---------
    x : np.ndarray
        data to histogram
    weights : np.ndarray, optional
        multidimensional array of weights for ``x`` the first element
        of the ``shape`` attribute must be equal to the length of ``x``.
    bins : np.ndarray
        number of bins
    range : (float, float)
        axis range
    flow : bool
        include over and underflow content in first and last bins

    Returns
    -------
    count : np.ndarray
        The values of the histograms calculated from the weights
        Shape will be (bins, ``weights.shape[0]``)
    error : np.ndarray
        The poission uncertainty on the bin heights (shape will be
        the same as ``count``.
    edges : np.ndarray
        The bin edges

    Notes
    -----
    If the dtype of the ``weights`` is not the same as ``x``, then it
    is converted to the dtype of ``x``.

    """
    edges = np.linspace(range[0], range[1], bins + 1)
    assert x.shape[0] == weights.shape[0], "weights shape is not compatible with x"
    if x.dtype == np.float64:
        hfunc = hj._float64_multiweights
    elif x.dtype == np.float32:
        hfunc = hj._float32_multiweights
    else:
        raise TypeError("dtype of input must be float32 or float64")
    res, err = hfunc(x, weights.astype(x.dtype), bins, range[0], range[1], flow)
    return (res, err, edges)
