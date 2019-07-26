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
