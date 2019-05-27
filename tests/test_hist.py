import humba
import numpy as np

def test1():
    x = np.random.randn(10000)
    hh, _ = humba.histogram(x, bins=20, range=(-3, 3))
    hn, _ = np.histogram(x, bins=20, range=(-3, 3))
    assert np.allclose(hh, hn)

def test2():
    x = np.random.randn(10000)
    w = np.random.uniform(0.5, 1.5, x.shape[0])
    hh, _ = humba.histogram(x, bins=20, range=(-3, 3), weights=w)
    hn, _ = np.histogram(x, bins=20, range=(-3, 3), weights=w)
    assert np.allclose(hh, hn)
