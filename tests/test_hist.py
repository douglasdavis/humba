import humba
import numpy as np


def test64():
    x = np.random.randn(10000)
    hh, _, hedges = humba.histogram(x, bins=20, range=(-3, 3))
    hn, nedges = np.histogram(x, bins=20, range=(-3, 3))
    assert np.allclose(hh, hn)
    assert np.allclose(hedges, nedges)


def test64_weighted():
    x = np.random.randn(10000)
    w = np.random.uniform(0.5, 1.5, x.shape[0])
    hh, _, hedges = humba.histogram(x, bins=20, range=(-3, 3), weights=w)
    hn, nedges = np.histogram(x, bins=20, range=(-3, 3), weights=w)
    assert np.allclose(hh, hn)
    assert np.allclose(hedges, nedges)


def test32():
    x = np.random.randn(10000).astype(np.float32)
    hh, _, hedges = humba.histogram(x, bins=20, range=(-3, 3))
    hn, nedges = np.histogram(x, bins=20, range=(-3, 3))
    assert np.allclose(hh, hn)
    assert np.allclose(hedges, nedges)


def test32_weighted():
    x = np.random.randn(10000).astype(np.float32)
    w = np.random.uniform(0.5, 1.5, x.shape[0]).astype(np.float32)
    hh, _, hedges = humba.histogram(x, bins=20, range=(-3, 3), weights=w)
    hn, nedges = np.histogram(x, bins=20, range=(-3, 3), weights=w)
    assert np.allclose(hh, hn)
    assert np.allclose(hedges, nedges)
