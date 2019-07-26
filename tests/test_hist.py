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


def test64_manyweights():
    x = np.random.randn(10000)
    w = (0.1 + np.random.rand(10000, 20)) / 2.0
    hh, _, edges = humba.mwv_histogram(x, w, bins=20, range=(-3, 3))
    for i in range(hh.shape[1]):
        ihn, inedges = np.histogram(x, bins=20, range=(-3, 3), weights=w.T[i])
        assert np.allclose(hh.T[i], ihn)


def test32_manyweights():
    x = np.random.randn(10000).astype(np.float32)
    w = (0.1 + np.random.rand(10000, 20)) / 2.0
    hh, _, edges = humba.mwv_histogram(x, w, bins=20, range=(-3, 3))
    for i in range(hh.shape[1]):
        ihn, inedges = np.histogram(x, bins=20, range=(-3, 3), weights=w.T[i])
        assert np.allclose(hh.T[i], ihn)
