"""
Viet and Luke
CISC-820: Quantitative Foundations
Project 1: Linear Feature Engineering

Test numbers refer to exercises given in class. The order of the feature columns differ
from the order used in class, so the weight vectors do not correspond exactly to the
weight vectors supplied in class.
"""

import numpy as np
import pytest as pt

from linear_regression import LR
from transforms import PolynomialTransform


def test_12():
    x = np.array([0.23, 0.88, 0.21, 0.92, 0.49, 0.62, 0.77, 0.52, 0.30, 0.19])
    X = np.stack((x, np.ones_like(x)), axis=-1)
    y = [0.19, 0.96, 0.33, 0.80, 0.46, 0.45, 0.67, 0.32, 0.38, 0.37]
    reg = LR(bias=False).fit(X, y)
    msr = reg.score(X, y)
    assert np.allclose(reg.w, [0.7653, 0.1004], atol=1e-3)
    assert 10 * msr == pt.approx(0.1128, abs=1e-3)


def test_14():
    X = np.array([[0.1, 0.5], [0.2, 0.6], [0.3, 0.7], [0.4, 0.7]])
    y = [0.9, 1, 1.1, 1.2]
    reg = LR(bias=False).fit(X, y)
    msr = reg.score(X, y)
    assert np.allclose(reg.w, [-0.15942, 1.73913])
    assert 4 * msr == pt.approx(0.00927, abs=1e-3)


def test_16():
    x = np.array([0.23, 0.88, 0.21, 0.92, 0.49, 0.62, 0.77, 0.52, 0.30, 0.19])
    X = np.stack((x**2, x), axis=-1)
    y = [0.19, 0.96, 0.33, 0.80, 0.46, 0.45, 0.67, 0.32, 0.38, 0.37]
    reg = LR(bias=True).fit(X, y)
    msr = reg.score(X, y)
    assert np.allclose(reg.w, [1.369, -0.721, 0.406], atol=1e-3)
    assert 10 * msr == pt.approx(0.0632, abs=1e-3)


def test_17():
    x = np.array([0.23, 0.88, 0.21, 0.92, 0.49, 0.62, 0.77, 0.52, 0.30, 0.19])
    X = np.stack((np.sin(2 * x), np.log(x), np.sqrt(x)), axis=-1)
    y = [0.19, 0.96, 0.33, 0.80, 0.46, 0.45, 0.67, 0.32, 0.38, 0.37]
    reg = LR(bias=False).fit(X, y)
    msr = reg.score(X, y)
    assert np.allclose(reg.w, [-1.438, 0.134, 2.409], atol=1e-3)
    assert 10 * msr == pt.approx(0.0672, abs=1e-3)


def test_polynomial_transform_1():
    # Test one
    X = np.array([
        [0, 1],
        [2, 3],
        [4, 5]
    ])
    X_ = np.array([
        [ 0.,  1.,  0.,  0.,  1.],
        [ 2.,  3.,  4.,  6.,  9.],
        [ 4.,  5., 16., 20., 25.]
    ])
    X__ = PolynomialTransform(degree=2, interactions=True).fit_transform(X)
    assert np.allclose(X_, X__)


def test_polynomial_transform_2():
    X = np.array([
        [0, 1],
        [2, 3],
        [4, 5]
    ])
    X_ = np.array([
        [0.,   1.,   0.,   0.,   1.,   0.,   0.,   0.,   1.],
        [2.,   3.,   4.,   6.,   9.,   8.,  12.,  18.,  27.],
        [4.,   5.,  16.,  20.,  25.,  64.,  80., 100., 125.]
    ])
    X__ = PolynomialTransform(degree=3, interactions=True).fit_transform(X)
    assert np.allclose(X_, X__)
