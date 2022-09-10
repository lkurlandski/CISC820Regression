"""
Viet and Luke
CISC-820: Quantitative Foundations
Project 1: Linear Feature Engineering
"""

from __future__ import annotations
from typing import Optional

import numpy as np


class NotFittedError(ValueError):
    ...


class LR:
    """Linear regression model."""

    w: Optional[np.ndarry]

    def __init__(self, bias: bool = True):
        """Create the regression model.

        Args:
            bias (bool): whether to include a bias term
        """
        self.w = None
        self.bias = bias

    def __repr__(self):
        return f"LR()"

    def add_bias(self, X: np.ndarray) -> np.ndarray:
        """Add a bias term to the data if necessary.

        Args:
            X (np.ndarray): data to add bias to

        Returns:
            (np.ndarray) data with bias term if necessary or the original data
        """
        X = np.concatenate((X, np.ones((len(X), 1))), axis=1) if self.bias else X
        return X

    def fit(self, X: np.ndarray, y: np.ndarray) -> LR:
        """Fit the regression model.

        Args:
            X: training data
            y: corresponding regression values

        Returns:
            the fitted model
        """
        X_ = self.add_bias(X)
        s = np.dot(y, X_)
        M = np.dot(X_.T, X_)
        self.w = np.linalg.solve(M, s)
        msr = self.score(X, y)  # for debugging
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Perform predictions based upon the fitted model.

        Args:
            X: training data

        Returns:
            predictions for the corresponding regression values
        """
        if self.w is None:
            raise NotFittedError()
        X = self.add_bias(X)
        y = np.dot(X, self.w)
        return y

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Score the fitted model according to mean square error.

        Args:
            X: training data
            y: corresponding regression values

        Returns:
            MSR score
        """
        if self.w is None:
            raise NotFittedError()
        X = self.add_bias(X)
        return float(np.sum((np.dot(X, self.w) - y) ** 2) / len(y))
