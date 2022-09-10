"""
Viet and Luke
CISC-820: Quantitative Foundations
Project 1: Linear Feature Engineering
"""
from __future__ import annotations
from itertools import chain, combinations_with_replacement
from typing import Optional

import numpy as np


class PolynomialTransform:
    def __init__(self, degree: int, interactions: bool):
        """Create the polynomial transform.

        Args:
            degree (int): the degree of the transform, e.g., the degree 3 expansion of
                [a, b, c] is [a^3, a^2b, a^2c, ab^2, abc, b^3, ac^2, bc^2, c^3, a, b, c]
            interactions (bool): whether to include interactions between features
        """
        if degree < 2:
            raise ValueError("Expansion degree must be an integer greater than 1.")
        self.degree = degree
        self.interactions = interactions

    def __repr__(self):
        return (
            "PolynomialTransform("
            + f"degree={self.degree}, "
            + f"interactions={self.interactions}"
            + ")"
        )

    def fit(
        self, X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None
    ) -> PolynomialTransform:
        return self

    def transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Perform polynomial expansion of the features.

        Args:
            X: training data

        Returns:
            polynomial expansion of the data
        """
        # Expresses the columns from the original X to multiply together to form new X
        # e.g., [(0,), (0, 0), (0, 0, 0)] signifies X[:, 0]^1, X[:, 0]^2, X[:, 0]^3
        if self.interactions:
            combos = list(
                chain.from_iterable(
                    (
                        combinations_with_replacement(range(X.shape[1]), i)
                        for i in range(1, self.degree + 1)
                    )
                )
            )
        else:
            combos = [[i for _ in range(self.degree)] for i in range(X.shape[1])]
        # Create the new X by performing element-wise multiplication
        X = np.concatenate(
            [np.prod(X[:, c], axis=1, keepdims=True) for c in combos], axis=1
        )
        return X

    def fit_transform(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        self.fit(X)
        return self.transform(X)


class NoTransform:
    """Adheres to the scikit-learn transformation API, but does not transform."""

    def __repr__(self):
        return "NoTransform()"

    def __str__(self):
        return repr(self)

    def fit(
        self, X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None
    ) -> NoTransform:
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X

    def fit_transform(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        return X
