"""
Viet and Luke
CISC-820: Quantitative Foundations
Project 1: Linear Feature Engineering
"""
# %%

from __future__ import annotations
import json
from typing import List, Optional

import numpy as np
from sklearn.base import RegressorMixin
from sklearn.cluster import FeatureAgglomeration
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler


np.random.seed(0)


class LinearRegression:
    """Linear regression model."""

    w: Optional[np.ndarry]

    def __init__(self, bias: bool, which_feature: List[int]=None, n_expand: int=None):
        """Create the regression model.

        Args:
            bias: to include a bias term. If True, adds a column of 1s to features
        """
        self.bias = bias
        self.w = None
        self.which_feature = which_feature
        self.n_expand = n_expand

    def __repr__(self):
        return f"LinearRegression(bias={self.bias})"

    def fit(self, X: np.ndarray, y: np.ndarray) -> LinearRegression:
        """Fit the regression model.

        Args:
            X: training data
            y: corresponding regression values

        Returns:
            the fitted model
        """
        # Perform poly expansion
        if self.which_feature and self.n_expand:
            # There the number of features can be reduced due to pca, so no need to perform them
            try:
                X = poly_expand(X, self.which_feature, self.n_expand)
            except:
                pass

        # Bias term addition
        X = np.concatenate((X, np.ones((len(X), 1))), axis=1) if self.bias else X
        
        s = np.dot(y, X)
        M = np.dot(X.T, X)
        self.w = np.linalg.solve(M, s)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Perform predictions based upon the fitted model.

        Args:
            X: training data

        Returns:
            predictions for the corresponding regression values
        """
        if self.w is None:
            raise ValueError("Model has not been fitted with predict yet.")
        X = np.concatenate((X, np.ones((len(X), 1))), axis=1) if self.bias else X
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
            raise ValueError("Model has not been fitted with predict yet.")
        X = np.concatenate((X, np.ones((len(X), 1))), axis=1) if self.bias else X
        return float(np.sum((np.dot(X, self.w) - y) ** 2))

def poly_expand(X: np.ndarray, which_feature: List[int], n_expand: int) -> np.ndarray:
    """ Perform polynomial expansion

    Args:
        X (np.ndarray): 2D features. each column is one instance. Expand row rules:
        x -> 1 -> x^2 -> x^3 -> x^4 -> etc.

    Returns:
        np.ndarray: _description_
    """
    rows, cols = X.shape
    X_hats = []
    for feature in which_feature:
        
        assert feature < cols, "Wrong feature to expand."

        X_hat = np.array(X[:, which_feature])[..., np.newaxis] # (rows, 1)
        X_hat = np.broadcast_to(X_hat, (rows, n_expand)) # (rows, n_expand)
        
        power_mat = np.array([(2 + i) for i in range(n_expand)])  # power_mat without first or 2 first rows: [2,3,4,5,6...]
        power_mat = power_mat[np.newaxis, ...] # (1, n_expand): [[2,3,4,5,6...]]
        power_mat = np.broadcast_to(power_mat, (rows, n_expand)) # (rows, n_expand): [[2,3,4,5,6...]; [2,3,4,5,6...]; [2,3,4,5,6...]; ...]

        X_hat = np.power(X_hat, power_mat)
        X_hats.append(X_hat)

    return np.concatenate((X, *X_hat), axis=1)

class NoTransform:
    """Adheres to the scikit-learn transformation API, but does not transform anything."""

    def __repr__(self):
        return "NoTransform()"

    def fit(self, X: np.ndarray, y: np.ndarray) -> NoTransform:
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return X


def cross_validation(
    reg: RegressorMixin, X: np.ndarray, y: np.ndarray, k: int
) -> np.ndarray:
    """Perform cross validation and return the error for each fold.

    Args:
        reg: a regression model that implements fit() and score()
        X: training data
        y: corresponding regression values
        k: number of folds

    Returns:
        scores for each fold, as returned by reg.score()
    """
    n = len(y)
    errors = np.empty(k)
    for k_i in range(k):
        start = k_i * n // k
        end = (k_i + 1) * n // k
        X_train, X_test = np.concatenate((X[0:start], X[end:n])), X[start:end]
        y_train, y_test = np.concatenate((y[0:start], y[end:n])), y[start:end]
        reg = reg.fit(X_train, y_train)
        error = reg.score(X_test, y_test)
        errors[k_i] = error
    return errors


def main():
    """Test different transformations and models and record predictions for test data.
    """
    test_data = np.loadtxt("testinputs.txt")
    train_data = np.loadtxt("traindata.txt")
    X, y = train_data[:, 0:8], train_data[:, 8]
    k = 5
    # Various transformations and regression algorithms
    normalizers = [
        lambda: Normalizer("l1"),
        lambda: Normalizer("l2"),
        lambda: Normalizer("max"),
        lambda: NoTransform()
    ]
    standardizers = [
        lambda: StandardScaler(),
        lambda: NoTransform()
    ]
    reducers = [
        lambda: FeatureAgglomeration(),
        lambda: PCA(),
        lambda: NoTransform()
    ]
    regressors = [
        lambda: LinearRegression(True),
        lambda: LinearRegression(False),
        lambda: LinearRegression(True, [2], 15),
        lambda: LinearRegression(False, [4, 6], 5),
        lambda: LinearRegression(True, [1, 0, 4], 10),
        lambda: LinearRegression(False, [0,1,2,3,4,5,6,7], 3),
    ]
    
    # Store regression results from all combinations of preprocessing and regression
    results = []
    for normalizer in normalizers:
        for scalar in standardizers:
            for reducer in reducers:
                for regressor in regressors:
                    pipeline = Pipeline(
                        [
                            ("scalar", scalar()),
                            ("normalizer", normalizer()),
                            ("reducer", reducer()),
                            ("regressor", regressor()),
                        ]
                    )
                    error = cross_validation(pipeline, X, y, k).mean()
                    rep = str(pipeline).replace("\n", " ").replace("  ", "")
                    predictions = list(pipeline.predict(test_data))
                    results.append({
                        "MSR": error,
                        "Pipeline": rep,
                        "Predictions": predictions
                    })
    # Sort results by MSR score and output to files
    results.sort(key=lambda x: x["MSR"])
    with open(f"results.json", "w") as f:
        json.dump(results, f, indent=4)
    np.savetxt("testoutputs.txt", results[0]["Predictions"])


    # print(f"Shape x: {X.shape}; shape y: {y.shape}; test input shape: {test_data.shape}")
    # reg = LinearRegression_v2(True)
    
    # Xx = reg.poly_expand(X[:, 0:1], 0, 1)
    # print(np.mean(X[:, 0]))
    # print(np.mean(Xx[:, 1]))




if __name__ == "__main__":
    main()
