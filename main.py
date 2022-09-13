"""
Viet and Luke
CISC-820: Quantitative Foundations
Project 1: Linear Feature Engineering
"""

from __future__ import annotations
from argparse import ArgumentParser
import json

import numpy as np
from sklearn.base import RegressorMixin
from sklearn.cluster import FeatureAgglomeration
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import SVR

from linear_regression import LR
from transforms import NoTransform, PolynomialTransform
import neural_net_eval
import neural_net_train


np.random.seed(0)


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the mean squared error between two arrays.

    Args:
        y_true (np.ndarray): true values
        y_pred (np.ndarray): predicted values

    Returns:
        float: mean squared error
    """
    return ((y_true - y_pred) ** 2).mean()


def cross_validation(
    reg: RegressorMixin, X: np.ndarray, y: np.ndarray, k: int
) -> np.ndarray:
    """Perform cross validation and return the msr for each fold.

    Args:
        reg: a regression model that implements fit() and predict()
        X (np.ndarray): training data
        y (np.ndarray): corresponding regression values
        k (np.ndarray): number of folds

    Returns:
        (np.ndarray) mean squared error for each fold
    """
    n = len(y)
    msrs = np.empty(k)
    for k_i in range(k):
        start = k_i * n // k
        end = (k_i + 1) * n // k
        X_train, X_test = np.concatenate((X[0:start], X[end:n])), X[start:end]
        y_train, y_test = np.concatenate((y[0:start], y[end:n])), y[start:end]
        reg = reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        msrs[k_i] = mean_squared_error(y_test, y_pred)
    return msrs


def main(k: int, p: int, results_file: str, save_all: bool, verbose: bool) -> None:
    """Test different transformations and models and record predictions for test data.

    Args:
        k (int): number of folds for cross validation
        p (int): degree of polynomial transform
        results_file (str): .json file to save results to
        save_all (bool): whether to save predictions for all models or only best model
        verbose (bool): whether to print progress to stdout or not
    """
    # Acquire the train and test data
    test_data = np.loadtxt("testinputs.txt")
    train_data = np.loadtxt("traindata.txt")
    X, y = train_data[:, 0:8], train_data[:, 8]
    # Establish the transformations to perform on the data
    expanders = (
        [lambda: PolynomialTransform(i, True) for i in range(2, p + 1)]
        + [lambda: PolynomialTransform(i, False) for i in range(2, p + 1)]
        + [lambda: NoTransform()]
    )
    preprocessors = [
        lambda: Normalizer(),
        lambda: StandardScaler(),
        lambda: NoTransform(),
    ]
    reducers = [lambda: FeatureAgglomeration(), lambda: PCA(), lambda: NoTransform()]
    regressors = [
        lambda: LR(),
        lambda: SVR(),
    ]
    # Perform regression with all different kinds of transformations and models
    results = []
    i = 0
    total = len(expanders) * len(preprocessors) * len(reducers) * len(regressors)
    for expander in expanders:
        for preprocessor in preprocessors:
            for reducer in reducers:
                for regressor in regressors:
                    i += 1
                    if verbose:
                        print(f"{i} / {total} = {i / total * 100}%")
                    # Pipeline of transformations, ending with regressor model
                    pipeline = Pipeline(
                        [
                            ("expander", expander()),
                            ("preprocessor", preprocessor()),
                            ("reducer", reducer()),
                            ("regressor", regressor()),
                        ]
                    )
                    rep = [f"{s[0]}: {s[1]}" for s in pipeline.steps]
                    # Average mean squared error of each fold from cross val
                    msr = cross_validation(pipeline, X, y, k).mean()
                    # Train pipeline on complete set of data training data
                    pipeline.fit(X, y)
                    predictions = list(pipeline.predict(test_data))
                    # Update results
                    results.append(
                        {
                            "Rank": i,
                            "MSR": msr,
                            "Pipeline": rep,
                            "Predictions": predictions,
                        }
                    )
    # Sort results by MSR score
    results.sort(key=lambda x: x["MSR"])
    for i, r in enumerate(results):
        r["Rank"] = i
    # Save results to JSON file
    np.savetxt("testoutputs.txt", results[0]["Predictions"])
    if not save_all:
        for r in results:
            del r["Predictions"]
    # Save the predictions of the best model to txt file
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4, default=lambda x: repr(x))


def submission():
    """Produces the submission file only."""
    neural_net_train.main()
    neural_net_eval.main()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--submission", action="store_true", help="produce submission")
    parser.add_argument("-k", type=int, default=5, help="number of folds for cv")
    parser.add_argument("-p", type=int, default=3, help="max polynomial degree")
    parser.add_argument("-r", type=str, default="results.json", help="results file")
    parser.add_argument("-s", action="store_true", help="save all predictions")
    parser.add_argument("-v", action="store_true", help="verbose")
    args = parser.parse_args()
    if args.submission:
        submission()
    else:
        main(args.k, args.p, args.r, args.s, args.v)
