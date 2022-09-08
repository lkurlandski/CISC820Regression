def least_squares(Z:np.ndarray, y:np.ndarray) -> (np.ndarray, float):
    """Solve the least squares problem Zw = y for w.

    Args:
        Z (np.ndarray): matrix of shape (n, p)
        y (np.ndarray): vector of shape (n,)

    Returns:
        (np.ndarray): vector of shape (p,) solving Zw = y
        (float): residual sum of squares, i.e., the error
    """
    s = np.dot(y, Z)
    M = np.dot(Z.T, Z)
    w = np.linalg.solve(M, s)
    r = np.sum((np.dot(Z, w) - y) ** 2)
    return w, r


def cross_validation(X:np.ndarray, y:np.ndarray, k:int) -> np.ndarray:
    """Perform k-fold cross validation on the data.

    Args:
        X (np.ndarray): matrix of shape (n, p)
        y (np.ndarray): vector of shape (n,)
        k (int): number of folds

    Returns:
        (np.ndarray): vector of shape (k,) containing the residual sum of squares
            for each fold
    """
    n = len(y)
    errors = np.empty(k)
    for k_i in range(k):
        start = k_i * n // k
        end = (k_i + 1) * n // k
        X_train, X_test = np.concatenate((X[0:start], X[end:n])), X[start:end]
        y_train, y_test = np.concatenate((y[0:start], y[end:n])), y[start:end]
        w, _ = least_squares(X_train, y_train)
        r = np.sum((np.dot(X_test, w) - y_test) ** 2)
        errors[k_i] = r
    return errors
