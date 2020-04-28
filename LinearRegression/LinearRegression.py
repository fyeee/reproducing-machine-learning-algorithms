import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


class LinearRegression:
    def __init__(self, learning_rate=0.03, num_iters=1500, normalize=False):
        self.learning_rate = learning_rate
        self.num_iters = num_iters
        self.normalize = normalize
        self.params = None
        self.coef_  = None
        self.intercept_ = None

    def fit(self, X, y):
        if self.normalize:
            mu = np.mean(X, 0)
            sigma = np.std(X, 0)
            X = (X - mu) / sigma
        X = np.hstack((np.ones((len(y), 1)), X))
        num_features = np.size(X, 1)
        self.params = np.zeros((num_features, 1))
        (J_history, self.params) = gradient_descent(X, y, self.params, self.learning_rate, self.num_iters)
        self.intercept_ = self.params[0]
        self.coef_ = self.params[1:]

    def predict(self, X):
        if self.normalize:
            mu = np.mean(X, 0)
            sigma = np.std(X, 0)
            X = (X - mu) / sigma
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        return X @ self.params

    def score(self, X, y):
        y_pred = self.predict(X)
        return 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))

def compute_cost(X, y, params):
    pred = X @ params
    n = len(y)
    return 1 / (2 * n) * np.sum((pred - y) ** 2)


def gradient_descent(X, y, params, learning_rate, num_iters):
    n = len(y)
    J_history = np.zeros((num_iters,1))
    for i in range(num_iters):
        t = X @ params
        params = params - (learning_rate / n) * X.T @ (t -y)
        J_history[i] = compute_cost(X, y, params)
    return J_history, params


if __name__ == "__main__":
    dataset = datasets.load_boston()
    X = dataset.data
    y = dataset.target[:, np.newaxis]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    clf = LinearRegression(normalize=True)
    clf.fit(X_train, y_train)
    # pred = clf.predict(X_test)
    print(clf.score(X_train, y_train))
    print(clf.score(X_test, y_test))
