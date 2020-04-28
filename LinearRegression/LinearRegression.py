import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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

    n_samples = len(y)

    mu = np.mean(X, 0)
    sigma = np.std(X, 0)

    X = (X - mu) / sigma

    X = np.hstack((np.ones((n_samples, 1)), X))

    n_features = np.size(X, 1)
    param = np.zeros((n_features, 1))

    n_iters = 1500
    learning_rate = 0.01

    initial_cost = compute_cost(X, y, param)

    print("Initial cost is: ", initial_cost, "\n")

    (J_history, optimal_params) = gradient_descent(X, y, param, learning_rate, n_iters)

    print("Optimal parameters are: \n", optimal_params, "\n")

    print("Final cost is: ", J_history[-1])

    plt.plot(range(len(J_history)), J_history, 'r')

    plt.title("Convergence Graph of Cost Function")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Cost")
    plt.show()