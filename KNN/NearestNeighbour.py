import math
import numpy as np


def euclidean_distance(x, y):
    return math.sqrt(sum([(x[i] - y[i]) ** 2 for i in range(len(x))]))


def nearest_neighbour_no_library(train_X, train_y, test_X):
    pred_y = []
    for i in range(len(test_X)):
        dist = [euclidean_distance(test_X[i], train_X[j]) for j in range(len(train_X))]
        pred_y.append(train_y[dist.index(min(dist))])
    return pred_y


def nearest_neighbour_numpy(train_X, train_y, test_X):
    train_X = np.array(train_X)
    test_X = np.array(test_X)
    pred_y = []
    for x in test_X:
        dist = np.sqrt(np.sum((train_X - x) ** 2, axis=1))
        pred_y.append(train_y[np.argmin(dist)])
    return pred_y


if __name__ == "__main__":
    X = [[2.7810836, 2.550537003],
               [1.465489372, 2.362125076],
               [3.396561688, 4.400293529],
               [1.38807019, 1.850220317],
               [3.06407232, 3.005305973],
               [7.627531214, 2.759262235],
               [5.332441248, 2.088626775],
               [6.922596716, 1.77106367],
               [8.675418651, -0.242068655],
               [7.673756466, 3.508563011]]
    y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    print(nearest_neighbour_no_library(X, y, [[2, 2], [3, 3]]))
    print(nearest_neighbour_numpy(X, y, [[2, 2], [3, 3]]))
