import math
import numpy as np
# for testing the algo
from sklearn import datasets
from sklearn.model_selection import train_test_split


def accuracy(prediction, actual):
    count = 0
    for i in range(len(prediction)):
        if prediction[i] == actual[i]:
            count += 1
    return count/len(prediction)


def euclidean_distance(x, y):
    return math.sqrt(sum([(x[i] - y[i]) ** 2 for i in range(len(x))]))


def K_nearest_neighbours_no_library(train_X, train_y, test_X, k=1):
    pred_y = []
    for i in range(len(test_X)):
        dist = [(euclidean_distance(test_X[i], train_X[j]), j) for j in range(len(train_X))]
        dist.sort(key=lambda tup: tup[0])
        neighbours = []
        for j in range(k):
            neighbours.append(train_y[dist[j][1]])
        predition = max(set(neighbours), key=neighbours.count)
        pred_y.append(predition)
    return pred_y


def K_nearest_neighbours_numpy(train_X, train_y, test_X, k=1):
    train_X = np.array(train_X)
    train_y = np.array(train_y)
    test_X = np.array(test_X)
    pred_y = []
    for x in test_X:
        dist = np.sqrt(np.sum((train_X - x) ** 2, axis=1))
        neighbours = train_y[np.argpartition(dist, k)][:k]
        counts = np.bincount(neighbours)
        pred_y.append(np.argmax(counts))
    return pred_y


if __name__ == "__main__":
    data = datasets.load_iris()
    X = data["data"].tolist()
    y = data["target"].tolist()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    prediction = K_nearest_neighbours_no_library(X_train, y_train, X_test, k=10)
    print("Accuracy for KNN without using any library: {0}%".format(accuracy(prediction, y_test) * 100))
    prediction = K_nearest_neighbours_numpy(X_train, y_train, X_test, k=10)
    print("Accuracy for KNN using only numpy: {0}%".format(accuracy(prediction, y_test) * 100))
