import math


def entropy(count, n):
    return -1 * math.log(count / n, 2) * (count / n)


def entropy_multi(array):
    """
    H(Y)
    """
    s = set(array)
    n = len(array)
    total_entropy = 0
    for item in s:
        count = 0
        for element in array:
            if element == item:
                count += 1
        total_entropy += entropy(count, n)
    return total_entropy


def information_gain(X, y, element_index):
    """
    IG(Y|element)
    """
    s = set([row[element_index] for row in X])
    entropy_y = entropy_multi(y)
    entropy_condition = float("inf")
    for item in s:
        array_left = []
        array_right = []
        for i in range(len(X)):
            if X[i][element_index] <= item:
                array_left.append(y[i])
            else:
                array_right.append(y[i])
        entropy_condition = min(entropy_condition, len(array_left) / len(X) * entropy_multi(array_left) +
                                len(array_right) / len(X) * entropy_multi(array_right))
    return entropy_y - entropy_condition


if __name__ == "__main__":
    X = [[0, 1, 0],
       [1, 1, 0],
       [1, 1, 1],
       [1, 0, 1],
       [1, 1, 1],
       [2, 0, 0],
       [0, 0, 0],
       [2, 1, 0],
       [1, 0, 0],
       [0, 1, 1]]
    y = [0, 0, 1, 0, 1, 1, 0, 1, 1, 1]
    print(information_gain(X, y, 2), entropy_multi(y))
