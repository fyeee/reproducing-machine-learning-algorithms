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
    entropy_condition = 0
    for item in s:
        array = []
        for i in range(len(X)):
            if X[i][element_index] == item:
                array.append(y[i])
        entropy_condition += len(array) / len(X) * entropy_multi(array)
    return entropy_y - entropy_condition


if __name__ == "__main__":
    print(entropy_multi([1, 1, 1, 1, 1, 0]))
