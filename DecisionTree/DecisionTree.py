import math
# for testing the algo
from sklearn import datasets
from sklearn.model_selection import train_test_split

class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y, node={}, depth=0):
        if node is None:
            return None
        elif len(y) == 0:
            return None
        elif all(x == y[0] for x in y):
            return {"val": y[0]}
        elif depth > self.max_depth:
            return None
        else:
            node["entropy"] = entropy_multi(y)
            col, cutoff = best_split(X, y)
            node["feature"] = col
            node["cutoff"] = cutoff
            X_left, X_right = split_data_on_val(X, X, col, cutoff)
            y_left, y_right = split_data_on_val(y, X, col, cutoff)
            node["left"] = self.fit(X_left, y_left, {}, depth + 1)
            node["right"] = self.fit(X_right, y_right, {}, depth + 1)
        self.root = node
        return node

    def predict(self, X):
        pred = []
        for row in X:
            node = self.root
            while node:
                if len(node.keys()) == 1:
                    pred.append(node["val"])
                    break
                if row[node["feature"]] < node["cutoff"]:
                    node = node["left"]
                else:
                    node = node["right"]
        return pred


def split_data_on_val(data, ref_data, col, val):
    result_left = []
    result_right = []
    for i, row in enumerate(ref_data):
        if row[col] < val:
            result_left.append(data[i])
        elif row[col] >= val:
            result_right.append(data[i])
    return result_left, result_right


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

def best_split(X, y):
    max_info_gain = float("-inf")
    best_split_feature = None
    best_cutoff = None
    for i in range(len(X[0])):
        info_gain, cutoff = information_gain(X, y, i)
        if info_gain > max_info_gain:
            max_info_gain = info_gain
            best_split_feature = i
            best_cutoff = cutoff
    return best_split_feature, best_cutoff


def information_gain(X, y, element_index):
    """
    IG(Y|element)
    """
    s = set([row[element_index] for row in X])
    entropy_y = entropy_multi(y)
    entropy_condition = float("inf")
    best_split = None
    for item in s:
        array_left = []
        array_right = []
        for i in range(len(X)):
            if X[i][element_index] < item:
                array_left.append(y[i])
            else:
                array_right.append(y[i])
        curr_entropy = len(array_left) / len(X) * entropy_multi(array_left) + len(array_right) / len(X) * \
                       entropy_multi(array_right)
        if curr_entropy < entropy_condition:
            entropy_condition = curr_entropy
            best_split = item
    return entropy_y - entropy_condition, best_split


def accuracy(prediction, actual):
    count = 0
    for i in range(len(prediction)):
        if prediction[i] == actual[i]:
            count += 1
    return count/len(prediction)


if __name__ == "__main__":
    data = datasets.load_iris()
    X = data["data"].tolist()
    y = data["target"].tolist()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    clf = DecisionTreeClassifier(max_depth=6)
    print(clf.fit(X_train, y_train))
    pred = clf.predict(X_test)
    print(accuracy(pred, y_test))
