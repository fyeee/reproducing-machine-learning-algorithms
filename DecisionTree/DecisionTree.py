import math

def entropy(x, n):
    return -1 * math.log(x / n, 2) * (x / n)


def entropy_multi(array):
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
