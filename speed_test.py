import timeit
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from pyrus_decision_tree import PyrusDecisionTree

if __name__ == '__main__':

    X = np.random.random(100000).reshape(-1, 10)
    y = np.random.randint(0, 5, X.shape[0])

    clf = DecisionTreeClassifier(max_depth=10)
    number = 1
    avg = timeit.timeit('clf.fit(X, y)', number=number, globals=globals()) / number
    print(f"Scikit-Learn time: {avg:.4f}")

    clf = PyrusDecisionTree(10)
    avg = timeit.timeit('clf.fit(X, y)', number=number, globals=globals()) / number
    print(f"Rust time: {avg:.4f}")
