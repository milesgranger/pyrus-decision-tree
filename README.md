# pyrus-decision-tree
Decision Tree written in Rust, with Python bindings

## Extremely fast and efficient Decision Tree written in Rust 

*This is currently the first release and the tree only implements 
the `scikit-learn` API's `fit(X, y)` and `predict(X)` methods as of now, 
and only as a classifier (no regression tree yet)*

In a short time, it should be a drop-in replacement for both 
`DecisionTreeClassifier` and `DecisionTreeRegressor` :-)

---

#### Install:
`pip install --upgrade pyrus-decision-tree`

#### Uninstall:
`pip uninstall pyrus-decision-tree`

---

#### Use:
```python
from pyrus_decision_tree import PyrusDecisionTree

dataset = [[2.771244718, 1.7847839292],
           [1.728571309, 1.1697614132],
           [3.678319846, 2.812813571],
           [3.961043357, 2.619950321],
           [2.999208922, 2.209014212],
           [7.497545867, 3.162953546],
           [9.00220326,  3.339047188],
           [7.444542326, 0.476683375],
           [10.12493903, 3.234550982],
           [6.642287351, 3.319983761]]
targets = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

clf = PyrusDecisionTree(5)
clf.fit(dataset, targets)
predictions = clf.predict(dataset)
```