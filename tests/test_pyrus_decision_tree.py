# -*- coding: utf-8 -*-

import unittest
import logging
import numpy as np

logger = logging.getLogger(__name__)


class PyrusDecisionTreeTestCase(unittest.TestCase):

    def test_import(self):
        """
        At least we can import the damn thing?
        """
        from pyrus_decision_tree import PyrusDecisionTree

    def test_tree_sanity(self):
        """
        Basic sanity test that the tree can actually learn something by re-predicting on
        training data.
        """
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
        self.assertEqual(targets, predictions)

if __name__ == '__main__':
    unittest.main()
