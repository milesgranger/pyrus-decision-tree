extern crate pyrus_decision_tree;
extern crate float_cmp;

use float_cmp::ApproxEq;
use pyrus_decision_tree::{DecisionTree, cost_funcs, Sample};

#[test]
fn test_tree() {
    let mut clf = DecisionTree::default();
}

#[test]
fn test_gini_index() {

    // Expected gini of 0.5
    let groups = vec![
        vec![
            Sample::new(vec![1, 1], 1),
            Sample::new(vec![1, 0], 0)
        ],
        vec![
            Sample::new(vec![1, 1], 1),
            Sample::new(vec![1, 0], 0)
        ]
    ];

    let gini = cost_funcs::gini_index(groups, vec![0, 1]);
    assert_eq!(gini, 0.5);

    // Expected gini of 0.0
    let groups = vec![
        vec![
            Sample::new(vec![1, 1], 1),
            Sample::new(vec![1, 1], 1)
        ],
        vec![
            Sample::new(vec![1, 0], 0),
            Sample::new(vec![1, 0], 0)
        ]
    ];

    let gini = cost_funcs::gini_index(groups, vec![1, 0]);
    assert_eq!(gini, 0.0);
}