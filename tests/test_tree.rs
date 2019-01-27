extern crate pyrus_decision_tree;
extern crate float_cmp;

use float_cmp::ApproxEq;
use pyrus_decision_tree::{DecisionTree, cost_funcs, Sample};

#[test]
fn test_tree() {

    let data = vec![
        vec![2.771244718,   1.784783929],
	    vec![1.728571309,   1.169761413],
	    vec![3.678319846,   2.81281357],
	    vec![3.961043357,   2.61995032],
        vec![2.999208922,   2.209014212],
	    vec![7.497545867,   3.162953546],
	    vec![9.00220326,    3.339047188],
	    vec![7.444542326,   0.476683375],
	    vec![10.12493903,   3.234550982],
	    vec![6.642287351,   3.319983761]
    ];

    let y = vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1];

    let mut clf = DecisionTree::new(3);
    assert!(clf.fit(data.clone(), y.clone()).is_ok());

    let predictions = clf.predict(data);
    assert_eq!(predictions, y);
    println!("Predictions: {:#?}", &predictions);

}

#[test]
fn test_gini_index() {

    // Expected gini of 0.5
    let group1 = vec![
        Sample::new(vec![1., 1.], Some(1)),
        Sample::new(vec![1., 0.], Some(0))
    ];

    let group2 = vec![
        Sample::new(vec![1., 1.], Some(1)),
        Sample::new(vec![1., 0.], Some(0))
    ];

    let groups = (
        group1.iter().collect::<Vec<&Sample>>(),
        group2.iter().collect::<Vec<&Sample>>()
    );

    let gini = cost_funcs::gini_index(groups, &vec![0, 1]);
    assert_eq!(gini, 0.5);

    // Expected gini of 0.0
    let group1 = vec![
        Sample::new(vec![1., 1.], Some(0)),
        Sample::new(vec![1., 1.], Some(0))
    ];

    let group2 = vec![
        Sample::new(vec![1., 0.], Some(1)),
        Sample::new(vec![1., 0.], Some(1))
    ];

    let groups = (
        group1.iter().collect::<Vec<&Sample>>(),
        group2.iter().collect::<Vec<&Sample>>()
    );

    let gini = cost_funcs::gini_index(groups, &vec![0, 1]);
    assert_eq!(gini, 0.0);
}