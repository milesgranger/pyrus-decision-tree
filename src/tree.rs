
use std::f32;
use std::time::{Duration, Instant};

use pyo3::prelude::*;
use stats;
use itertools::{Itertools, Either};
use rayon::prelude::*;

use crate::cost_funcs;

pub type Group = Vec<Sample>;
pub type Groups = Vec<Group>;
pub type Number = f32;

// Splits a dataset into two groups, given a feature and the value of that feature
fn split_into_groups<'b>(index: usize, value: Number, dataset: &'b Vec<&Sample>) -> (Vec<&'b Sample>, Vec<&'b Sample>) {
    let (left, right): (Vec<_>, Vec<_>) = dataset
        .iter()
        .partition_map(|sample| {
            if sample.data[index] < value {
                Either::Left(sample)
            } else {
                Either::Right(sample)
            }
        });
    (left, right)
}

// Determine the best feature to split on
// Returns a tuple of (best feature split index, best feature split value, best gini)
// will panic if any sample does not have an assigned class
fn best_split(x: &Vec<&Sample>) -> (usize, Number, f32) {
    //let now = Instant::now();

    let classes: Vec<i32> = x
        .iter()
        .map(|s| s.class.expect("Does not have an assigned class!"))
        .unique()
        .into_iter()
        .collect();

    let n_features = x[0].data.len();

    let mut gini_splits = x.par_iter()
        .map(|sample| {
            (0..n_features)
                .map(|idx| {
                    let groups = split_into_groups(idx, sample.data[idx], &x);
                    let gini = cost_funcs::gini_index(groups, &classes);
                    (idx, sample.data[idx], gini)
                })
                .collect::<Vec<(usize, Number, f32)>>()
        })
        .flat_map(|vals| vals)
        .collect::<Vec<(usize, Number, f32)>>();

    gini_splits.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
    //println!("Finished best_split in {}ms", now.elapsed().as_millis());
    gini_splits[0]
}

#[derive(From)]
pub struct Sample {
    pub data: Vec<Number>,
    pub class: Option<i32>
}

impl Sample {
    pub fn new(data: Vec<Number>, class: Option<i32>) -> Sample {
        Sample { data, class }
    }
}

#[derive(Default, Clone, Serialize, Deserialize, Debug)]
pub struct Node {
    gini: f32,
    depth: usize,
    split_value: Number,
    split_index: usize,
    target: Option<i32>,
    left_child_node: Option<Box<Node>>,
    right_child_node: Option<Box<Node>>
}

impl Node {

    pub fn new(depth: usize, max_depth: usize, dataset: &Vec<&Sample>) -> Node {
        let (split_index, split_value, gini): (usize, Number, f32) = best_split(&dataset);

        // No more growing if we're at the max depth, or min samples
        if max_depth == depth || dataset.len() < 3 {

            // Calculate the target of this node
            let modes = stats::modes(dataset.iter().map(|sample| sample.class.unwrap()));
            let target;
            if modes.len() > 0 {
                target = Some(modes[0])
            } else {
                target = Some(dataset[0].class.unwrap());
            };

            Node {
                gini, depth, split_value, split_index, left_child_node: None, right_child_node: None, target
            }

        } else {

            let (left, right): (Vec<&Sample>, Vec<&Sample>) = split_into_groups(split_index, split_value, &dataset);

            let left_child_node = if left.len() > 0 {
                Some(Box::new(Node::new(depth + 1, max_depth, &left)))
            } else {
                None
            };

            let right_child_node = if right.len() > 0 {
                Some(Box::new(Node::new(depth + 1, max_depth, &right)))
            } else {
                None
            };

            Node {
                gini, depth, split_value, split_index, left_child_node, right_child_node, target: None
            }
        }
    }

    pub fn predict(&self, x: &Sample) -> i32 {
        match self.target {

            // If we're the terminal node (we have a target) then we must make a prediction
            Some(target) => {
                target
            },

            // Otherwise, we split and pass of prediction to our child node(s)
            None => {
                if x.data[self.split_index] < self.split_value {
                    match &self.left_child_node {
                        Some(node) => node.predict(&x),
                        None => panic!("Expected a left child node!")
                    }
                } else {
                    match &self.right_child_node{
                        Some(node) => node.predict(&x),
                        None => panic!("Expected a right child node!")
                    }
                }
            }
        }
    }
}


#[derive(Default, Clone, Serialize, Deserialize, Debug)]
pub struct DecisionTree {
    root_node: Node,
    max_depth: usize,
}

impl DecisionTree {

    pub fn new(max_depth: usize) -> DecisionTree {
        DecisionTree { max_depth, root_node: Node::default() }
    }

    pub fn fit(&mut self, x: Vec<Vec<Number>>, y: Vec<i32>) -> PyResult<()> {

        let x = x.into_iter()
            .zip(y.into_iter())
            .map(|(sample, target)| Sample::new(sample, Some(target)))
            .collect::<Vec<Sample>>();

        let now = Instant::now();
        let dataset = x
            .iter()
            .collect::<Vec<&Sample>>();

        let now = Instant::now();
        let root_node = Node::new(1, self.max_depth, &dataset);

        self.root_node = root_node;

        Ok(())
    }

    pub fn predict(&self, x: Vec<Vec<Number>>) -> Vec<i32> {
        x.into_iter()
            .map(|row| Sample::new(row, None))
            .map(|sample| self.root_node.predict(&sample))
            .collect::<Vec<i32>>()
    }

}

#[pyclass]
#[derive(Default, Clone, Serialize, Deserialize, Debug)]
pub struct PyrusDecisionTree {
    model: DecisionTree
}

#[pymethods]
impl PyrusDecisionTree {

    #[new]
    pub fn __new__(obj: &PyRawObject, max_depth: usize) -> PyResult<()> {
        obj.init(|_| PyrusDecisionTree {
            model: DecisionTree::new(max_depth)
        })?;
        Ok(())
    }

    pub fn fit(&mut self, x: Vec<Vec<Number>>, y: Vec<i32>) -> PyResult<()> {
        self.model.fit(x, y)
    }

    pub fn predict(&self, x: Vec<Vec<Number>>) -> PyResult<Vec<i32>> {
        Ok(self.model.predict(x))
    }

}

#[pymodinit]
pub fn pyrus_decision_tree(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyrusDecisionTree>()?;
    Ok(())
}
