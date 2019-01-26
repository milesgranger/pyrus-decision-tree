
use pyo3::prelude::*;

pub type Group = Vec<Sample>;
pub type Groups = Vec<Group>;


#[derive(From)]
pub struct Sample {
    pub sample: Vec<i32>,
    pub class: i32
}

impl Sample {
    pub fn new(sample: Vec<i32>, class: i32) -> Sample {
        Sample { sample, class }
    }
}

#[derive(Default, Clone, Serialize, Deserialize, Debug)]
pub struct Node {
    data: i32
}

#[pyclass]
#[derive(Default, Clone, Serialize, Deserialize, Debug)]
pub struct DecisionTree {
    nodes: Vec<Node>
}