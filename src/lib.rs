#![feature(specialization, duration_as_u128)]

#[macro_use] extern crate serde_derive;
#[macro_use] extern crate derive_more;

mod tree;
pub mod cost_funcs;

pub use crate::tree::{DecisionTree, Sample, Groups, Group};
