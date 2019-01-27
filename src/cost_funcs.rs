
use crate::*;

/// Calculate the gini index based on groups of samples.
/// `Vec<Vec<Sample>> Where the inner most is one sample, the middle
/// is a collection of samples, where all are assumed to be the same class
/// and the outermost are the 'groups'
/// **Will panic if any provided sample does not have an assigned class**
pub fn gini_index(groups: (Vec<&Sample>, Vec<&Sample>), classes: &Vec<i32>) -> f32 {

    let groups = vec![groups.0, groups.1];

    let n_samples = groups
        .iter()
        .map(|group| group.len())
        .sum::<usize>() as f32;

    groups
        .iter()
        .filter(|group| group.len() > 0)
        .map(|group| {
            let score = classes
                .iter()
                .map(|class| {
                    group
                        .iter()
                        .filter(|sample| &sample.class.unwrap() == class)
                        .count() as f32 / group.len() as f32
                })
                .fold(0.0, |acc, p| acc + (p * p));

            (1.0 - score) * (group.len() as f32 / n_samples)
        })
        .sum::<f32>()
}