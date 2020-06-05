use crate::functions;
use ndarray::prelude::*;

pub fn softmax_arr1(xs: &Array1<f64>) -> Array1<f64> {
    let xs_iter = xs.iter().cloned();
    let c = xs_iter.clone().fold(xs[0], |a, b| a.max(b));
    let exp = xs_iter.clone().map(|x| std::f64::consts::E.powf(x - c));
    let exp_sum = exp.clone().sum::<f64>();
    exp.map(|x| x / exp_sum).collect()
}

pub fn step_arr1(xs: &Array1<f64>) -> Array1<f64> {
    xs.mapv(functions::step)
}

pub fn relu_arr1(xs: &Array1<f64>) -> Array1<f64> {
    xs.mapv(functions::relu)
}

pub fn sigmoid_arr1(xs: &Array1<f64>) -> Array1<f64> {
    xs.mapv(functions::sigmoid)
}

pub fn identity_arr1(xs: &Array1<f64>) -> Array1<f64> {
    xs.mapv(functions::identity)
}

#[test]
fn test_softmax() {
    assert_eq!(
        array![
            0.018211273295547534,
            0.24519181293507392,
            0.7365969137693786
        ],
        softmax_arr1(&array![0.3, 2.9, 4.0])
    );
}
