use crate::functions;
use ndarray::prelude::*;
use ndarray::Zip;

pub fn softmax_batch(a: ArrayView2<f32>) -> Array2<f32> {
    let c = a.fold(0f32, |a, &b| a.max(b));
    let exp_a = a.map(|x| std::f32::consts::E.powf(x - c));
    let exp_sum = exp_a.sum_axis(Axis(1)).insert_axis(Axis(1));
    let y = exp_a / exp_sum;
    y
}

pub fn step_arr1(xs: ArrayView1<f32>) -> Array1<f32> {
    xs.mapv(functions::step)
}

pub fn relu_arr1(xs: ArrayView1<f32>) -> Array1<f32> {
    xs.mapv(functions::relu)
}

pub fn sigmoid_arr1(xs: ArrayView1<f32>) -> Array1<f32> {
    xs.mapv(functions::sigmoid)
}

pub fn identity_arr1(xs: ArrayView1<f32>) -> Array1<f32> {
    xs.mapv(functions::identity)
}

fn cross_entropy_error_scalar(x: f32, t: f32) -> f32 {
    let delta = 1e-7;
    -t * (x + delta).log(std::f32::consts::E)
}

pub fn cross_entropy_error(x: ArrayView1<f32>, t: ArrayView1<f32>) -> f32 {
    Zip::from(x)
        .and(t)
        .apply_collect(|&x, &t| cross_entropy_error_scalar(x, t))
        .sum()
}

pub fn cross_entropy_error_batch(x: ArrayView2<f32>, t: ArrayView2<f32>) -> f32 {
    let batch_size = x.len_of(Axis(0));
    Zip::from(x)
        .and(t)
        .apply_collect(|&x, &t| cross_entropy_error_scalar(x, t))
        .sum()
        / batch_size as f32
}

pub fn sum_squared_error(x: ArrayView1<f32>, t: ArrayView1<f32>) -> f32 {
    0.5 * Zip::from(x)
        .and(t)
        .apply_collect(|x, t| (x - t).powi(2))
        .sum()
}

#[test]
fn test_softmax() {
    assert_eq!(
        array![[
            0.018211273295547534,
            0.24519181293507392,
            0.7365969137693786
        ]],
        softmax_batch(array![[0.3, 2.9, 4.0]].view())
    );
}
