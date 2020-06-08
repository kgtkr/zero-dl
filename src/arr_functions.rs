use crate::functions;
use ndarray::prelude::*;
use ndarray::Zip;
use std::ops::AddAssign;

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

pub fn im2col(
    input_data: ArrayView4<f32>,
    filter_h: usize,
    filter_w: usize,
    stride: usize,
    pad: usize,
) -> (Array2<f32>, usize, usize) {
    let (n, c, h, w) = input_data.dim();

    let out_h = (h + 2 * pad - filter_h) / stride + 1;
    let out_w = (w + 2 * pad - filter_w) / stride + 1;

    let mut img = Array::zeros((
        input_data.len_of(Axis(0)),
        input_data.len_of(Axis(1)),
        input_data.len_of(Axis(2)) + pad * 2,
        input_data.len_of(Axis(3)) + pad * 2,
    ));
    img.slice_mut(s![
        ..,
        ..,
        pad..img.len_of(Axis(2)) - pad,
        pad..img.len_of(Axis(3)) - pad
    ])
    .assign(&input_data);

    let img_w = img.len_of(Axis(2));
    let img_h = img.len_of(Axis(3));

    let mut col = Array::zeros((n, c, filter_h, filter_w, out_h, out_w));

    for y in 0..filter_h {
        let y_max = (y + stride * out_h).min(img_h);
        for x in 0..filter_w {
            let x_max = (x + stride * out_w).min(img_w);

            col.slice_mut(s![.., .., y, x, .., ..])
                .assign(&img.slice(s![.., .., y..y_max;stride, x..x_max;stride]));
        }
    }

    let s1 = n * out_h * out_w;
    let s2 = col.len() / s1;

    let col = col
        .permuted_axes([0, 4, 5, 1, 2, 3])
        .to_shared()
        .reshape((n * out_h * out_w, s2))
        .to_owned();
    (col, out_h, out_w)
}

pub fn col2im(
    col: ArrayView2<f32>,
    input_shape: (usize, usize, usize, usize),
    filter_h: usize,
    filter_w: usize,
    stride: usize,
    pad: usize,
) -> Array4<f32> {
    let (n, c, h, w) = input_shape;
    let out_h = (h + 2 * pad - filter_h) / stride + 1;
    let out_w = (w + 2 * pad - filter_w) / stride + 1;
    let col = col
        .to_shared()
        .reshape((n, out_h, out_w, c, filter_h, filter_w))
        .permuted_axes([0, 3, 4, 5, 1, 2]);

    let mut img = Array::zeros((n, c, h + 2 * pad + stride - 1, w + 2 * pad + stride - 1));
    for y in 0..filter_h {
        let y_max = y + stride * out_h;
        for x in 0..filter_w {
            let x_max = x + stride * out_w;
            img.slice_mut(s![..,..,y..y_max;stride, x..x_max;stride])
                .add_assign(&col.slice(s![.., .., y, x, .., ..]));
        }
    }

    img.slice(s![.., .., pad..h + pad, pad..w + pad]).to_owned()
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
