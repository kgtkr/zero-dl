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

pub fn im2col(
    input_data: Array4<f32>,
    filter_h: usize,
    filter_w: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
) -> (Array2<f32>, usize, usize) {
    let (n, c, h, w) = input_data.dim();

    let out_h = (h + 2 * pad_h - filter_h) / stride_h + 1;
    let out_w = (w + 2 * pad_w - filter_w) / stride_w + 1;

    // input_data.pad([(0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)])
    let mut img = Array::zeros((
        input_data.len_of(Axis(0)),
        input_data.len_of(Axis(1)),
        input_data.len_of(Axis(2)) + pad_h * 2,
        input_data.len_of(Axis(3)) + pad_w * 2,
    ));

    for i1 in 0..input_data.len_of(Axis(0)) {
        for i2 in 0..input_data.len_of(Axis(1)) {
            for i3 in 0..input_data.len_of(Axis(2)) {
                for i4 in 0..input_data.len_of(Axis(3)) {
                    img[(i1, i2, i3 + pad_h, i4 + pad_w)] = input_data[(i1, i2, i3, i4)];
                }
            }
        }
    }

    let mut col = Array::zeros((n, c, filter_h, filter_w, out_h, out_w));

    for y in 0..filter_h {
        let y_max = y + stride_h * out_h;
        for x in 0..filter_w {
            let x_max = x + stride_w * out_w;
            col.slice_mut(s![.., .., y, x, .., ..])
                .assign(&img.slice(s![.., .., y..y_max;stride_h, x..x_max;stride_w]));
        }
    }

    let s1 = n * out_h * out_w;
    let s2 = col.len() / s1;

    let col = col
        .permuted_axes([0, 4, 5, 1, 2, 3])
        .into_shape((n * out_h * out_w, s2))
        .unwrap();
    (col, out_h, out_w)
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
