#[macro_use]
extern crate zero_dl;

use flate2::read::GzDecoder;
use frunk::labelled::{Field, Transmogrifier};
use frunk::{field, hlist, HCons, HNil};
use frunk_labelled_proc_macro::label;
use ndarray::prelude::*;
use ndarray::Zip;
use rand::prelude::*;
use std::fs::File;
use zero_dl::initializer::Initializer;
use zero_dl::initializers;
use zero_dl::initializers::{random::Random, zero::Zero};
use zero_dl::layer::{LabelledLayers, LabelledOptimizers, Layer, Optimizer, UnconnectedLayer};
use zero_dl::layers::{
    Affine, Convolution, NDimTo2Dim, Placeholder, Pooling, Relu, SoftmaxCrossEntropy, Variable,
};
use zero_dl::mnist::{MnistImages, MnistLabels};

fn main() {
    let train_t = MnistLabels::parse(&mut GzDecoder::new(
        File::open("mnist-data/train-labels-idx1-ubyte.gz").unwrap(),
    ))
    .unwrap()
    .to_one_hot();

    let train_x = MnistImages::parse(&mut GzDecoder::new(
        File::open("mnist-data/train-images-idx3-ubyte.gz").unwrap(),
    ))
    .unwrap()
    .to_arr3(true)
    .insert_axis(Axis(1));

    let test_t = MnistLabels::parse(&mut GzDecoder::new(
        File::open("mnist-data/t10k-labels-idx1-ubyte.gz").unwrap(),
    ))
    .unwrap();

    let test_x = MnistImages::parse(&mut GzDecoder::new(
        File::open("mnist-data/t10k-images-idx3-ubyte.gz").unwrap(),
    ))
    .unwrap()
    .to_arr3(true)
    .insert_axis(Axis(1));

    let mut rng = rand::thread_rng();

    let input_dim = (1, 28, 28);
    let filter_num = 30;
    let filter_size = 5;
    let filter_pad = 0;
    let filter_stride = 1;
    let hidden_size = 100;
    let output_size = 10;
    let input_size = input_dim.1;
    let conv_output_size = (input_size - filter_size + 2 * filter_pad) / filter_stride + 1;
    let pool_output_size = filter_num * (conv_output_size / 2) * (conv_output_size / 2);

    let x = Placeholder::<label!(x), Array4<f32>>::new().join(record! {});
    let t = Placeholder::<label!(t), Array2<f32>>::new().join(record! {});

    let weight1 = Variable::<label!(weight1), Ix4, _>::new(Random::new((
        filter_num,
        input_dim.0,
        filter_size,
        filter_size,
    )))
    .join(record! {});
    let bias1 = Variable::<label!(bias1), Ix1, _>::new(Zero::new((filter_num,))).join(record! {});
    let conv1 = Convolution::new(filter_stride, filter_pad).join(record! {
        x: &x,
        weight: &weight1,
        bias: &bias1
    });
    let relu1 = Relu::new().join(record! {
        x: &conv1
    });
    let pool1 = Pooling::new(2, 2, 2, 0).join(record! {
        x: &relu1
    });

    let nto2 = NDimTo2Dim::new().join(record! {
        x: &pool1
    });

    let weight2 =
        Variable::<label!(weight2), Ix2, _>::new(Random::new((pool_output_size, hidden_size)))
            .join(record! {});
    let bias2 = Variable::<label!(bias2), Ix1, _>::new(Zero::new((hidden_size,))).join(record! {});
    let affine1 = Affine::new().join(record! {
        x: &nto2,
        weight: &weight2,
        bias: &bias2
    });
    let relu2 = Relu::new().join(record! {
        x: &affine1
    });

    let weight3 = Variable::<label!(weight3), Ix2, _>::new(Random::new((hidden_size, output_size)))
        .join(record! {});
    let bias3 = Variable::<label!(bias3), Ix1, _>::new(Zero::new((output_size,))).join(record! {});
    let affine2 = Affine::new().join(record! {
        x: &relu2,
        weight: &weight3,
        bias: &bias3
    });

    let softmax_with_loss = SoftmaxCrossEntropy::new().join(record! {
        x: &affine2,
        t: &t
    });

    let mut variables = softmax_with_loss.initial_variables();

    let iters_num = 100;
    let batch_size = 100;
    let learning_rate = 0.1;

    for n in 0..iters_num {
        let ixs = (0..batch_size)
            .map(|_| rng.gen_range(0, train_x.len_of(Axis(0))))
            .collect::<Vec<_>>();

        let x = train_x.select(Axis(0), &ixs[..]);
        let t = train_t.select(Axis(0), &ixs[..]);
        let (loss, optimizer) = softmax_with_loss.forward(
            record! {
                x: x,
                t: t
            },
            variables.clone(),
        );
        optimizer.optimize(1., variables.to_mut(), learning_rate);

        println!("i:{} loss:{}", n, loss);
    }

    let (ac, per) = ac_per(
        &affine2.forward(record! { x: test_x }, variables.clone()).0,
        &test_t.labels,
    );

    println!("{}/{}", ac, per);
}

fn max_idx(arr: ArrayView1<f32>) -> usize {
    arr.iter()
        .enumerate()
        .fold(
            (0, 0.),
            |(a, max), (i, &cur)| if cur > max { (i, cur) } else { (a, max) },
        )
        .0
}

fn ac_per(x: &Array2<f32>, t: &Array1<u8>) -> (usize, usize) {
    let res = Zip::from(&x.map_axis(Axis(1), |x| max_idx(x)))
        .and(&t.mapv(|t| t as usize))
        .apply_collect(|x, y| if x == y { 1 } else { 0 });

    let per = res.len_of(Axis(0));
    let ac = res.sum();
    (ac, per)
}
