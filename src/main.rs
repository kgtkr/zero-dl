use flate2::read::GzDecoder;
use frunk::labelled::chars;
use frunk::{field, hlist};
use ndarray::prelude::*;
use ndarray::Zip;
use rand::prelude::*;
use std::fs::File;
use zero_dl::layer::{Layer, Optimizer};
use zero_dl::layers::{
    Affine, AffineParams, Convolution, ConvolutionParams, NDimTo2Dim, Placeholder, Pooling, Relu,
    SoftmaxWithLoss, Variable,
};
use zero_dl::mnist::{MnistImages, MnistLabels};

macro_rules! layers {
    ($x: expr,[$prev_n: expr, $n: expr, $next_n: expr, $( $rest: expr, ) *]) => {
        {
            let params = Variable::new(AffineParams::initialize($prev_n, $n));
            let affine = Affine::new($x, params);
            let relu = Relu::new(affine);

            layers!(relu, [$n, $next_n, $($rest,)*])
        }
    };

    ($x: expr, [$prev_n: expr, $n: expr,] ) => {
        {
            let params = Variable::new(AffineParams::initialize($prev_n, $n));
            let affine = Affine::new($x, params);

            affine
        }
    };

}

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

    let input_dim = (0, 28, 28);
    let filter_num = 30;
    let filter_size = 5;
    let filter_pad = 0;
    let filter_stride = 1;
    let hidden_size = 100;
    let output_size = 10;
    let input_size = input_dim.1;
    let conv_output_size = (input_size - filter_size + 2 * filter_pad) / filter_stride + 1;
    let pool_output_size = filter_num * (conv_output_size / 2) * (conv_output_size / 2);

    let x = Placeholder::<chars::x, Array4<f32>>::new();
    let t = Placeholder::<chars::t, Array2<f32>>::new();

    let params1 = Variable::new(ConvolutionParams::initialize(
        filter_num,
        input_dim.0,
        filter_size,
        filter_size,
    ));
    let conv1 = Convolution::new(&x, &params1, filter_stride, filter_pad);
    let relu1 = Relu::new(&conv1);
    let pool1 = Pooling::new(&relu1, 2, 2, 2, 0);

    let nto2 = NDimTo2Dim::new(&pool1);
    let params2 = Variable::new(AffineParams::initialize(pool_output_size, hidden_size));
    let affine1 = Affine::new(&nto2, &params2);
    let relu2 = Relu::new(&affine1);

    let params3 = Variable::new(AffineParams::initialize(hidden_size, output_size));
    let affine2 = Affine::new(&relu2, &params3);

    let softmax_with_loss = SoftmaxWithLoss::new(&affine2, &t);

    let iters_num = 100;
    let batch_size = 100;
    let learning_rate = 0.1;

    for n in 0..iters_num {
        let ixs = (0..batch_size)
            .map(|_| rng.gen_range(0, train_x.len_of(Axis(0))))
            .collect::<Vec<_>>();

        let x = train_x.select(Axis(0), &ixs[..]);
        let t = train_t.select(Axis(0), &ixs[..]);
        let (loss, optimizer) =
            softmax_with_loss.forward(hlist![field![chars::x, x], field![chars::t, t]]);
        optimizer.optimize(1., learning_rate);

        println!("i:{} loss:{}", n, loss);
    }

    let (ac, per) = ac_per(
        &affine2.forward(hlist![field![chars::x, test_x]]).0,
        &test_t.labels,
    );

    println!("{}/{}", ac, per);
}

fn _main() {
    let train_t = MnistLabels::parse(&mut GzDecoder::new(
        File::open("mnist-data/train-labels-idx1-ubyte.gz").unwrap(),
    ))
    .unwrap()
    .to_one_hot();

    let train_x = MnistImages::parse(&mut GzDecoder::new(
        File::open("mnist-data/train-images-idx3-ubyte.gz").unwrap(),
    ))
    .unwrap()
    .to_arr2(true);

    let test_t = MnistLabels::parse(&mut GzDecoder::new(
        File::open("mnist-data/t10k-labels-idx1-ubyte.gz").unwrap(),
    ))
    .unwrap();

    let test_x = MnistImages::parse(&mut GzDecoder::new(
        File::open("mnist-data/t10k-images-idx3-ubyte.gz").unwrap(),
    ))
    .unwrap()
    .to_arr2(true);

    let mut rng = rand::thread_rng();

    let x = Placeholder::<chars::x, Array2<f32>>::new();
    let t = Placeholder::<chars::t, Array2<f32>>::new();

    let affine = layers!(&x, [784, 1000, 10,]);

    let softmax_with_loss = SoftmaxWithLoss::new(&affine, &t);

    let iters_num = 100;
    let batch_size = 100;
    let learning_rate = 0.1;

    for n in 0..iters_num {
        let ixs = (0..batch_size)
            .map(|_| rng.gen_range(0, train_x.len_of(Axis(0))))
            .collect::<Vec<_>>();

        let x = train_x.select(Axis(0), &ixs[..]);
        let t = train_t.select(Axis(0), &ixs[..]);
        let (loss, optimizer) =
            softmax_with_loss.forward(hlist![field![chars::x, x], field![chars::t, t]]);
        optimizer.optimize(1., learning_rate);

        println!("i:{} loss:{}", n, loss);
    }

    let (ac, per) = ac_per(
        &affine.forward(hlist![field![chars::x, test_x]]).0,
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
