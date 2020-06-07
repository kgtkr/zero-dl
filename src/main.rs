use flate2::read::GzDecoder;
use frunk::labelled::chars;
use frunk::{field, hlist};
use ndarray::prelude::*;
use ndarray::Zip;
use rand::prelude::*;
use std::fs::File;
use zero_dl::layer::{Layer, Optimizer};
use zero_dl::layers::{Affine, AffineParams, Placeholder, Relu, SoftmaxWithLoss, Variable};
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
    .to_data();

    let train_x = MnistImages::parse(&mut GzDecoder::new(
        File::open("mnist-data/train-images-idx3-ubyte.gz").unwrap(),
    ))
    .unwrap()
    .to_data();

    let test_t = MnistLabels::parse(&mut GzDecoder::new(
        File::open("mnist-data/t10k-labels-idx1-ubyte.gz").unwrap(),
    ))
    .unwrap();

    let test_x = MnistImages::parse(&mut GzDecoder::new(
        File::open("mnist-data/t10k-images-idx3-ubyte.gz").unwrap(),
    ))
    .unwrap()
    .to_data();

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

    let res = Zip::from(
        &affine
            .forward(hlist![field![chars::x, test_x]])
            .0
            .map_axis(Axis(1), |x| max_idx(x)),
    )
    .and(&test_t.labels.mapv(|t| t as usize))
    .apply_collect(|x, y| if x == y { 1 } else { 0 });

    let per = res.len_of(Axis(0));
    let ac = res.sum();

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
