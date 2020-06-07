use flate2::read::GzDecoder;
use frunk::labelled::chars;
use frunk::{field, hlist};
use ndarray::prelude::*;
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

    let x = Placeholder::<chars::x, Array1<f32>>::new();
    let t = Placeholder::<chars::t, Array1<f32>>::new();

    let affine = layers!(&x, [784, 1000, 10,]);
    let softmax_with_loss = SoftmaxWithLoss::new(&affine, &t);

    let iters_num = 100;
    let batch_size = 100;
    let learning_rate = 0.1;

    for n in 0..iters_num * batch_size {
        let i = rng.gen_range(0, train_x.len_of(Axis(0)));
        let x = train_x.index_axis(Axis(0), i);
        let t = train_t.index_axis(Axis(0), i);
        let (loss, optimizer) = softmax_with_loss.forward(hlist![
            field![chars::x, x.to_owned()],
            field![chars::t, t.to_owned()]
        ]);
        optimizer.optimize(1., learning_rate);

        if n % batch_size == batch_size - 1 {
            println!("i:{} loss:{}", n, loss);
        }
    }

    let (ac, per) =
        test_x
            .axis_iter(Axis(0))
            .zip(test_t.labels.iter())
            .fold((0, 0), |(ac, per), (x, &t)| {
                let y = max_idx(affine.forward(hlist![field![chars::x, x.to_owned()]]).0);

                (ac + if y == t as usize { 1 } else { 0 }, per + 1)
            });

    println!("{}/{}", ac, per);
}

fn max_idx(arr: Array1<f32>) -> usize {
    arr.iter()
        .enumerate()
        .fold(
            (0, 0.),
            |(a, max), (i, &cur)| if cur > max { (i, cur) } else { (a, max) },
        )
        .0
}
