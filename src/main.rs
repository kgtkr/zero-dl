use flate2::read::GzDecoder;
use frunk::labelled::{chars, Transmogrifier};
use frunk::{field, hlist};
use ndarray::prelude::*;
use ndarray::Zip;
use rand::prelude::*;
use std::fs::File;
use zero_dl::arr_functions;
use zero_dl::functions::*;
use zero_dl::mnist::{MnistImages, MnistLabels};
use zero_dl::network::{
    Affine, AffineParams, Layer, Optimizer, Placeholder, Relu, SoftmaxWithLoss, Variable,
};

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

    let iters_num = 1000;
    let batch_size = 1000;

    for n in 0..iters_num * batch_size {
        let i = rng.gen_range(0, train_x.len_of(Axis(0)));
        let x = train_x.index_axis(Axis(0), i);
        let t = train_t.index_axis(Axis(0), i);
        let (loss, ba) = softmax_with_loss.forward(hlist![
            field![chars::x, x.to_owned()],
            field![chars::t, t.to_owned()]
        ]);
        ba.optimize(1.);

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

#[test]
fn test_3_nn() {
    let x = array![1., 0.5];
    let w1 = array![[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]];
    let b1 = array![0.1, 0.2, 0.3];

    let a1 = x.dot(&w1) + b1;
    let z1 = a1.map(|&x| sigmoid(x));

    assert_eq!(array![0.30000000000000004, 0.7, 1.1], a1);
    assert_eq!(
        array![0.574442516811659, 0.668187772168166, 0.7502601055951177],
        z1
    );

    let w2 = array![[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]];
    let b2 = array![0.1, 0.2];

    let a2 = z1.dot(&w2) + b2;
    let z2 = a2.map(|&x| sigmoid(x));

    let w3 = array![[0.1, 0.3], [0.2, 0.4]];
    let b3 = array![0.1, 0.2];

    let a3 = z2.dot(&w3) + b3;
    let y = a3.map(|&x| x);
    assert_eq!(array![0.31682707641102975, 0.6962790898619668], y);
}

#[test]
fn ndarray_test() {
    let a = array![[1., 2.], [3., 4.]];
    let b = array![[5., 6.], [7., 8.]];

    assert_eq!(&[2, 2], a.shape());
    assert_eq!(&[2, 2], b.shape());
    assert_eq!(array![[19., 22.], [43., 50.]], a.dot(&b));
}
