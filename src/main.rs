use flate2::read::GzDecoder;
use frunk::labelled::{chars, Transmogrifier};
use frunk::{field, hlist};
use ndarray::prelude::*;
use rand::prelude::*;
use std::fs::File;
use zero_dl::arr_functions;
use zero_dl::functions::*;
use zero_dl::mnist::{MnistImages, MnistLabels};
use zero_dl::network::{
    Affine, AffineParams, Layer, LayerBackward, LossPlaceholders, Network, Placeholder,
    PredictPlaceholders, Relu, SoftmaxWithLoss, Variable,
};

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

    let params1 = Variable::new(AffineParams::initialize(784, 100));
    let affine1 = Affine::new(&x, &params1);
    let relu1 = Relu::new(&affine1);

    let params2 = Variable::new(AffineParams::initialize(100, 10));
    let affine2 = Affine::new(&relu1, &params2);
    let softmax_with_loss = SoftmaxWithLoss::new(&affine2, &t);

    let iters_num = 10000;

    for n in 0..iters_num {
        let i = rng.gen_range(0, train_x.len_of(Axis(0)));
        let x = train_x.index_axis(Axis(0), i);
        let t = train_t.index_axis(Axis(0), i);
        let (loss, ba) = softmax_with_loss.forward(hlist![
            field![chars::x, x.to_owned()],
            field![chars::t, t.to_owned()]
        ]);
        ba.backward(1.);

        println!("i:{} loss:{}", n, loss);
    }

    let mut succ = 0;

    for i in 0..1000 {
        let x = test_x.index_axis(Axis(0), i);
        let t = test_t.labels[i];

        let answer = max_idx(affine2.forward(hlist![field![chars::x, x.to_owned()]]).0);

        if answer == t as usize {
            succ += 1;
        }
    }

    println!("{}/1000", succ);
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
