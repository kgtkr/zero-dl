use flate2::read::GzDecoder;
use ndarray::prelude::*;
use std::fs::File;
use zero_dl::arr_functions;
use zero_dl::functions::*;
use zero_dl::mnist::{MnistImages, MnistLabels};
use zero_dl::network::{Affine, ImplNetworkConfig, Network2, Relu};

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

    let mut network = Network2::initialize(
        (Affine::new(0), (Relu::new(), Affine::new(1))),
        784,
        vec![100],
        10,
    );

    println!("start");
    network.learning(train_x, train_t);

    let test_t = MnistLabels::parse(&mut GzDecoder::new(
        File::open("mnist-data/t10k-labels-idx1-ubyte.gz").unwrap(),
    ))
    .unwrap();

    let test_x = MnistImages::parse(&mut GzDecoder::new(
        File::open("mnist-data/t10k-images-idx3-ubyte.gz").unwrap(),
    ))
    .unwrap()
    .to_data();

    let mut succ = 0;

    for i in 0..1000 {
        let x = test_x.index_axis(Axis(0), i);
        let t = test_t.labels[i];

        let answer = network
            .predict(x.to_owned())
            .iter()
            .enumerate()
            .fold(
                (0, 0.),
                |(a, max), (i, &cur)| if cur > max { (i, cur) } else { (a, max) },
            )
            .0;

        if answer == t as usize {
            succ += 1;
        }
    }

    println!("{}/1000", succ);
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
