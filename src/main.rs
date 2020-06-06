use flate2::read::GzDecoder;
use ndarray::prelude::*;
use std::fs::File;
use zero_dl::arr_functions;
use zero_dl::functions::*;
use zero_dl::mnist::{MnistImages, MnistLabels};
use zero_dl::network::{ImplNetworkConfig, Network};

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

    let config = ImplNetworkConfig {
        h_activation_function: arr_functions::sigmoid_arr1,
        a_activation_function: arr_functions::softmax_arr1,
        input_size: 784,
        hidden_size: vec![15],
        output_size: 10,
    };

    let mut network = Network::initialize(config);

    println!("start");
    network.learning(train_x.view(), train_t.view());

    let test_t = MnistLabels::parse(&mut GzDecoder::new(
        File::open("mnist-data/t10k-labels-idx1-ubyte.gz").unwrap(),
    ))
    .unwrap()
    .to_data();

    let test_x = MnistImages::parse(&mut GzDecoder::new(
        File::open("mnist-data/t10k-images-idx3-ubyte.gz").unwrap(),
    ))
    .unwrap()
    .to_data();
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
