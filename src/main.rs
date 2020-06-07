use flate2::read::GzDecoder;
use ndarray::prelude::*;
use std::fs::File;
use zero_dl::arr_functions;
use zero_dl::functions::*;
use zero_dl::mnist::{MnistImages, MnistLabels};
use zero_dl::network::{Affine, AffineParams, Network, Placeholder, Relu, Variable, PredictPlaceholders, LossPlaceholders};

fn label_test() {
    use frunk::field;
    use frunk::hlist::Selector;
    use frunk::hlist::{HCons, HList, HNil};
    use frunk::indices::{Here, There};
    use frunk::labelled::{chars, Field, LabelledGeneric, Transmogrifier};
    use zero_dl::hlist_extra::Has;

    #[derive(frunk::LabelledGeneric, Clone, Debug)]
    struct Hoge {
        x: i32,
        y: i64,
    }

    let hoge = Hoge { x: 1, y: 1 };

    let hoge_repr = LabelledGeneric::into(hoge);
    let x = get_x(&hoge_repr);
    let y = get_y(&hoge_repr);

    fn get_x<I, T: Has<chars::x, I, TargetValue = i32>>(obj: &T) -> i32 {
        obj.get()
    }

    fn get_y<I, T: Has<chars::y, I, TargetValue = i64>>(obj: &T) -> i64 {
        obj.get()
    }

    fn set_x<Tail>(obj: Tail, x: i32) -> HCons<Field<chars::x, i32>, Tail> {
        HCons {
            head: field![chars::x, x],
            tail: obj,
        }
    }

    fn set_y<Tail>(obj: Tail, y: i64) -> HCons<Field<chars::y, i64>, Tail> {
        HCons {
            head: field![chars::y, y],
            tail: obj,
        }
    }

    let obj1 = set_y(set_x(HNil, x), y);
    let obj2 = set_x(set_y(HNil, y), x);

    let hoge1: Hoge = LabelledGeneric::from(obj1.transmogrify());
    let hoge2: Hoge = LabelledGeneric::from(obj2.transmogrify());
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

    let params1 = AffineParams::initialize(784, 100);
    let params2 = AffineParams::initialize(100, 10);

    let var1 = Variable::new(params1.clone());
    let var2 = Variable::new(params2.clone());

    let x = Placeholder

    let mut network = Network::initialize((
        Affine::new(params1.clone()),
        (Relu::new(), Affine::new(params2.clone())),
    ));

    println!("start");
    network.learning(&vec![params1, params2], train_x, train_t);

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
            .0
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
