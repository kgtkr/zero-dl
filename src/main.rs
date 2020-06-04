use ndarray::prelude::*;

fn main() {}

#[test]
fn plot_activation_function() {
    graph_plot("step", step_function);
    graph_plot("sigmoid", sigmoid);
    graph_plot("relu", relu);
}

fn graph_plot(name: &str, f: impl Fn(f64) -> f64) {
    use plotlib::page::Page;
    use plotlib::repr::Plot;
    use plotlib::style::LineStyle;
    use plotlib::view::ContinuousView;

    let s = Plot::from_function(f, -6., 6.).line_style(LineStyle::new());

    let v = ContinuousView::new()
        .add(s)
        .x_range(-6., 6.)
        .y_range(-0.5, 1.5)
        .x_label("x")
        .y_label("y");
    Page::single(&v)
        .save(format!("plots/{}.svg", name))
        .unwrap();
}

fn step_function(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else {
        0.0
    }
}

fn relu(x: f64) -> f64 {
    x.max(0.)
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + std::f64::consts::E.powf(-x))
}

fn softmax(xs: &Vec<f64>) -> Vec<f64> {
    let xs_iter = xs.iter().cloned();
    let c = xs_iter.clone().fold(xs[0], |a, b| a.max(b));
    let exp = xs_iter.clone().map(|x| std::f64::consts::E.powf(x - c));
    let exp_sum = exp.clone().sum::<f64>();
    exp.map(|x| x / exp_sum).collect()
}

#[test]
fn test_softmax() {
    assert_eq!(
        vec![
            0.018211273295547534,
            0.24519181293507392,
            0.7365969137693786
        ],
        softmax(&vec![0.3, 2.9, 4.0])
    );
}

fn and(x1: f64, x2: f64) -> f64 {
    let x = vec![x1, x2];
    let w = vec![0.5, 0.5];
    let b = -0.7;
    if x.into_iter().zip(w).map(|(x, w)| x * w).sum::<f64>() + b > 0.0 {
        1.0
    } else {
        0.0
    }
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

#[test]
fn test_and() {
    assert_eq!(0.0, and(0.0, 0.0));
    assert_eq!(0.0, and(1.0, 0.0));
    assert_eq!(0.0, and(0.0, 1.0));
    assert_eq!(1.0, and(1.0, 1.0));
}

fn nand(x1: f64, x2: f64) -> f64 {
    let x = vec![x1, x2];
    let w = vec![-0.5, -0.5];
    let b = 0.7;
    if x.into_iter().zip(w).map(|(x, w)| x * w).sum::<f64>() + b > 0.0 {
        1.0
    } else {
        0.0
    }
}

#[test]
fn test_nand() {
    assert_eq!(1.0, nand(0.0, 0.0));
    assert_eq!(1.0, nand(1.0, 0.0));
    assert_eq!(1.0, nand(0.0, 1.0));
    assert_eq!(0.0, nand(1.0, 1.0));
}

fn or(x1: f64, x2: f64) -> f64 {
    let x = vec![x1, x2];
    let w = vec![0.5, 0.5];
    let b = -0.2;
    if x.into_iter().zip(w).map(|(x, w)| x * w).sum::<f64>() + b > 0.0 {
        1.0
    } else {
        0.0
    }
}

#[test]
fn test_or() {
    assert_eq!(0.0, or(0.0, 0.0));
    assert_eq!(1.0, or(1.0, 0.0));
    assert_eq!(1.0, or(0.0, 1.0));
    assert_eq!(1.0, or(1.0, 1.0));
}

fn xor(x1: f64, x2: f64) -> f64 {
    let s1 = nand(x1, x2);
    let s2 = or(x1, x2);
    let y = and(s1, s2);
    y
}

#[test]
fn test_xor() {
    assert_eq!(0.0, xor(0.0, 0.0));
    assert_eq!(1.0, xor(1.0, 0.0));
    assert_eq!(1.0, xor(0.0, 1.0));
    assert_eq!(0.0, xor(1.0, 1.0));
}
