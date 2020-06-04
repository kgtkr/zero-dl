use ndarray::prelude::*;

fn main() {
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
fn ndarray_test() {
    let a = array![[1., 2.], [3., 4.]];
    let b = array![[5., 6.], [7., 8.]];

    println!("{:?} {:?} {:?}", a.shape(), b.shape(), a.dot(&b));
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
