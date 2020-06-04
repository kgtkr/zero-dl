fn main() {
    println!("Hello, world!");
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
