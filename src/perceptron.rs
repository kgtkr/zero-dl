use crate::activation_function::{ActivationFunction, Step};
use crate::hlist_extra::AList;
use frunk::{HCons, HNil};
use generic_array::{arr, ArrayLength, GenericArray};
use std::marker::PhantomData;
use typenum::{IsLess, U2};

pub trait PerceptronHList {}

impl PerceptronHList for HNil {}

impl<H: Perceptron, T: PerceptronHList> PerceptronHList for HCons<H, T> {}

/*
x1 <- input(0)
x2 <- input(1)
s1 = nand(x1, x2)
s2 = or(x1, x2)
return and(s1, s2)

*/

// hlist使いたい
pub trait Perceptron {
    type InputN;
    type PlacefolderN;

    fn exec(&self, xs: &impl AList<f64>) -> f64;
}

impl Perceptron<U2> {
    pub fn and() -> impl Perceptron<U2> {
        SimplePerceptron {
            ws: arr![f64;0.5,0.5],
            b: -0.7,
            f: Step::new(0.0),
        }
    }

    pub fn nand() -> impl Perceptron<U2> {
        SimplePerceptron {
            ws: arr![f64;-0.5,-0.5],
            b: 0.7,
            f: Step::new(0.0),
        }
    }

    pub fn or() -> impl Perceptron<U2> {
        SimplePerceptron {
            ws: arr![f64;0.5,0.5],
            b: -0.2,
            f: Step::new(0.0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SimplePerceptron<N: ArrayLength<f64>, F: ActivationFunction> {
    ws: GenericArray<f64, N>,
    b: f64,
    f: F,
}

impl<N: ArrayLength<f64>, F: ActivationFunction> Perceptron<N> for SimplePerceptron<N, F> {
    fn exec(&self, xs: &GenericArray<f64, N>) -> f64 {
        self.f
            .run(xs.iter().zip(&self.ws).map(|(x, w)| x * w).sum::<f64>() + self.b)
    }
}

#[derive(Debug, Clone)]
pub struct InputPerceptron<I: ArrayLength<f64>>(PhantomData<I>);

impl<I: ArrayLength<f64>> InputPerceptron<I> {
    pub fn new() -> InputPerceptron<I> {
        InputPerceptron(PhantomData)
    }
}

impl<N: ArrayLength<f64>, I: IsLess<N> + ArrayLength<f64>> Perceptron<N> for InputPerceptron<I> {
    fn exec(&self, xs: &GenericArray<f64, N>) -> f64 {
        xs[I::to_usize()]
    }
}
