use crate::layer::{UnconnectedLayer, UnconnectedOptimizer};
use frunk::labelled::{ByNameFieldPlucker, Field};
use frunk::traits::ToMut;
use frunk::{field, hlist, HNil, Hlist};
use frunk_labelled_proc_macro::label;
use ndarray::prelude::*;
use ndarray::Zip;
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;
use std::cell::RefCell;
use std::sync::Arc;

pub struct AffineOptimizer {
    pub weight: Array2<f32>,
    pub bias: Array1<f32>,
    pub x: Array2<f32>,
}

impl UnconnectedOptimizer for AffineOptimizer {
    type Inputs = Record! {
        x: Array2<f32>,
        weight: Array2<f32>,
        bias: Array1<f32>
    };
    type Output = Array2<f32>;
    type Variables = HNil;

    fn optimize<'a>(
        self,
        dout: Self::Output,
        variables: <Self::Variables as ToMut<'a>>::Output,
        learning_rate: f32,
    ) -> Self::Inputs {
        let dx = dout.dot(&self.weight.t());

        let dw = self.x.t().dot(&dout);
        let db = dout.sum_axis(Axis(0));

        record! {
            x: dx,
            weight: dw,
            bias: db
        }
    }
}

pub struct Affine {}

impl Affine
where
    Self: UnconnectedLayer,
{
    pub fn new() -> Self {
        Affine {}
    }
}

impl UnconnectedLayer for Affine {
    type Inputs = Record! {
        x: Array2<f32>,
        weight: Array2<f32>,
        bias: Array1<f32>
    };
    type Placeholders = HNil;
    type Output = Array2<f32>;
    type Optimizer = AffineOptimizer;
    type Variables = HNil;

    fn forward(
        &self,
        HNil: Self::Placeholders,
        variables: Self::Variables,
        inputs: Self::Inputs,
    ) -> (Self::Output, Self::Optimizer) {
        record_dest!({
            x,
            weight,
            bias,
        } = inputs);
        let y = x.dot(&weight) + &bias;

        (y, AffineOptimizer { x, weight, bias })
    }
}
