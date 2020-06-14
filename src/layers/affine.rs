use super::NetworkVar;
use crate::layer::{LabelledLayerValues, LayerValue, UnconnectedLayer, UnconnectedOptimizer};
use frunk::labelled::{ByNameFieldPlucker, Field};
use frunk::{field, hlist, HNil, Hlist};
use frunk_labelled_proc_macro::label;
use ndarray::prelude::*;
use ndarray::Zip;
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;
use std::cell::RefCell;
use std::sync::Arc;

impl LayerValue for AffineParams {
    type Grad = AffineParamsValue;
}

#[derive(Debug)]
pub struct AffineParamsValue {
    pub weight: Array2<f32>,
    pub bias: Array1<f32>,
}

#[derive(Debug, Clone)]
pub struct AffineParams(Arc<RefCell<AffineParamsValue>>);

impl AffineParams {
    pub fn initialize(prev_n: usize, n: usize) -> AffineParams {
        let weight_init_std = 0.01;

        AffineParams(Arc::new(RefCell::new(AffineParamsValue {
            weight: Array::random((prev_n, n), Normal::new(0., 1.).unwrap()) * weight_init_std,
            bias: Array::zeros((n,)),
        })))
    }
}

impl NetworkVar for AffineParams {
    fn optimize(&self, grad: &Self::Grad, learning_rate: f32) {
        let mut value = self.0.borrow_mut();
        Zip::from(&mut value.weight)
            .and(&grad.weight)
            .apply(|x, y| *x -= learning_rate * y);
        Zip::from(&mut value.bias)
            .and(&grad.bias)
            .apply(|x, y| *x -= learning_rate * y);
    }
}

pub struct AffineOptimizer {
    pub params: AffineParams,
    pub x: Array2<f32>,
}

impl UnconnectedOptimizer for AffineOptimizer {
    type Inputs = Record! {
        params: AffineParams,
        x: Array2<f32>
    };
    type Output = Array2<f32>;

    fn optimize(
        self,
        dout: <Self::Output as LayerValue>::Grad,
        learning_rate: f32,
    ) -> <Self::Inputs as LabelledLayerValues>::Grads {
        let dx = {
            let params = self.params.0.borrow();
            dout.dot(&params.weight.t())
        };

        let dw = self.x.t().dot(&dout);
        let db = dout.sum_axis(Axis(0));

        record! {
            params: AffineParamsValue {
                weight: dw,
                bias: db,
            },
            x: dx
        }
    }
}

pub struct Affine {}

impl Affine {
    pub fn new() -> Self {
        Affine {}
    }
}

impl UnconnectedLayer for Affine {
    type Inputs = Record! {
        params: AffineParams,
        x: Array2<f32>
    };
    type Placeholders = HNil;
    type Output = Array2<f32>;
    type Optimizer = AffineOptimizer;

    fn forward(
        &self,
        HNil: Self::Placeholders,
        inputs: Self::Inputs,
    ) -> (Self::Output, Self::Optimizer) {
        record_dest!({
            params,
            x,
        } = inputs);
        let y = {
            let params_ref = params.0.borrow();
            x.dot(&params_ref.weight) + &params_ref.bias
        };

        (y, AffineOptimizer { params, x })
    }
}
