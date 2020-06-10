use super::NetworkVar;
use crate::hlist_extra::ConcatAndSplit;
use crate::layer::{Layer, LayerValue, Optimizer};
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

pub struct AffineOptimizer<XOpz, ParamsOpz> {
    pub x_optimizer: XOpz,
    pub params_optimizer: ParamsOpz,
    pub params: AffineParams,
    pub x: Array2<f32>,
}

impl<XOpz: Optimizer<Output = Array2<f32>>, ParamsOpz: Optimizer<Output = AffineParams>> Optimizer
    for AffineOptimizer<XOpz, ParamsOpz>
{
    type Output = Array2<f32>;

    fn optimize(self, dout: <Self::Output as LayerValue>::Grad, learning_rate: f32) {
        let dx = {
            let params = self.params.0.borrow();
            dout.dot(&params.weight.t())
        };

        let dw = self.x.t().dot(&dout);
        let db = dout.sum_axis(Axis(0));

        self.x_optimizer.optimize(dx, learning_rate);
        self.params_optimizer.optimize(
            AffineParamsValue {
                weight: dw,
                bias: db,
            },
            learning_rate,
        );
    }
}

pub struct Affine<XL, ParamsL> {
    pub x_layer: XL,
    pub params_layer: ParamsL,
}

impl<XL, ParamsL> Affine<XL, ParamsL>
where
    Self: Layer,
{
    pub fn new(x_layer: XL, params_layer: ParamsL) -> Self {
        Affine {
            x_layer,
            params_layer,
        }
    }
}

impl<XL, ParamsL> Layer for Affine<XL, ParamsL>
where
    XL: Layer<Output = Array2<f32>>,
    ParamsL: Layer<Output = AffineParams>,
    XL::Optimizer: Optimizer,
    ParamsL::Optimizer: Optimizer,
    AffineOptimizer<XL::Optimizer, ParamsL::Optimizer>: Optimizer<Output = Array2<f32>>,
    XL::Placeholders: ConcatAndSplit<ParamsL::Placeholders>,
{
    type Output = Array2<f32>;
    type Optimizer = AffineOptimizer<XL::Optimizer, ParamsL::Optimizer>;
    type Placeholders = <XL::Placeholders as ConcatAndSplit<ParamsL::Placeholders>>::Out;

    fn forward(&self, placeholders: Self::Placeholders) -> (Self::Output, Self::Optimizer) {
        let (x_placeholders, params_placeholders) = ConcatAndSplit::split(placeholders);
        let (x, x_optimizer) = self.x_layer.forward(x_placeholders);

        let (params, params_optimizer) = self.params_layer.forward(params_placeholders);

        let y = {
            let params = params.0.borrow();
            x.dot(&params.weight) + &params.bias
        };

        (
            y,
            AffineOptimizer {
                params,
                x,
                x_optimizer,
                params_optimizer,
            },
        )
    }
}
