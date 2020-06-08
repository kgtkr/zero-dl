/*

use super::NetworkVar;
use crate::arr_functions;
use crate::hlist_extra::ConcatAndSplit;
use crate::layer::{Layer, LayerOutput, Optimizer};
use ndarray::prelude::*;
use ndarray::Zip;
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;
use std::cell::RefCell;
use std::sync::Arc;

impl LayerOutput for ConvolutionParams {
    type Grad = ConvolutionParamsValue;
}

#[derive(Debug)]
pub struct ConvolutionParamsValue {
    pub weight: Array4<f32>,
    pub bias: Array1<f32>,
}

#[derive(Debug, Clone)]
pub struct ConvolutionParams(Arc<RefCell<ConvolutionParamsValue>>);

impl ConvolutionParams {
    pub fn initialize(n: usize, c: usize, h: usize, w: usize) -> ConvolutionParams {
        let weight_init_std = 0.01;

        ConvolutionParams(Arc::new(RefCell::new(ConvolutionParamsValue {
            weight: Array::random((n, c, h, w), Normal::new(0., 1.).unwrap()) * weight_init_std,
            bias: Array::zeros((n,)),
        })))
    }
}

impl NetworkVar for ConvolutionParams {
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

pub struct ConvolutionOptimizer<XOpz, ParamsOpz> {
    pub x_optimizer: XOpz,
    pub params_optimizer: ParamsOpz,
    pub params: ConvolutionParams,
    pub x: Array2<f32>,
}

impl<XOpz: Optimizer<Output = Array2<f32>>, ParamsOpz: Optimizer<Output = ConvolutionParams>>
    Optimizer for ConvolutionOptimizer<XOpz, ParamsOpz>
{
    type Output = Array2<f32>;

    fn optimize(self, dout: <Self::Output as LayerOutput>::Grad, learning_rate: f32) {
        let dx = self.params.convolution_optimize(&dout);

        let dw = self.x.t().dot(&dout);
        let db = dout.sum_axis(Axis(0));

        self.x_optimizer.optimize(dx, learning_rate);
        self.params_optimizer.optimize(
            ConvolutionParamsValue {
                weight: dw,
                bias: db,
            },
            learning_rate,
        );
    }
}

pub struct Convolution<XL, ParamsL> {
    pub x_layer: XL,
    pub params_layer: ParamsL,
}

impl<XL, ParamsL> Convolution<XL, ParamsL>
where
    Self: Layer,
{
    pub fn new(x_layer: XL, params_layer: ParamsL) -> Self {
        Convolution {
            x_layer,
            params_layer,
        }
    }
}

impl<XL, ParamsL> Layer for Convolution<XL, ParamsL>
where
    XL: Layer<Output = Array2<f32>>,
    ParamsL: Layer<Output = ConvolutionParams>,
    XL::Optimizer: Optimizer,
    ParamsL::Optimizer: Optimizer,
    ConvolutionOptimizer<XL::Optimizer, ParamsL::Optimizer>: Optimizer,
    XL::Placeholders: ConcatAndSplit<ParamsL::Placeholders>,
{
    type Output = Array2<f32>;
    type Optimizer = ConvolutionOptimizer<XL::Optimizer, ParamsL::Optimizer>;
    type Placeholders = <XL::Placeholders as ConcatAndSplit<ParamsL::Placeholders>>::Output;

    fn forward(&self, placeholders: Self::Placeholders) -> (Self::Output, Self::Optimizer) {
        let (x_placeholders, params_placeholders) = ConcatAndSplit::split(placeholders);
        let (x, x_optimizer) = self.x_layer.forward(x_placeholders);
        let (params, params_optimizer) = self.params_layer.forward(params_placeholders);

        let y = params.convolution_forward(&x);
        (
            y,
            ConvolutionOptimizer {
                params,
                x,
                x_optimizer,
                params_optimizer,
            },
        )
    }
}

*/
