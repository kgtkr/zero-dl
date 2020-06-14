use super::NetworkVar;
use crate::arr_functions;
use crate::hlist_extra::ConcatAndSplit;
use crate::layer::UnconnectedLayer;
use crate::layer::{LabelledLayerValues, UnconnectedOptimizer};
use crate::layer::{Layer, LayerValue, Optimizer};
use frunk::labelled::{ByNameFieldPlucker, Field};
use frunk::HNil;
use frunk::{field, hlist, Hlist};
use ndarray::prelude::*;
use ndarray::Zip;
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;
use std::cell::RefCell;
use std::sync::Arc;

impl LayerValue for ConvolutionParams {
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

pub struct ConvolutionOptimizer {
    pub params: ConvolutionParams,
    pub x: Array4<f32>,
    pub stride: usize,
    pub pad: usize,
    pub col: Array2<f32>,
    pub col_W: Array2<f32>,
}

impl UnconnectedOptimizer for ConvolutionOptimizer {
    type Inputs = Record! {
        params: ConvolutionParams,
        x: Array4<f32>
    };
    type Output = Array4<f32>;

    fn optimize(
        self,
        dout: <Self::Output as LayerValue>::Grad,
        learning_rate: f32,
    ) -> <Self::Inputs as LabelledLayerValues>::Grads {
        let (dW, db, dx) = {
            let params = self.params.0.borrow();
            let (FN, C, FH, FW) = params.weight.dim();
            let dout_len = dout.len();
            let dout = dout
                .permuted_axes([0, 2, 3, 1])
                .to_shared()
                .reshape((dout_len / FN, FN));

            let db = dout.sum_axis(Axis(0));
            let dW = self.col.t().dot(&dout);
            let dW = dW.t().to_shared().reshape((FN, C, FH, FW)).to_owned();

            let dcol = dout.dot(&self.col_W.t());
            let dx =
                arr_functions::col2im(dcol.view(), self.x.dim(), FH, FW, self.stride, self.pad);

            (dW, db, dx)
        };

        record! {
            params: ConvolutionParamsValue {
                weight: dW,
                bias: db,
            },
            x: dx
        }
    }
}

pub struct Convolution {
    pub stride: usize,
    pub pad: usize,
}

impl Convolution {
    pub fn new(stride: usize, pad: usize) -> Self {
        Convolution { stride, pad }
    }
}

impl UnconnectedLayer for Convolution {
    type Inputs = Record! {
        params: ConvolutionParams,
        x: Array4<f32>
    };
    type Output = Array4<f32>;
    type Optimizer = ConvolutionOptimizer;
    type Placeholders = HNil;

    fn forward(
        &self,
        placeholders: Self::Placeholders,
        inputs: Self::Inputs,
    ) -> (Self::Output, Self::Optimizer) {
        record_dest!({
            params,
            x,
        } = inputs);

        let (out, col, col_W) = {
            let params = params.0.borrow();
            let (FN, C, FH, FW) = params.weight.dim();
            let (N, C, H, W) = x.dim();
            let out_h = 1 + (H + 2 * self.pad - FH) / self.stride;
            let out_w = 1 + (W + 2 * self.pad - FW) / self.stride;

            let col = arr_functions::im2col(x.view(), FH, FW, self.stride, self.pad).0;
            let col_W = params
                .weight
                .to_shared()
                .reshape((FN, C * FH * FW))
                .t()
                .to_owned();

            let out = col.dot(&col_W) + &params.bias;
            let out_len = out.len();
            let out = out
                .to_shared()
                .reshape((N, out_h, out_w, out_len / N / out_h / out_w))
                .permuted_axes([0, 3, 1, 2])
                .to_owned();
            (out, col, col_W)
        };

        (
            out,
            ConvolutionOptimizer {
                params,
                x,
                col,
                col_W,
                stride: self.stride,
                pad: self.pad,
            },
        )
    }
}
