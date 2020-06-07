use crate::arr_functions;
use crate::hlist_extra::ConcatAndSplit;
use frunk::labelled::{Field, LabelledGeneric, Transmogrifier};
use frunk::{HCons, HNil};
use ndarray::prelude::*;
use ndarray::Zip;
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::marker::PhantomData;
use std::sync::Arc;

pub trait LayerOutput {
    // 微分した時の型。変数を除いて基本的にSelfになる
    type Grad;
}

impl LayerOutput for f32 {
    type Grad = Self;
}

impl<Ix> LayerOutput for Array<f32, Ix> {
    type Grad = Self;
}

impl LayerOutput for AffineParams {
    type Grad = AffineParamsValue;
}

pub trait Optimizer {
    // TODO: これ型クラス作ってOutputの微分値みたいな関連型でまとめたい
    type Output: LayerOutput;

    fn optimize(self, dout: <Self::Output as LayerOutput>::Grad);
}

pub trait Layer {
    type Output;
    type Optimizer: Optimizer;
    type Placeholders;

    fn forward(&self, placeholders: Self::Placeholders) -> (Self::Output, Self::Optimizer);
}

impl<T: Layer> Layer for &T {
    type Output = T::Output;
    type Optimizer = T::Optimizer;
    type Placeholders = T::Placeholders;

    fn forward(&self, placeholders: Self::Placeholders) -> (Self::Output, Self::Optimizer) {
        (*self).forward(placeholders)
    }
}

#[derive(Debug)]
pub struct AffineParamsValue {
    pub weight: Array2<f32>,
    pub bias: Array1<f32>,
}

pub trait NetworkVar: Clone + LayerOutput {
    fn optimize(&self, grad: &Self::Grad, learning_rate: f32);
}

#[derive(Debug, Clone)]
pub struct VariableOptimizer<V: NetworkVar> {
    pub value: V,
}

impl<V: NetworkVar> Optimizer for VariableOptimizer<V> {
    type Output = V;

    fn optimize(self, dout: <Self::Output as LayerOutput>::Grad) {
        &self.value.optimize(&dout, 0.1);
    }
}

#[derive(Debug, Clone)]
pub struct Variable<V: NetworkVar> {
    pub value: V,
}

impl<V: NetworkVar> Variable<V> {
    pub fn new(value: V) -> Self {
        Variable { value }
    }
}

impl<V: NetworkVar> Layer for Variable<V> {
    type Output = V;

    type Optimizer = VariableOptimizer<V>;

    type Placeholders = HNil;

    fn forward(&self, placeholders: Self::Placeholders) -> (Self::Output, Self::Optimizer) {
        (
            self.value.clone(),
            VariableOptimizer {
                value: self.value.clone(),
            },
        )
    }
}

#[derive(Debug, Clone)]
pub struct PlaceholderOptimizer<K, V> {
    pub phantom: PhantomData<(K, V)>,
}

impl<K, V: LayerOutput> Optimizer for PlaceholderOptimizer<K, V> {
    type Output = V;

    fn optimize(self, dout: <Self::Output as LayerOutput>::Grad) {}
}

#[derive(Debug, Clone)]
pub struct Placeholder<K, V> {
    pub phantom: PhantomData<(K, V)>,
}

impl<K, V> Placeholder<K, V> {
    pub fn new() -> Self {
        Placeholder {
            phantom: PhantomData,
        }
    }
}

impl<K, V: LayerOutput> Layer for Placeholder<K, V> {
    type Output = V;

    type Optimizer = PlaceholderOptimizer<K, V>;

    type Placeholders = HCons<Field<K, V>, HNil>;

    fn forward(&self, placeholders: Self::Placeholders) -> (Self::Output, Self::Optimizer) {
        (
            placeholders.head.value,
            PlaceholderOptimizer {
                phantom: PhantomData,
            },
        )
    }
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

    pub fn affine_forward(&self, x: &Array1<f32>) -> Array1<f32> {
        x.dot(&self.0.borrow().weight) + &self.0.borrow().bias
    }

    pub fn affine_optimize(&self, dout: &Array1<f32>) -> Array1<f32> {
        dout.dot(&self.0.borrow().weight.t())
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
    pub x: Array1<f32>,
}

impl<XOpz: Optimizer<Output = Array1<f32>>, ParamsOpz: Optimizer<Output = AffineParams>> Optimizer
    for AffineOptimizer<XOpz, ParamsOpz>
{
    type Output = Array1<f32>;

    fn optimize(self, dout: <Self::Output as LayerOutput>::Grad) {
        let dx = self.params.affine_optimize(&dout);

        let dw = self
            .x
            .broadcast((1, self.x.len_of(Axis(0))))
            .unwrap()
            .t()
            .dot(&dout.broadcast((1, dout.len_of(Axis(0)))).unwrap());
        let db = dout;

        self.x_optimizer.optimize(dx);
        self.params_optimizer.optimize(AffineParamsValue {
            weight: dw,
            bias: db,
        });
    }
}

pub struct Affine<XL, ParamsL> {
    pub x_layer: XL,
    pub params_layer: ParamsL,
}

impl<XL, ParamsL> Affine<XL, ParamsL> {
    pub fn new(x_layer: XL, params_layer: ParamsL) -> Self {
        Affine {
            x_layer,
            params_layer,
        }
    }
}

impl<XL, ParamsL> Layer for Affine<XL, ParamsL>
where
    XL: Layer<Output = Array1<f32>>,
    ParamsL: Layer<Output = AffineParams>,
    XL::Optimizer: Optimizer,
    ParamsL::Optimizer: Optimizer,
    AffineOptimizer<XL::Optimizer, ParamsL::Optimizer>: Optimizer<Output = Array1<f32>>,
    XL::Placeholders: ConcatAndSplit<ParamsL::Placeholders>,
{
    type Output = Array1<f32>;
    type Optimizer = AffineOptimizer<XL::Optimizer, ParamsL::Optimizer>;
    type Placeholders = <XL::Placeholders as ConcatAndSplit<ParamsL::Placeholders>>::Output;

    fn forward(&self, placeholders: Self::Placeholders) -> (Self::Output, Self::Optimizer) {
        let (x_placeholders, params_placeholders) = ConcatAndSplit::split(placeholders);
        let (x, x_optimizer) = self.x_layer.forward(x_placeholders);
        let (params, params_optimizer) = self.params_layer.forward(params_placeholders);

        let y = params.affine_forward(&x);
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

pub struct ReluOptimizer<XOpz> {
    pub x: Array1<f32>,
    pub x_optimizer: XOpz,
}

impl<XOpz> Optimizer for ReluOptimizer<XOpz>
where
    XOpz: Optimizer<Output = Array1<f32>>,
{
    type Output = Array1<f32>;

    fn optimize(self, mut dout: <Self::Output as LayerOutput>::Grad) {
        Zip::from(&mut dout).and(&self.x).apply(|dout_x, &x| {
            if x <= 0. {
                *dout_x = 0.;
            }
        });

        self.x_optimizer.optimize(dout);
    }
}

pub struct Relu<XL> {
    pub x_layer: XL,
}

impl<XL> Relu<XL> {
    pub fn new(x_layer: XL) -> Relu<XL> {
        Relu { x_layer }
    }
}

impl<XL> Layer for Relu<XL>
where
    XL: Layer<Output = Array1<f32>>,
    ReluOptimizer<XL::Optimizer>: Optimizer<Output = Array1<f32>>,
{
    type Output = Array1<f32>;
    type Optimizer = ReluOptimizer<XL::Optimizer>;
    type Placeholders = XL::Placeholders;

    fn forward(&self, placeholders: Self::Placeholders) -> (Self::Output, Self::Optimizer) {
        let (x, x_optimizer) = self.x_layer.forward(placeholders);
        let y = x.mapv(|x| x.max(0.));
        (y, ReluOptimizer { x, x_optimizer })
    }
}

pub struct SoftmaxWithLossOptimizer<XOpz, TOpz> {
    pub y: Array1<f32>,
    pub t: Array1<f32>,
    pub x_optimizer: XOpz,
    pub t_optimizer: TOpz,
}

impl<XOpz, TOpz> Optimizer for SoftmaxWithLossOptimizer<XOpz, TOpz>
where
    XOpz: Optimizer<Output = Array1<f32>>,
    TOpz: Optimizer<Output = Array1<f32>>,
{
    type Output = f32;

    fn optimize(self, dout: f32) {
        let d = &self.y - &self.t;

        self.x_optimizer.optimize(d);
        // TODO: 本当はtの微分も考えるべきかも？
    }
}

pub struct SoftmaxWithLoss<XL, TL> {
    pub x_layer: XL,
    pub t_layer: TL,
}

impl<XL, TL> SoftmaxWithLoss<XL, TL> {
    pub fn new(x_layer: XL, t_layer: TL) -> Self {
        SoftmaxWithLoss { x_layer, t_layer }
    }
}

impl<XL, TL> Layer for SoftmaxWithLoss<XL, TL>
where
    XL: Layer<Output = Array1<f32>>,
    TL: Layer<Output = Array1<f32>>,
    SoftmaxWithLossOptimizer<XL::Optimizer, TL::Optimizer>: Optimizer<Output = f32>,
    XL::Placeholders: ConcatAndSplit<TL::Placeholders>,
{
    type Output = f32;
    type Optimizer = SoftmaxWithLossOptimizer<XL::Optimizer, TL::Optimizer>;
    type Placeholders = <XL::Placeholders as ConcatAndSplit<TL::Placeholders>>::Output;

    fn forward(&self, placeholders: Self::Placeholders) -> (Self::Output, Self::Optimizer) {
        let (x_placeholders, t_placeholders) = ConcatAndSplit::split(placeholders);

        let (x, x_optimizer) = self.x_layer.forward(x_placeholders);
        let (t, t_optimizer) = self.t_layer.forward(t_placeholders);

        let y = arr_functions::softmax_arr1(x.view());
        let loss = arr_functions::cross_entropy_error(y.view(), t.view());

        (
            loss,
            SoftmaxWithLossOptimizer {
                t,
                y,
                x_optimizer,
                t_optimizer,
            },
        )
    }
}
