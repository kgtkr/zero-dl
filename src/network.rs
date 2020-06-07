use crate::arr_functions;
use crate::hlist_extra::ConcatAndSplit;
use frunk::labelled::{Field, LabelledGeneric, Transmogrifier};
use frunk::{HCons, HNil};
use ndarray::prelude::*;
use ndarray::Zip;
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::marker::PhantomData;
use std::sync::Arc;

#[derive(frunk::LabelledGeneric, Clone, Debug)]
pub struct PredictPlaceholders {
    x: Array1<f32>,
}

pub type PredictPlaceholdersRepr = <PredictPlaceholders as LabelledGeneric>::Repr;

#[derive(frunk::LabelledGeneric, Clone, Debug)]
pub struct LossPlaceholders {
    x: Array1<f32>,
    t: Array1<f32>,
}

pub type LossPlaceholdersRepr = <LossPlaceholders as LabelledGeneric>::Repr;

pub struct Network<L, LastL, LTransmogrifyIndexIndices, LastLTransmogrifyIndexIndices> {
    pub layers: L,
    pub last_layer: LastL,
    pub phantom: PhantomData<(LTransmogrifyIndexIndices, LastLTransmogrifyIndexIndices)>,
}

impl<L, LastL, LTransmogrifyIndexIndices, LastLTransmogrifyIndexIndices>
    Network<L, LastL, LTransmogrifyIndexIndices, LastLTransmogrifyIndexIndices>
where
    L: Layer<Output = Array1<f32>, Placeholders = PredictPlaceholdersRepr>,
    LastL: Layer<Output = f32, Placeholders = LossPlaceholdersRepr>,
    L::Placeholders: Transmogrifier<PredictPlaceholdersRepr, LTransmogrifyIndexIndices>,
    LastL::Placeholders: Transmogrifier<LossPlaceholdersRepr, LastLTransmogrifyIndexIndices>,
    LastL::Backward: LayerBackward<OutputGrad = f32>,
{
    pub fn initialize(layers: L, last_layer: LastL) -> Self {
        Network {
            layers,
            last_layer,
            phantom: PhantomData,
        }
    }

    pub fn predict(&self, x: Array1<f32>) -> (Array1<f32>, L::Backward) {
        self.layers
            .forward(LabelledGeneric::into(PredictPlaceholders { x }))
    }

    pub fn loss(&mut self, x: Array1<f32>, t: Array1<f32>) -> (f32, LastL::Backward) {
        self.last_layer
            .forward(LabelledGeneric::into(LossPlaceholders { x, t }))
    }

    pub fn learning(&mut self, x_train: Array2<f32>, t_train: Array2<f32>) {
        let iters_num = 100;
        let batch_size = 100;
        let learning_rate = 0.1;
        let mut rng = rand::thread_rng();

        for index in 0..iters_num {
            for _ in 0..batch_size {
                let i = rng.gen_range(0, x_train.len_of(Axis(0)));
                let x = x_train.index_axis(Axis(0), i);
                let t = t_train.index_axis(Axis(0), i);
                let (_, ba) = self.loss(x.to_owned(), t.to_owned());
                ba.backward(1.);
            }

            let i = rng.gen_range(0, x_train.len_of(Axis(0)));
            let x = x_train.index_axis(Axis(0), i);
            let t = t_train.index_axis(Axis(0), i);

            println!(
                "i:{} loss:{}",
                index,
                self.loss(x.to_owned(), t.to_owned()).0
            );
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct NetworkParams(pub Vec<(Array2<f32>, Array1<f32>)>);

impl NetworkParams {
    pub fn serialize(&self) -> Vec<u8> {
        bincode::serialize(&self).unwrap()
    }

    pub fn deserialize(buf: &[u8]) -> Option<NetworkParams> {
        bincode::deserialize(buf).ok()
    }
}

pub fn numerical_diff(f: impl Fn(f32) -> f32, x: f32) -> f32 {
    let h = 1e-4;
    (f(x + h) - f(x - h)) / (2. * h)
}

pub trait LayerBackward {
    // TODO: これ型クラス作ってOutputの微分値みたいな関連型でまとめたい
    type Output;
    type OutputGrad;

    // TODO: 最適化みたいな名前に変える
    fn backward(&self, dout: Self::OutputGrad);
}

pub trait Layer {
    type Output;
    type Backward: LayerBackward;
    type Placeholders;

    fn forward(&self, placeholders: Self::Placeholders) -> (Self::Output, Self::Backward);
}

impl<T: Layer> Layer for &T {
    type Output = T::Output;
    type Backward = T::Backward;
    type Placeholders = T::Placeholders;

    fn forward(&self, placeholders: Self::Placeholders) -> (Self::Output, Self::Backward) {
        (*self).forward(placeholders)
    }
}

#[derive(Debug)]
pub struct AffineParamsValue {
    pub weight: Array2<f32>,
    pub bias: Array1<f32>,
}

pub trait NetworkVar {
    type MutableRef: Clone;
    fn optimize(target: &Self::MutableRef, grad: &Self, learning_rate: f32);
}

#[derive(Debug, Clone)]
pub struct VariableBackend<V: NetworkVar> {
    pub value: V::MutableRef,
}

impl<V: NetworkVar> LayerBackward for VariableBackend<V> {
    type Output = V::MutableRef;
    type OutputGrad = V;

    fn backward(&self, dout: Self::OutputGrad) {
        V::optimize(&self.value, &dout, 0.1);
    }
}

#[derive(Debug, Clone)]
pub struct Variable<V: NetworkVar> {
    pub value: V::MutableRef,
}

impl<V: NetworkVar> Variable<V> {
    pub fn new(value: V::MutableRef) -> Self {
        Variable { value }
    }
}

impl<V: NetworkVar> Layer for Variable<V>
where
    V: NetworkVar,
{
    type Output = V::MutableRef;

    type Backward = VariableBackend<V>;

    type Placeholders = HNil;

    fn forward(&self, placeholders: Self::Placeholders) -> (Self::Output, Self::Backward) {
        (
            self.value.clone(),
            VariableBackend {
                value: self.value.clone(),
            },
        )
    }
}

#[derive(Debug, Clone)]
pub struct PlaceholderBackend<K, V> {
    pub phantom: PhantomData<(K, V)>,
}

impl<K, V> LayerBackward for PlaceholderBackend<K, V> {
    type OutputGrad = V;
    type Output = V;

    fn backward(&self, dout: Self::Output) {}
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

impl<K, V> Layer for Placeholder<K, V> {
    type Output = V;

    type Backward = PlaceholderBackend<K, V>;

    type Placeholders = HCons<Field<K, V>, HNil>;

    fn forward(&self, placeholders: Self::Placeholders) -> (Self::Output, Self::Backward) {
        (
            placeholders.head.value,
            PlaceholderBackend {
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

    pub fn affine_backward(&self, dout: &Array1<f32>) -> Array1<f32> {
        dout.dot(&self.0.borrow().weight.t())
    }
}

impl NetworkVar for AffineParamsValue {
    type MutableRef = AffineParams;

    fn optimize(target: &AffineParams, grad: &Self, learning_rate: f32) {
        let mut value = target.0.borrow_mut();
        Zip::from(&mut value.weight)
            .and(&grad.weight)
            .apply(|x, y| *x -= learning_rate * y);
        Zip::from(&mut value.bias)
            .and(&grad.bias)
            .apply(|x, y| *x -= learning_rate * y);
    }
}

pub struct AffineBackward<XL, ParamsL> {
    pub x_layer: XL,
    pub params_layer: ParamsL,
    pub params: AffineParams,
    pub x: Array1<f32>,
}

impl<
        XL: LayerBackward<OutputGrad = Array1<f32>>,
        ParamsL: LayerBackward<OutputGrad = AffineParamsValue>,
    > LayerBackward for AffineBackward<XL, ParamsL>
{
    type OutputGrad = Array1<f32>;
    type Output = Array1<f32>;

    fn backward(&self, dout: Self::OutputGrad) {
        let dx = self.params.affine_backward(&dout);

        let dw = self
            .x
            .broadcast((1, self.x.len_of(Axis(0))))
            .unwrap()
            .t()
            .dot(&dout.broadcast((1, dout.len_of(Axis(0)))).unwrap());
        let db = dout;

        self.x_layer.backward(dx);
        self.params_layer.backward(AffineParamsValue {
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
    XL::Backward: LayerBackward,
    ParamsL::Backward: LayerBackward,
    AffineBackward<XL::Backward, ParamsL::Backward>: LayerBackward<Output = Array1<f32>>,
    XL::Placeholders: ConcatAndSplit<ParamsL::Placeholders>,
{
    type Output = Array1<f32>;
    type Backward = AffineBackward<XL::Backward, ParamsL::Backward>;
    type Placeholders = <XL::Placeholders as ConcatAndSplit<ParamsL::Placeholders>>::Output;

    fn forward(&self, placeholders: Self::Placeholders) -> (Self::Output, Self::Backward) {
        let (x_placeholders, params_placeholders) = ConcatAndSplit::split(placeholders);
        let (x, x_layer) = self.x_layer.forward(x_placeholders);
        let (params, params_layer) = self.params_layer.forward(params_placeholders);

        let y = params.affine_forward(&x);
        (
            y,
            AffineBackward {
                params,
                x,
                x_layer,
                params_layer,
            },
        )
    }
}

pub struct ReluBackward<XL> {
    pub x: Array1<f32>,
    pub x_layer: XL,
}

impl<XL> LayerBackward for ReluBackward<XL>
where
    XL: LayerBackward<OutputGrad = Array1<f32>>,
{
    type Output = Array1<f32>;
    type OutputGrad = XL::OutputGrad;

    fn backward(&self, mut dout: Self::OutputGrad) {
        Zip::from(&mut dout).and(&self.x).apply(|dout_x, &x| {
            if x <= 0. {
                *dout_x = 0.;
            }
        });

        self.x_layer.backward(dout);
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
    ReluBackward<XL::Backward>: LayerBackward<Output = Array1<f32>>,
{
    type Output = Array1<f32>;
    type Backward = ReluBackward<XL::Backward>;
    type Placeholders = XL::Placeholders;

    fn forward(&self, placeholders: Self::Placeholders) -> (Self::Output, Self::Backward) {
        let (x, x_layer) = self.x_layer.forward(placeholders);
        let y = x.mapv(|x| x.max(0.));
        (y, ReluBackward { x, x_layer })
    }
}

pub struct SoftmaxWithLossBackward<XL, TL> {
    pub y: Array1<f32>,
    pub t: Array1<f32>,
    pub x_layer: XL,
    pub t_layer: TL,
}

impl<XL, TL> LayerBackward for SoftmaxWithLossBackward<XL, TL>
where
    XL: LayerBackward<OutputGrad = Array1<f32>>,
    TL: LayerBackward<OutputGrad = Array1<f32>>,
{
    type OutputGrad = f32;
    type Output = f32;

    fn backward(&self, dout: f32) {
        let d = &self.y - &self.t;

        self.x_layer.backward(d);
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
    SoftmaxWithLossBackward<XL::Backward, TL::Backward>: LayerBackward<Output = f32>,
    XL::Placeholders: ConcatAndSplit<TL::Placeholders>,
{
    type Output = f32;
    type Backward = SoftmaxWithLossBackward<XL::Backward, TL::Backward>;
    type Placeholders = <XL::Placeholders as ConcatAndSplit<TL::Placeholders>>::Output;

    fn forward(&self, placeholders: Self::Placeholders) -> (Self::Output, Self::Backward) {
        let (x_placeholders, t_placeholders) = ConcatAndSplit::split(placeholders);

        let (x, x_layer) = self.x_layer.forward(x_placeholders);
        let (t, t_layer) = self.t_layer.forward(t_placeholders);

        let y = arr_functions::softmax_arr1(x.view());
        let loss = arr_functions::cross_entropy_error(y.view(), t.view());

        (
            loss,
            SoftmaxWithLossBackward {
                t,
                y,
                x_layer,
                t_layer,
            },
        )
    }
}
