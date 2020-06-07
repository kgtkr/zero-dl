use crate::arr_functions;
use crate::hlist_extra::Has;
use frunk::labelled::Field;
use frunk::labelled::{chars, LabelledGeneric, Transmogrifier};
use frunk::{field, HCons, HNil};
use ndarray::prelude::*;
use ndarray::Zip;
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::marker::PhantomData;
use std::ops::Add;
use std::sync::Arc;

#[derive(frunk::LabelledGeneric, Clone, Debug)]
pub struct PredictPlaceholders {}

#[derive(frunk::LabelledGeneric, Clone, Debug)]
pub struct LossPlaceholders {}

pub struct Network<L, LastL> {
    pub layers: L,
    pub last_layer: LastL,
}

impl<L, LastL> Network<L, LastL>
where
    L: Layer<PredictPlaceholders, Output = Array1<f32>>,
    LastL: Layer<LossPlaceholders, Output = f32>,
    LastL::Backward: LayerBackward<LossPlaceholders, OutputGrad = f32>,
{
    pub fn initialize(layers: L, last_layer: LastL) -> Network<L, LastL> {
        Network { layers, last_layer }
    }

    pub fn predict(&self, x: Array1<f32>) -> (Array1<f32>, L::Backward) {
        self.layers
            .forward(&LabelledGeneric::into(PredictPlaceholders {}))
    }

    pub fn loss(&mut self, x: Array1<f32>, t: Array1<f32>) -> (f32, LastL::Backward) {
        self.last_layer
            .forward(&LabelledGeneric::into(PredictPlaceholders {}))
    }

    pub fn learning(
        &mut self,
        params: &Vec<AffineParams>,
        x_train: Array2<f32>,
        t_train: Array2<f32>,
    ) {
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
                ba.backward(1., &LabelledGeneric::into(PredictPlaceholders {}));
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

pub trait LayerBackward<Placeholders: LabelledGeneric> {
    type Output;
    type OutputGrad;
    type Grads;

    fn backward(&self, dout: Self::OutputGrad, placeholders: &Placeholders::Repr) -> Self::Grads;
}

pub trait Layer<Placeholders: LabelledGeneric> {
    type Output;
    type Backward: LayerBackward<Placeholders>;

    fn forward(&self, placeholders: &Placeholders::Repr) -> (Self::Output, Self::Backward);
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

impl<V: NetworkVar, Placeholders: LabelledGeneric> LayerBackward<Placeholders>
    for VariableBackend<V>
{
    type Output = V::MutableRef;
    type OutputGrad = V;
    type Grads = ();

    fn backward(&self, dout: Self::OutputGrad, placeholders: &Placeholders::Repr) -> () {
        V::optimize(&self.value, &dout, 0.1);
    }
}

#[derive(Debug, Clone)]
pub struct Variable<V: NetworkVar> {
    pub value: V::MutableRef,
}

impl<V: NetworkVar, Placeholders: LabelledGeneric> Layer<Placeholders> for Variable<V>
where
    V: NetworkVar,
{
    type Output = V::MutableRef;

    type Backward = VariableBackend<V>;

    fn forward(&self, placeholders: &Placeholders::Repr) -> (Self::Output, Self::Backward) {
        (
            self.value.clone(),
            VariableBackend {
                value: self.value.clone(),
            },
        )
    }
}

#[derive(Debug, Clone)]
pub struct PlaceholderBackend<K, I, V> {
    pub phantom: PhantomData<(K, V, I)>,
}

impl<K, I, V, Placeholders: LabelledGeneric> LayerBackward<Placeholders>
    for PlaceholderBackend<K, I, V>
{
    type OutputGrad = V;
    type Output = V;
    type Grads = HNil;

    fn backward(&self, dout: Self::Output, placeholders: &Placeholders::Repr) -> Self::Grads {
        HNil
    }
}

#[derive(Debug, Clone)]
pub struct Placeholder<K, I, V> {
    pub phantom: PhantomData<(K, V, I)>,
}

impl<K, I, V, Placeholders: LabelledGeneric> Layer<Placeholders> for Placeholder<K, V, I>
where
    Placeholders: LabelledGeneric,
    Placeholders::Repr: Has<K, I, TargetValue = V>,
{
    type Output = V;

    type Backward = PlaceholderBackend<K, I, V>;

    fn forward(&self, placeholders: &Placeholders::Repr) -> (Self::Output, Self::Backward) {
        (
            placeholders.get(),
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
        Placeholders: LabelledGeneric,
        XL: LayerBackward<Placeholders, OutputGrad = Array1<f32>>,
        ParamsL: LayerBackward<Placeholders, OutputGrad = AffineParamsValue>,
    > LayerBackward<Placeholders> for AffineBackward<XL, ParamsL>
where
    XL::Grads: Add<<ParamsL as LayerBackward<Placeholders>>::Grads>,
{
    type OutputGrad = Array1<f32>;
    type Output = Array1<f32>;
    type Grads = <<XL as LayerBackward<Placeholders>>::Grads as Add<
        <ParamsL as LayerBackward<Placeholders>>::Grads,
    >>::Output;

    fn backward(&self, dout: Self::OutputGrad, placeholders: &Placeholders::Repr) -> Self::Grads {
        let dx = self.params.affine_backward(&dout);

        let dw = self
            .x
            .broadcast((1, self.x.len_of(Axis(0))))
            .unwrap()
            .t()
            .dot(&dout.broadcast((1, dout.len_of(Axis(0)))).unwrap());
        let db = dout;

        self.x_layer.backward(dx, placeholders)
            + self.params_layer.backward(
                AffineParamsValue {
                    weight: dw,
                    bias: db,
                },
                placeholders,
            )
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

impl<XL, ParamsL, Placeholders: LabelledGeneric> Layer<Placeholders> for Affine<XL, ParamsL>
where
    XL: Layer<Placeholders, Output = Array1<f32>>,
    ParamsL: Layer<Placeholders, Output = AffineParams>,
    XL::Backward: LayerBackward<Placeholders>,
    ParamsL::Backward: LayerBackward<Placeholders>,
    AffineBackward<XL::Backward, ParamsL::Backward>:
        LayerBackward<Placeholders, Output = Array1<f32>>,
{
    type Output = Array1<f32>;
    type Backward = AffineBackward<XL::Backward, ParamsL::Backward>;

    fn forward(&self, placeholders: &Placeholders::Repr) -> (Self::Output, Self::Backward) {
        let (x, x_layer) = self.x_layer.forward(placeholders);
        let (params, params_layer) = self.params_layer.forward(placeholders);

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

impl<XL, Placeholders: LabelledGeneric> LayerBackward<Placeholders> for ReluBackward<XL>
where
    XL: LayerBackward<Placeholders, OutputGrad = Array1<f32>>,
{
    type Output = Array1<f32>;
    type OutputGrad = XL::OutputGrad;
    type Grads = XL::Grads;

    fn backward(
        &self,
        mut dout: Self::OutputGrad,
        placeholders: &Placeholders::Repr,
    ) -> Self::Grads {
        Zip::from(&mut dout).and(&self.x).apply(|dout_x, &x| {
            if x <= 0. {
                *dout_x = 0.;
            }
        });

        self.x_layer.backward(dout, placeholders)
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

impl<XL, Placeholders: LabelledGeneric> Layer<Placeholders> for Relu<XL>
where
    XL: Layer<Placeholders, Output = Array1<f32>>,
    ReluBackward<XL::Backward>: LayerBackward<Placeholders, Output = Array1<f32>>,
{
    type Output = Array1<f32>;
    type Backward = ReluBackward<XL::Backward>;

    fn forward(&self, placeholders: &Placeholders::Repr) -> (Self::Output, Self::Backward) {
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

impl<XL, TL, Placeholders: LabelledGeneric> LayerBackward<Placeholders>
    for SoftmaxWithLossBackward<XL, TL>
where
    XL: LayerBackward<Placeholders, OutputGrad = Array1<f32>>,
    TL: LayerBackward<Placeholders, OutputGrad = Array1<f32>>,
    XL::Grads: Add<<TL as LayerBackward<Placeholders>>::Grads>,
{
    type OutputGrad = f32;
    type Output = f32;
    type Grads = <<XL as LayerBackward<Placeholders>>::Grads as Add<
        <TL as LayerBackward<Placeholders>>::Grads,
    >>::Output;

    fn backward(&self, dout: f32, placeholders: &Placeholders::Repr) -> Self::Grads {
        let d = &self.y - &self.t;

        self.x_layer.backward(d.clone(), placeholders)
            + self.t_layer.backward(d.clone(), placeholders)
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

impl<XL, TL, Placeholders: LabelledGeneric> Layer<Placeholders> for SoftmaxWithLoss<XL, TL>
where
    XL: Layer<Placeholders, Output = Array1<f32>>,
    TL: Layer<Placeholders, Output = Array1<f32>>,
    SoftmaxWithLossBackward<XL::Backward, TL::Backward>: LayerBackward<Placeholders, Output = f32>,
{
    type Output = f32;
    type Backward = SoftmaxWithLossBackward<XL::Backward, TL::Backward>;

    fn forward(&self, placeholders: &Placeholders::Repr) -> (Self::Output, Self::Backward) {
        let (x, x_layer) = self.x_layer.forward(placeholders);
        let (t, t_layer) = self.t_layer.forward(placeholders);

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
