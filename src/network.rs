use crate::arr_functions;
use crate::hlist_extra::Has;
use ndarray::prelude::*;
use ndarray::Zip;
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::marker::PhantomData;
use std::sync::Arc;

pub struct Network<L> {
    pub layers: L,
    pub last_layer: SoftmaxWithLoss,
}

impl<L: Layer<(), (), Input = Array1<f32>, Output = Array1<f32>>> Network<L> {
    pub fn initialize(layers: L) -> Network<L> {
        Network {
            layers,
            last_layer: SoftmaxWithLoss::new(),
        }
    }

    pub fn predict(
        &self,
        x: Array1<f32>,
    ) -> (
        Array1<f32>,
        impl LayerBackward<(), (), Input = Array1<f32>, Output = Array1<f32>>,
    ) {
        self.layers.forward(x, &(), &())
    }

    pub fn loss(
        &mut self,
        x: Array1<f32>,
        t: Array1<f32>,
    ) -> (
        f32,
        impl LayerBackward<(), (), Input = Array1<f32>, Output = f32>,
    ) {
        let (y, ba1) = self.layers.forward(x, &(), &());
        self.last_layer.t = t;
        let (loss, ba2) = self.last_layer.forward(y, &(), &());
        (loss, (ba1, ba2))
    }

    pub fn gradient(&mut self, x: Array1<f32>, t: Array1<f32>) -> Vec<AffineParamsValue> {
        let (_, ba) = self.loss(x, t);

        let mut grads = Vec::new();
        let dout = ba.backward(&mut grads, 1., &(), &());
        grads.reverse();

        grads
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
                let grads = self.gradient(x.to_owned(), t.to_owned());
                for (i, grad) in grads.into_iter().enumerate() {
                    params[i].optimize(learning_rate, &grad);
                }
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

pub trait LayerBackward<Placeholders, Variables> {
    type Input;
    type Output;

    fn backward(
        &self,
        grads: &mut Vec<AffineParamsValue>,
        dout: Self::Output,
        placeholders: &Placeholders,
        variables: &Variables,
    ) -> Self::Input;
}

pub trait Layer<Placeholders, Variables> {
    type Input;
    type Output;
    type Backward: LayerBackward<
        Placeholders,
        Variables,
        Input = Self::Input,
        Output = Self::Output,
    >;

    fn forward(
        &self,
        x: Self::Input,
        placeholders: &Placeholders,
        variables: &Variables,
    ) -> (Self::Output, Self::Backward);
}

impl<
        Placeholders,
        Variables,
        A: LayerBackward<Placeholders, Variables>,
        B: LayerBackward<Placeholders, Variables, Input = A::Output>,
    > LayerBackward<Placeholders, Variables> for (A, B)
{
    type Input = A::Input;
    type Output = B::Output;

    fn backward(
        &self,
        grads: &mut Vec<AffineParamsValue>,
        dout: Self::Output,
        placeholders: &Placeholders,
        variables: &Variables,
    ) -> Self::Input {
        let dout2 = self.1.backward(grads, dout, placeholders, variables);
        let dout3 = self.0.backward(grads, dout2, placeholders, variables);
        dout3
    }
}

impl<
        'a,
        Placeholders,
        Variables,
        A: Layer<Placeholders, Variables>,
        B: Layer<Placeholders, Variables, Input = A::Output>,
    > Layer<Placeholders, Variables> for (A, B)
{
    type Input = A::Input;
    type Output = B::Output;
    type Backward = (A::Backward, B::Backward);

    fn forward(
        &self,
        x: Self::Input,
        placeholders: &Placeholders,
        variables: &Variables,
    ) -> (Self::Output, Self::Backward) {
        let (y, ba1) = self.0.forward(x, placeholders, variables);
        let (z, ba2) = self.1.forward(y, placeholders, variables);
        (z, (ba1, ba2))
    }
}

#[derive(Debug)]
pub struct AffineParamsValue {
    pub weight: Array2<f32>,
    pub bias: Array1<f32>,
}

pub trait NetworkVar {
    type Grad;
    fn optimize(&self, learning_rate: f32, grad: &Self::Grad);
}

#[derive(Debug, Clone)]
pub struct VariableBackend<K, I, V> {
    pub phantom: PhantomData<(K, V, I)>,
}

impl<K, I, V, Placeholders, Variables> LayerBackward<Placeholders, Variables>
    for VariableBackend<K, I, V>
{
    type Input = ();
    type Output = V;

    fn backward(
        &self,
        grads: &mut Vec<AffineParamsValue>,
        dout: Self::Output,
        placeholders: &Placeholders,
        variables: &Variables,
    ) -> Self::Input {
        ()
    }
}

#[derive(Debug, Clone)]
pub struct Variable<K, I, V> {
    pub phantom: PhantomData<(K, V, I)>,
}

impl<K, I, V, Placeholders, Variables: Has<K, I, TargetValue = V>> Layer<Placeholders, Variables>
    for Variable<K, V, I>
{
    type Input = ();
    type Output = V;

    type Backward = VariableBackend<K, I, V>;

    fn forward(
        &self,
        x: (),
        placeholders: &Placeholders,
        variables: &Variables,
    ) -> (Self::Output, Self::Backward) {
        (
            variables.get(),
            VariableBackend {
                phantom: PhantomData,
            },
        )
    }
}

#[derive(Debug, Clone)]
pub struct PlaceholderBackend<K, I, V> {
    pub phantom: PhantomData<(K, V, I)>,
}

impl<K, I, V, Placeholders, Variables> LayerBackward<Placeholders, Variables>
    for PlaceholderBackend<K, I, V>
{
    type Input = ();
    type Output = V;

    fn backward(
        &self,
        grads: &mut Vec<AffineParamsValue>,
        dout: Self::Output,
        placeholders: &Placeholders,
        variables: &Variables,
    ) -> Self::Input {
        ()
    }
}

#[derive(Debug, Clone)]
pub struct Placeholder<K, I, V> {
    pub phantom: PhantomData<(K, V, I)>,
}

impl<K, I, V, Placeholders: Has<K, I, TargetValue = V>, Variables> Layer<Placeholders, Variables>
    for Placeholder<K, V, I>
{
    type Input = ();
    type Output = V;

    type Backward = PlaceholderBackend<K, I, V>;

    fn forward(
        &self,
        x: (),
        placeholders: &Placeholders,
        variables: &Variables,
    ) -> (Self::Output, Self::Backward) {
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

impl NetworkVar for AffineParams {
    type Grad = AffineParamsValue;

    fn optimize(&self, learning_rate: f32, grad: &Self::Grad) {
        let mut value = self.0.borrow_mut();
        Zip::from(&mut value.weight)
            .and(&grad.weight)
            .apply(|x, y| *x -= learning_rate * y);
        Zip::from(&mut value.bias)
            .and(&grad.bias)
            .apply(|x, y| *x -= learning_rate * y);
    }
}

pub struct AffineBackward {
    pub params: AffineParams,
    pub x: Array1<f32>,
}

impl<Placeholders, Variables> LayerBackward<Placeholders, Variables> for AffineBackward {
    type Input = Array1<f32>;
    type Output = Array1<f32>;

    fn backward(
        &self,
        grads: &mut Vec<AffineParamsValue>,
        dout: Array1<f32>,
        placeholders: &Placeholders,
        variables: &Variables,
    ) -> Array1<f32> {
        let dx = self.params.affine_backward(&dout);

        let dw = self
            .x
            .broadcast((1, self.x.len_of(Axis(0))))
            .unwrap()
            .t()
            .dot(&dout.broadcast((1, dout.len_of(Axis(0)))).unwrap());
        let db = dout;

        grads.push(AffineParamsValue {
            weight: dw,
            bias: db,
        });

        dx
    }
}

pub struct Affine {
    pub params: AffineParams,
}

impl Affine {
    pub fn new(params: AffineParams) -> Affine {
        Affine { params }
    }
}

impl<'a, Placeholders, Variables> Layer<Placeholders, Variables> for Affine {
    type Input = Array1<f32>;
    type Output = Array1<f32>;
    type Backward = AffineBackward;

    fn forward(
        &self,
        x: Array1<f32>,
        placeholders: &Placeholders,
        variables: &Variables,
    ) -> (Self::Output, Self::Backward) {
        let y = self.params.affine_forward(&x);
        (
            y,
            AffineBackward {
                params: self.params.clone(),
                x,
            },
        )
    }
}

pub struct ReluBackward {
    pub x: Array1<f32>,
}

impl<'a, Placeholders, Variables> LayerBackward<Placeholders, Variables> for ReluBackward {
    type Input = Array1<f32>;
    type Output = Array1<f32>;

    fn backward(
        &self,
        grads: &mut Vec<AffineParamsValue>,
        mut dout: Array1<f32>,
        placeholders: &Placeholders,
        variables: &Variables,
    ) -> Array1<f32> {
        Zip::from(&mut dout).and(&self.x).apply(|dout_x, &x| {
            if x <= 0. {
                *dout_x = 0.;
            }
        });

        dout
    }
}

pub struct Relu {}

impl Relu {
    pub fn new() -> Relu {
        Relu {}
    }
}

impl<Placeholders, Variables> Layer<Placeholders, Variables> for Relu {
    type Input = Array1<f32>;
    type Output = Array1<f32>;
    type Backward = ReluBackward;

    fn forward(
        &self,
        x: Array1<f32>,
        placeholders: &Placeholders,
        variables: &Variables,
    ) -> (Self::Output, Self::Backward) {
        let y = x.mapv(|x| x.max(0.));
        (y, ReluBackward { x })
    }
}

pub struct SoftmaxWithLossBackward {
    pub y: Array1<f32>,
    pub t: Array1<f32>,
}

impl<Placeholders, Variables> LayerBackward<Placeholders, Variables> for SoftmaxWithLossBackward {
    type Input = Array1<f32>;
    type Output = f32;

    fn backward(
        &self,
        grads: &mut Vec<AffineParamsValue>,
        dout: f32,
        placeholders: &Placeholders,
        variables: &Variables,
    ) -> Array1<f32> {
        &self.y - &self.t
    }
}

pub struct SoftmaxWithLoss {
    pub t: Array1<f32>,
}

impl SoftmaxWithLoss {
    pub fn new() -> SoftmaxWithLoss {
        SoftmaxWithLoss { t: array![] }
    }
}

impl<Placeholders, Variables> Layer<Placeholders, Variables> for SoftmaxWithLoss {
    type Input = Array1<f32>;
    type Output = f32;
    type Backward = SoftmaxWithLossBackward;

    // 呼び出す前にtセット(なんとかする)
    fn forward(
        &self,
        x: Array1<f32>,
        placeholders: &Placeholders,
        variables: &Variables,
    ) -> (Self::Output, Self::Backward) {
        let y = arr_functions::softmax_arr1(x.view());
        let loss = arr_functions::cross_entropy_error(y.view(), self.t.view());

        (
            loss,
            SoftmaxWithLossBackward {
                t: self.t.clone(), /* TODO: cloneしない */
                y,
            },
        )
    }
}
