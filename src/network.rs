use crate::arr_functions;
use ndarray::prelude::*;
use ndarray::Zip;
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::sync::Arc;

pub fn mk_params(prev_n: usize, n: usize) -> Arc<RefCell<(Array2<f32>, Array1<f32>)>> {
    let weight_init_std = 0.01;

    Arc::new(RefCell::new((
        Array::random((prev_n, n), Normal::new(0., 1.).unwrap()) * weight_init_std,
        Array::zeros((n,)),
    )))
}

pub struct Network {
    pub layers: (Affine, (Relu, Affine)),
    pub last_layer: SoftmaxWithLoss,
}

impl Network {
    pub fn initialize(layers: (Affine, (Relu, Affine))) -> Network {
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
        impl LayerBackward<Input = Array1<f32>, Output = Array1<f32>>,
    ) {
        self.layers.forward(x)
    }

    pub fn loss(
        &mut self,
        x: Array1<f32>,
        t: Array1<f32>,
    ) -> (f32, impl LayerBackward<Input = Array1<f32>, Output = f32>) {
        let (y, ba1) = self.layers.forward(x);
        self.last_layer.t = t;
        let (loss, ba2) = self.last_layer.forward(y);
        (loss, (ba1, ba2))
    }

    pub fn gradient(&mut self, x: Array1<f32>, t: Array1<f32>) -> Vec<(Array2<f32>, Array1<f32>)> {
        let (_, ba) = self.loss(x, t);

        let mut grads = Vec::new();
        let dout = ba.backward(&mut grads, 1.);
        grads.reverse();

        grads
    }

    pub fn learning(
        &mut self,
        params: &Vec<Arc<RefCell<(Array2<f32>, Array1<f32>)>>>,
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
                let grad = self.gradient(x.to_owned(), t.to_owned());
                for (i, (gw, gb)) in grad.into_iter().enumerate() {
                    let mut mut_params = params[i].borrow_mut();
                    Zip::from(&mut mut_params.0)
                        .and(&gw)
                        .apply(|x, y| *x -= learning_rate * y);
                    Zip::from(&mut mut_params.1)
                        .and(&gb)
                        .apply(|x, y| *x -= learning_rate * y);
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

pub trait LayerBackward<'a> {
    type Input;
    type Output;

    fn backward(
        &self,
        grads: &mut Vec<(Array2<f32>, Array1<f32>)>,
        dout: Self::Output,
    ) -> Self::Input;
}

pub trait Layer<'a> {
    type Input;
    type Output;
    type Backward: LayerBackward<'a, Input = Self::Input, Output = Self::Output>;

    fn forward<'b: 'a>(&'b self, x: Self::Input) -> (Self::Output, Self::Backward);
}

impl<'a, A: LayerBackward<'a>, B: LayerBackward<'a, Input = A::Output>> LayerBackward<'a>
    for (A, B)
{
    type Input = A::Input;
    type Output = B::Output;

    fn backward(
        &self,
        grads: &mut Vec<(Array2<f32>, Array1<f32>)>,
        dout: Self::Output,
    ) -> Self::Input {
        let dout2 = self.1.backward(grads, dout);
        let dout3 = self.0.backward(grads, dout2);
        dout3
    }
}

impl<'a, A: Layer<'a>, B: Layer<'a, Input = A::Output>> Layer<'a> for (A, B) {
    type Input = A::Input;
    type Output = B::Output;
    type Backward = (A::Backward, B::Backward);

    fn forward<'b: 'a>(&'b self, x: Self::Input) -> (Self::Output, Self::Backward) {
        let (y, ba1) = self.0.forward(x);
        let (z, ba2) = self.1.forward(y);
        (z, (ba1, ba2))
    }
}

pub struct AffineBackward {
    pub params: Arc<RefCell<(Array2<f32>, Array1<f32>)>>,
    pub x: Array1<f32>,
}

impl<'a> LayerBackward<'a> for AffineBackward {
    type Input = Array1<f32>;
    type Output = Array1<f32>;

    fn backward(
        &self,
        grads: &mut Vec<(Array2<f32>, Array1<f32>)>,
        dout: Array1<f32>,
    ) -> Array1<f32> {
        let dx = dout.dot(&self.params.borrow().0.t());

        let dw = self
            .x
            .broadcast((1, self.x.len_of(Axis(0))))
            .unwrap()
            .t()
            .dot(&dout.broadcast((1, dout.len_of(Axis(0)))).unwrap());
        let db = dout;

        grads.push((dw, db));

        dx
    }
}

pub struct Affine {
    pub params: Arc<RefCell<(Array2<f32>, Array1<f32>)>>,
}

impl Affine {
    pub fn new(params: Arc<RefCell<(Array2<f32>, Array1<f32>)>>) -> Affine {
        Affine { params }
    }
}

impl<'a> Layer<'a> for Affine {
    type Input = Array1<f32>;
    type Output = Array1<f32>;
    type Backward = AffineBackward;

    fn forward<'b: 'a>(&'b self, x: Array1<f32>) -> (Self::Output, Self::Backward) {
        let y = x.dot(&self.params.borrow().0) + &self.params.borrow().0;
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

impl<'a> LayerBackward<'a> for ReluBackward {
    type Input = Array1<f32>;
    type Output = Array1<f32>;

    fn backward(
        &self,
        grads: &mut Vec<(Array2<f32>, Array1<f32>)>,
        mut dout: Array1<f32>,
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

impl<'a> Layer<'a> for Relu {
    type Input = Array1<f32>;
    type Output = Array1<f32>;
    type Backward = ReluBackward;

    fn forward<'b: 'a>(&'b self, x: Array1<f32>) -> (Self::Output, Self::Backward) {
        let y = x.mapv(|x| x.max(0.));
        (y, ReluBackward { x })
    }
}

pub struct SoftmaxWithLossBackward<'a> {
    pub y: Array1<f32>,
    pub t: &'a Array1<f32>,
}

impl<'a> LayerBackward<'a> for SoftmaxWithLossBackward<'a> {
    type Input = Array1<f32>;
    type Output = f32;

    fn backward(&self, grads: &mut Vec<(Array2<f32>, Array1<f32>)>, dout: f32) -> Array1<f32> {
        &self.y - self.t
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

impl<'a> Layer<'a> for SoftmaxWithLoss {
    type Input = Array1<f32>;
    type Output = f32;
    type Backward = SoftmaxWithLossBackward<'a>;

    // 呼び出す前にtセット(なんとかする)
    fn forward<'b: 'a>(&'b self, x: Array1<f32>) -> (Self::Output, Self::Backward) {
        let y = arr_functions::softmax_arr1(x.view());
        let loss = arr_functions::cross_entropy_error(y.view(), self.t.view());

        (loss, SoftmaxWithLossBackward { t: &self.t, y })
    }
}
