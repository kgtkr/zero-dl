use crate::arr_functions;
use ndarray::prelude::*;
use ndarray::Zip;
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

pub struct Network<L> {
    pub layers: L,
    pub params: NetworkParams,
    pub last_layer: SoftmaxWithLoss<NetworkParams>,
}

impl<L: Layer<Input = Array1<f32>, Output = Array1<f32>, State = NetworkParams>> Network<L> {
    pub fn initialize(
        layers: L,
        input_size: usize,
        hidden_size: Vec<usize>,
        output_size: usize,
    ) -> Network<L> {
        let weight_init_std = 0.01;

        let ns_iter = || {
            std::iter::once(input_size)
                .chain(hidden_size.iter().copied())
                .chain(std::iter::once(output_size))
        };

        let params = ns_iter()
            .skip(1)
            .zip(ns_iter())
            .map(|(n, prev_n)| {
                (
                    Array::random((prev_n, n), Normal::new(0., 1.).unwrap()) * weight_init_std,
                    Array::zeros((n,)),
                )
            })
            .collect();

        Network {
            layers,
            params: NetworkParams(params),
            last_layer: SoftmaxWithLoss::new(),
        }
    }

    pub fn predict(&mut self, x: Array1<f32>) -> Array1<f32> {
        self.layers.forward(&mut self.params, x)
    }

    pub fn loss(&mut self, x: Array1<f32>, t: Array1<f32>) -> f32 {
        let y = self.predict(x);
        self.last_layer.t = t;
        self.last_layer.forward(&mut self.params, y)
    }

    pub fn gradient(&mut self, x: Array1<f32>, t: Array1<f32>) -> Vec<(Array2<f32>, Array1<f32>)> {
        self.loss(x, t);
        let dout = self.last_layer.backward(&mut self.params, 1.);
        let dout = self.layers.backward(&mut self.params, dout);

        let mut grads = Vec::new();
        self.layers.collect_grads(&mut grads);
        grads
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
                let grad = self.gradient(x.to_owned(), t.to_owned());
                for (i, (gw, gb)) in grad.into_iter().enumerate() {
                    Zip::from(&mut self.params.0[i].0)
                        .and(&gw)
                        .apply(|x, y| *x -= learning_rate * y);
                    Zip::from(&mut self.params.0[i].1)
                        .and(&gb)
                        .apply(|x, y| *x -= learning_rate * y);
                }
            }

            let i = rng.gen_range(0, x_train.len_of(Axis(0)));
            let x = x_train.index_axis(Axis(0), i);
            let t = t_train.index_axis(Axis(0), i);

            println!("i:{} loss:{}", index, self.loss(x.to_owned(), t.to_owned()));
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

pub trait Layer {
    type Input;
    type Output;
    type State;

    fn forward(&mut self, state: &mut Self::State, x: Self::Input) -> Self::Output;

    fn backward(&mut self, state: &mut Self::State, dout: Self::Output) -> Self::Input;

    fn collect_grads(&mut self, grads: &mut Vec<(Array2<f32>, Array1<f32>)>) {}
}

impl<A: Layer, B: Layer<Input = A::Output, State = A::State>> Layer for (A, B) {
    type Input = A::Input;
    type Output = B::Output;
    type State = A::State;

    fn forward(&mut self, state: &mut Self::State, x: Self::Input) -> Self::Output {
        let y = self.0.forward(state, x);
        self.1.forward(state, y)
    }

    fn backward(&mut self, state: &mut Self::State, dout: Self::Output) -> Self::Input {
        let dout2 = self.1.backward(state, dout);
        self.0.backward(state, dout2)
    }

    fn collect_grads(&mut self, grads: &mut Vec<(Array2<f32>, Array1<f32>)>) {
        self.0.collect_grads(grads);
        self.1.collect_grads(grads);
    }
}

pub struct Affine {
    pub params_i: usize,
    pub x: Array1<f32>,
    pub dw: Array2<f32>,
    pub db: Array1<f32>,
}

impl Affine {
    pub fn new(params_i: usize) -> Affine {
        Affine {
            params_i,
            x: Array::zeros((0,)),
            dw: Array::zeros((0, 0)),
            db: Array::zeros((0,)),
        }
    }
}

impl Layer for Affine {
    type Input = Array1<f32>;
    type Output = Array1<f32>;
    type State = NetworkParams;

    fn forward(&mut self, state: &mut NetworkParams, x: Array1<f32>) -> Array1<f32> {
        self.x = x;
        self.x.dot(&state.0[self.params_i].0) + &state.0[self.params_i].1
    }

    fn backward(&mut self, state: &mut NetworkParams, dout: Array1<f32>) -> Array1<f32> {
        let dx = dout.dot(&state.0[self.params_i].0.t());
        self.dw = self
            .x
            .broadcast((1, self.x.len_of(Axis(0))))
            .unwrap()
            .t()
            .dot(&dout.broadcast((1, dout.len_of(Axis(0)))).unwrap());
        self.db = dout;
        dx
    }

    fn collect_grads(&mut self, grads: &mut Vec<(Array2<f32>, Array1<f32>)>) {
        grads.push((self.dw.clone(), self.db.clone()));
    }
}

pub struct Relu<S> {
    pub x: Array1<f32>,
    pub state: PhantomData<S>,
}

impl<S> Relu<S> {
    pub fn new() -> Relu<S> {
        Relu {
            x: array![],
            state: PhantomData,
        }
    }
}

impl<S> Layer for Relu<S> {
    type Input = Array1<f32>;
    type Output = Array1<f32>;
    type State = S;

    fn forward(&mut self, state: &mut S, x: Array1<f32>) -> Array1<f32> {
        self.x = x;
        self.x.mapv(|x| x.max(0.))
    }

    fn backward(&mut self, state: &mut S, mut dout: Array1<f32>) -> Array1<f32> {
        Zip::from(&mut dout).and(&self.x).apply(|dout_x, &x| {
            if x <= 0. {
                *dout_x = 0.;
            }
        });

        dout
    }
}

pub struct SoftmaxWithLoss<S> {
    pub loss: f32,
    pub y: Array1<f32>,
    pub t: Array1<f32>,
    pub state: PhantomData<S>,
}

impl<S> SoftmaxWithLoss<S> {
    pub fn new() -> SoftmaxWithLoss<S> {
        SoftmaxWithLoss {
            loss: 0.,
            y: array![],
            t: array![],
            state: PhantomData,
        }
    }
}

impl<S> Layer for SoftmaxWithLoss<S> {
    type Input = Array1<f32>;
    type Output = f32;
    type State = S;

    // 呼び出す前にtセット(なんとかする)
    fn forward(&mut self, state: &mut S, x: Array1<f32>) -> f32 {
        self.y = arr_functions::softmax_arr1(x.view());
        self.loss = arr_functions::cross_entropy_error(self.y.view(), self.t.view());

        self.loss
    }

    fn backward(&mut self, state: &mut S, dout: f32) -> Array1<f32> {
        &self.y - &self.t
    }
}
