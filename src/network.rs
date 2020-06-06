use crate::arr_functions;
use ndarray::prelude::*;
use ndarray::Zip;
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;
use rand::prelude::*;
use serde::{Deserialize, Serialize};

pub trait NetworkConfig {
    fn h_activation_function(&self, xs: ArrayView1<f32>) -> Array1<f32>;
    fn a_activation_function(&self, xs: ArrayView1<f32>) -> Array1<f32>;
    fn input_size(&self) -> usize;
    fn hidden_size(&self) -> &Vec<usize>;
    fn output_size(&self) -> usize;

    fn predict(&self, params: &NetworkParams, x: ArrayView1<f32>) -> Array1<f32> {
        let mut cur = x.to_owned();
        for (i, (w, b)) in params.0.iter().enumerate() {
            let a = cur.dot(w) + b;

            let v = if i != params.0.len() - 1 {
                self.h_activation_function(a.view())
            } else {
                self.a_activation_function(a.view())
            };
            cur = v;
        }

        cur
    }

    fn loss(&self, params: &NetworkParams, x: ArrayView1<f32>, t: ArrayView1<f32>) -> f32 {
        let y = self.predict(params, x);
        arr_functions::cross_entropy_error(y.view(), t)
    }

    fn numerical_gradient(
        &self,
        params: &mut NetworkParams,
        x: ArrayView1<f32>,
        t: ArrayView1<f32>,
    ) -> Vec<(Array2<f32>, Array1<f32>)> {
        (0..params.0.len())
            .map(|i| {
                // TODO: 抽象化したいが所有権やばやば
                // これじゃrayonも使えん
                (
                    {
                        let mut grad = Array::zeros(params.0[i].0.raw_dim());

                        for idx in ndarray::indices(params.0[i].0.raw_dim()) {
                            grad[idx] = {
                                let h = 1e-4;

                                let tmp_val = params.0[i].0[idx];
                                params.0[i].0[idx] = tmp_val + h;
                                let fxh1 = self.loss(params, x, t);
                                params.0[i].0[idx] = tmp_val - h;
                                let fxh2 = self.loss(params, x, t);
                                params.0[i].0[idx] = tmp_val;
                                (fxh1 - fxh2) / (2. * h)
                            };
                        }

                        grad
                    },
                    {
                        let mut grad = Array::zeros(params.0[i].1.raw_dim());

                        for idx in ndarray::indices(params.0[i].1.raw_dim()) {
                            grad[idx] = {
                                let h = 1e-4;

                                let tmp_val = params.0[i].1[idx];
                                params.0[i].1[idx] = tmp_val + h;
                                let fxh1 = self.loss(params, x, t);
                                params.0[i].1[idx] = tmp_val - h;
                                let fxh2 = self.loss(params, x, t);
                                params.0[i].1[idx] = tmp_val;
                                (fxh1 - fxh2) / (2. * h)
                            };
                        }

                        grad
                    },
                )
            })
            .collect()
    }
}

#[derive(Debug, Clone)]
pub struct ImplNetworkConfig<HAF, AAF> {
    pub h_activation_function: HAF,
    pub a_activation_function: AAF,
    pub input_size: usize,
    pub hidden_size: Vec<usize>,
    pub output_size: usize,
}

impl<HAF: Fn(ArrayView1<f32>) -> Array1<f32>, AAF: Fn(ArrayView1<f32>) -> Array1<f32>> NetworkConfig
    for ImplNetworkConfig<HAF, AAF>
{
    fn h_activation_function(&self, xs: ArrayView1<f32>) -> Array1<f32> {
        (self.h_activation_function)(xs)
    }

    fn a_activation_function(&self, xs: ArrayView1<f32>) -> Array1<f32> {
        (self.a_activation_function)(xs)
    }

    fn input_size(&self) -> usize {
        self.input_size
    }

    fn hidden_size(&self) -> &Vec<usize> {
        &self.hidden_size
    }

    fn output_size(&self) -> usize {
        self.output_size
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct NetworkParams(pub Vec<(Array2<f32>, Array1<f32>)>);

impl NetworkParams {
    pub fn initialize(config: &impl NetworkConfig) -> NetworkParams {
        let weight_init_std = 0.01;

        let ns_iter = || {
            std::iter::once(config.input_size())
                .chain(config.hidden_size().iter().copied())
                .chain(std::iter::once(config.output_size()))
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

        NetworkParams(params)
    }

    pub fn serialize(&self) -> Vec<u8> {
        bincode::serialize(&self).unwrap()
    }

    pub fn deserialize(buf: &[u8]) -> Option<NetworkParams> {
        bincode::deserialize(buf).ok()
    }
}

#[derive(Debug, Clone)]
pub struct Network<C> {
    config: C,
    params: NetworkParams,
}

impl<C: NetworkConfig> Network<C> {
    pub fn initialize(config: C) -> Network<C> {
        let params = NetworkParams::initialize(&config);
        Network::new(config, params)
    }

    pub fn new(config: C, params: NetworkParams) -> Network<C> {
        Network { config, params }
    }

    pub fn predict(&self, x: ArrayView1<f32>) -> Array1<f32> {
        self.config.predict(&self.params, x)
    }

    pub fn loss(&self, x: ArrayView1<f32>, t: ArrayView1<f32>) -> f32 {
        self.config.loss(&self.params, x, t)
    }

    pub fn numerical_gradient(
        &mut self,
        x: ArrayView1<f32>,
        t: ArrayView1<f32>,
    ) -> Vec<(Array2<f32>, Array1<f32>)> {
        self.config.numerical_gradient(&mut self.params, x, t)
    }

    pub fn learning(&mut self, x_train: ArrayView2<f32>, t_train: ArrayView2<f32>) {
        let iters_num = 10000;
        let batch_size = 100;
        let learning_rate = 0.1;
        let mut rng = rand::thread_rng();

        for index in 0..iters_num * batch_size {
            let i = rng.gen_range(0, x_train.len_of(Axis(0)));
            let x = x_train.index_axis(Axis(0), i);
            let t = t_train.index_axis(Axis(0), i);

            let grad = self.numerical_gradient(x, t);

            for (i, (gw, gb)) in grad.into_iter().enumerate() {
                Zip::from(&mut self.params.0[i].0)
                    .and(&gw)
                    .apply(|x, y| *x -= learning_rate * y);

                Zip::from(&mut self.params.0[i].1)
                    .and(&gb)
                    .apply(|x, y| *x -= learning_rate * y);
            }

            println!("i:{} loss:{}", index, self.loss(x, t));
        }
    }
}

pub fn numerical_diff(f: impl Fn(f32) -> f32, x: f32) -> f32 {
    let h = 1e-4;
    (f(x + h) - f(x - h)) / (2. * h)
}

pub struct Affine {
    pub w: Array2<f32>,
    pub b: Array1<f32>,
    pub x: Array1<f32>,
    pub dw: Array2<f32>,
    pub db: Array1<f32>,
}

impl Affine {
    pub fn new(w: Array2<f32>, b: Array1<f32>) -> Affine {
        Affine {
            x: Array::zeros((w.len_of(Axis(0)),)),
            dw: Array::zeros(w.raw_dim()),
            db: Array::zeros(b.raw_dim()),
            w,
            b,
        }
    }

    pub fn forward(&mut self, x: Array1<f32>) -> Array1<f32> {
        self.x = x;
        self.x.dot(&self.w) + &self.b
    }

    pub fn backward(&mut self, dout: Array1<f32>) -> Array1<f32> {
        let dx = dout.dot(&self.w.t());
        self.dw = self
            .x
            .broadcast((1, self.x.len_of(Axis(0))))
            .unwrap()
            .t()
            .dot(&dout.broadcast((1, self.x.len_of(Axis(0)))).unwrap());
        self.db = dout;
        dx
    }
}

pub struct Relu {
    pub x: Array1<f32>,
}

impl Relu {
    pub fn new() -> Relu {
        Relu { x: array![] }
    }

    pub fn forward(&mut self, x: Array1<f32>) -> Array1<f32> {
        self.x = x;
        self.x.mapv(|x| x.max(0.))
    }

    pub fn backward(&mut self, mut dout: Array1<f32>) -> Array1<f32> {
        Zip::from(&mut dout).and(&self.x).apply(|dout_x, &x| {
            if x <= 0. {
                *dout_x = 0.;
            }
        });

        dout
    }
}

pub struct SoftmaxWithLoss {
    pub loss: f32,
    pub y: Array1<f32>,
    pub t: Array1<f32>,
}

impl SoftmaxWithLoss {
    pub fn new() -> SoftmaxWithLoss {
        SoftmaxWithLoss {
            loss: 0.,
            y: array![],
            t: array![],
        }
    }

    pub fn forward(&mut self, x: Array1<f32>, t: Array1<f32>) -> f32 {
        self.t = t;
        self.y = arr_functions::softmax_arr1(x.view());
        self.loss = arr_functions::cross_entropy_error(self.y.view(), self.t.view());

        self.loss
    }

    pub fn backward(&mut self) -> Array1<f32> {
        &self.y - &self.t
    }
}
