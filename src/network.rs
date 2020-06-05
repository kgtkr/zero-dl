use crate::arr_functions;
use ndarray::prelude::*;
use ndarray::Zip;
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;
use serde::{Deserialize, Serialize};

pub trait NetworkConfig {
    fn h_activation_function(&self, xs: &Array1<f64>) -> Array1<f64>;
    fn a_activation_function(&self, xs: &Array1<f64>) -> Array1<f64>;
    fn input_size(&self) -> usize;
    fn hidden_size(&self) -> &Vec<usize>;
    fn output_size(&self) -> usize;

    fn predict(&self, params: &NetworkParams, x: &Array1<f64>) -> Array1<f64> {
        let mut cur = x.clone();
        for (i, (w, b)) in params.0.iter().enumerate() {
            let a = cur.dot(w) + b;

            let v = if i != params.0.len() - 1 {
                self.h_activation_function(&a)
            } else {
                self.a_activation_function(&a)
            };
            cur = v;
        }

        cur
    }

    fn loss(&self, params: &NetworkParams, x: &Array1<f64>, t: &Array1<f64>) -> f64 {
        let y = self.predict(params, x);
        arr_functions::cross_entropy_error(&y, t)
    }

    fn numerical_gradient(&self, params: &mut NetworkParams, x: &Array1<f64>, t: &Array1<f64>) {
        for (w, b) in &mut params.0 {
            ({
                let mut grad = Array::zeros(w.raw_dim());

                Zip::from(&mut grad).and(w).apply(|grad_x, w| {
                    *grad_x = {
                        let h = 1e-4;

                        let tmp_val = *w;
                        *w = tmp_val + h;
                        let fxh1 = self.loss(params, x, t);
                        *w = tmp_val - h;
                        let fxh2 = self.loss(params, x, t);
                        *w = tmp_val;
                        (fxh1 - fxh2) / (2. * h)
                    };
                });
                grad
            });
        }
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

impl<HAF: Fn(&Array1<f64>) -> Array1<f64>, AAF: Fn(&Array1<f64>) -> Array1<f64>> NetworkConfig
    for ImplNetworkConfig<HAF, AAF>
{
    fn h_activation_function(&self, xs: &Array1<f64>) -> Array1<f64> {
        (self.h_activation_function)(xs)
    }

    fn a_activation_function(&self, xs: &Array1<f64>) -> Array1<f64> {
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
pub struct NetworkParams(pub Vec<(Array2<f64>, Array1<f64>)>);

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
        Network::new(config, NetworkParams::initialize(&config))
    }

    pub fn new(config: C, params: NetworkParams) -> Network<C> {
        Network { config, params }
    }
}

pub fn numerical_diff(f: impl Fn(f64) -> f64, x: f64) -> f64 {
    let h = 1e-4;
    (f(x + h) - f(x - h)) / (2. * h)
}

pub fn numerical_diff_i<T>(this: &mut T, f: &impl Fn(&T) -> f64, x: &mut f64) -> f64 {
    let h = 1e-4;

    let tmp_val = *x;

    *x = tmp_val + h;
    let fxh1 = f(this);

    *x = tmp_val - h;
    let fxh2 = f(this);

    *x = tmp_val;

    (fxh1 - fxh2) / (2. * h)
}

pub fn numerical_gradient<T, D: Dimension>(
    this: &mut T,
    f: impl Fn(&T) -> f64,
    xs: impl Fn(&mut T) -> &mut Array<f64, D>,
) -> Array<f64, D> {
    let mut grad = Array::zeros(xs(this).raw_dim());

    Zip::from(&mut grad).and(xs(this)).apply(|grad_x, x| {
        *grad_x = numerical_diff_i(this, &f, x);
    });

    grad
}
