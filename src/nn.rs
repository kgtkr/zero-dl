use crate::arr_functions;
use ndarray::prelude::*;
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
pub enum ActivationFunctionKind {
    Softmax,
    Step,
    Relu,
    Sigmoid,
    Identity,
}

impl ActivationFunctionKind {
    pub fn call(&self, xs: &Array1<f64>) -> Array1<f64> {
        match self {
            ActivationFunctionKind::Softmax => arr_functions::softmax_arr1(xs),
            ActivationFunctionKind::Step => arr_functions::step_arr1(xs),
            ActivationFunctionKind::Relu => arr_functions::relu_arr1(xs),
            ActivationFunctionKind::Sigmoid => arr_functions::sigmoid_arr1(xs),
            ActivationFunctionKind::Identity => arr_functions::identity_arr1(xs),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct NN {
    // 活性化関数
    h: ActivationFunctionKind,
    // 出力層で使う活性化関数
    a: ActivationFunctionKind,
    // 各層の重みとバイアス
    wbs: Vec<(Array2<f64>, Array1<f64>)>,
    input_size: usize,
    hidden_size: Vec<usize>,
    output_size: usize,
}

impl NN {
    pub fn new(
        h: ActivationFunctionKind,
        a: ActivationFunctionKind,
        input_size: usize,
        hidden_size: Vec<usize>,
        output_size: usize,
    ) -> NN {
        let weight_init_std = 0.01;

        let wbs = hidden_size
            .iter()
            .copied()
            .chain(std::iter::once(output_size))
            .zip(
                std::iter::once(input_size)
                    .chain(hidden_size.iter().copied())
                    .chain(std::iter::once(output_size)),
            )
            .map(|(n, prev_n)| {
                (
                    ArrayBase::random((prev_n, n), Normal::new(0., 1.).unwrap()) * weight_init_std,
                    ArrayBase::zeros((n,)),
                )
            })
            .collect();
        NN {
            h,
            a,
            wbs,
            input_size,
            hidden_size,
            output_size,
        }
    }

    pub fn serialize(&self) -> Vec<u8> {
        bincode::serialize(&self).unwrap()
    }

    pub fn deserialize(buf: &[u8]) -> Option<NN> {
        bincode::deserialize(buf).ok()
    }

    pub fn predict(&self, x: &Array1<f64>) -> Array1<f64> {
        let mut cur = x.clone();
        for (i, (w, b)) in self.wbs.iter().enumerate() {
            let a = cur.dot(w) + b;

            let v = (if i != self.wbs.len() - 1 {
                self.h
            } else {
                self.a
            })
            .call(&a);
            cur = v;
        }

        cur
    }

    pub fn loss(&self, x: &Array1<f64>, t: &Array1<f64>) -> f64 {
        let y = self.predict(x);
        arr_functions::cross_entroy_error(&y, t)
    }
}
