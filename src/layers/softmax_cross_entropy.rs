use crate::arr_functions;
use crate::hlist_extra::ConcatAndSplit;
use crate::layer::{LabelledLayerValues, Layer, Optimizer, UnconnectedLayer, UnconnectedOptimizer};
use frunk::HNil;
use ndarray::prelude::*;

pub struct SoftmaxCrossEntropyOptimizer {
    pub y: Array2<f32>,
    pub t: Array2<f32>,
}

impl UnconnectedOptimizer for SoftmaxCrossEntropyOptimizer {
    type Inputs = Record! {
        x: Array2<f32>,
        t: Array2<f32>
    };
    type Output = f32;

    fn optimize(
        self,
        dout: f32,
        learning_rate: f32,
    ) -> <Self::Inputs as LabelledLayerValues>::Grads {
        let batch_size = self.t.len_of(Axis(0));
        let dx = (&self.y - &self.t) / batch_size as f32;

        /**
         * TODO: doutが1の時しか正常に動かない
         * またdtは正常な値でない
         */
        record! {
            x: dx,
            t: Array2::zeros((0,0))
        }
    }
}

pub struct SoftmaxCrossEntropy {}

impl SoftmaxCrossEntropy {
    pub fn new() -> Self {
        SoftmaxCrossEntropy {}
    }
}

impl UnconnectedLayer for SoftmaxCrossEntropy {
    type Inputs = Record! {
        x: Array2<f32>,
        t: Array2<f32>
    };
    type Output = f32;
    type Optimizer = SoftmaxCrossEntropyOptimizer;
    type Placeholders = HNil;

    fn forward(
        &self,
        placeholders: Self::Placeholders,
        inputs: Self::Inputs,
    ) -> (Self::Output, Self::Optimizer) {
        record_dest!({
            x,
            t,
        } = inputs);

        let y = arr_functions::softmax_batch(x.view());
        let loss = arr_functions::cross_entropy_error_batch(y.view(), t.view());

        (loss, SoftmaxCrossEntropyOptimizer { t, y })
    }
}
