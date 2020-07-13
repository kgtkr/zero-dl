use crate::arr_functions;
use crate::layer::{UnconnectedBackward, UnconnectedLayer};
use frunk::traits::ToMut;
use frunk::HNil;
use ndarray::prelude::*;

pub struct SoftmaxCrossEntropyBackward {
    pub y: Array2<f32>,
    pub t: Array2<f32>,
}

impl UnconnectedBackward for SoftmaxCrossEntropyBackward {
    type Inputs = Record! {
        x: Array2<f32>,
        t: Array2<f32>
    };
    type Output = f32;
    type Variables = HNil;

    fn backward(self, _dout: f32) -> (Self::Inputs, Self::Variables) {
        let batch_size = self.t.len_of(Axis(0));
        let dx = (&self.y - &self.t) / batch_size as f32;

        /**
         * TODO: doutが1の時しか正常に動かない
         * またdtは正常な値でない
         */
        (
            record! {
                x: dx,
                t: Array2::zeros((0,0))
            },
            HNil,
        )
    }
}

pub struct SoftmaxCrossEntropy {}

impl SoftmaxCrossEntropy
where
    Self: UnconnectedLayer,
{
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
    type Backward = SoftmaxCrossEntropyBackward;
    type Placeholders = HNil;
    type Variables = HNil;

    fn forward(
        &self,
        _placeholders: Self::Placeholders,
        _variables: Self::Variables,
        inputs: Self::Inputs,
    ) -> (Self::Output, Self::Backward) {
        record_dest!({
            x,
            t,
        } = inputs);

        let y = arr_functions::softmax_batch(x.view());
        let loss = arr_functions::cross_entropy_error_batch(y.view(), t.view());

        (loss, SoftmaxCrossEntropyBackward { t, y })
    }

    fn initial_variables(&self) -> Self::Variables {
        HNil
    }
}
