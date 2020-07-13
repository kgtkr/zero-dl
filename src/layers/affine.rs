use crate::layer::{UnconnectedBackward, UnconnectedLayer};

use frunk::traits::ToMut;
use frunk::{field, hlist, HNil, Hlist};

use ndarray::prelude::*;

pub struct AffineBackward {
    pub weight: Array2<f32>,
    pub bias: Array1<f32>,
    pub x: Array2<f32>,
}

impl UnconnectedBackward for AffineBackward {
    type Inputs = Record! {
        x: Array2<f32>,
        weight: Array2<f32>,
        bias: Array1<f32>
    };
    type Output = Array2<f32>;
    type Variables = HNil;

    fn backward(self, dout: Self::Output) -> (Self::Inputs, Self::Variables) {
        let dx = dout.dot(&self.weight.t());

        let dw = self.x.t().dot(&dout);
        let db = dout.sum_axis(Axis(0));

        (
            record! {
                x: dx,
                weight: dw,
                bias: db
            },
            HNil,
        )
    }
}

pub struct Affine {}

impl Affine
where
    Self: UnconnectedLayer,
{
    pub fn new() -> Self {
        Affine {}
    }
}

impl UnconnectedLayer for Affine {
    type Inputs = Record! {
        x: Array2<f32>,
        weight: Array2<f32>,
        bias: Array1<f32>
    };
    type Placeholders = HNil;
    type Output = Array2<f32>;
    type Backward = AffineBackward;
    type Variables = HNil;

    fn forward(
        &self,
        HNil: Self::Placeholders,
        _variables: Self::Variables,
        inputs: Self::Inputs,
    ) -> (Self::Output, Self::Backward) {
        record_dest!({
            x,
            weight,
            bias,
        } = inputs);
        let y = x.dot(&weight) + &bias;

        (y, AffineBackward { x, weight, bias })
    }

    fn initial_variables(&self) -> Self::Variables {
        HNil
    }
}
