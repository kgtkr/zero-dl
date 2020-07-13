use crate::layer::{UnconnectedBackward, UnconnectedLayer};
use frunk::traits::ToMut;
use frunk::HNil;
use ndarray::prelude::*;
use std::marker::PhantomData;

pub struct NDimTo2DimBackward<D: Dimension> {
    pub original_x_shape: D::Pattern,
}

impl<D: Dimension> UnconnectedBackward for NDimTo2DimBackward<D> {
    type Inputs = Record! {
        x: Array<f32, D>
    };
    type Output = Array2<f32>;
    type Variables = HNil;

    fn backward(self, dout: Self::Output) -> (Self::Inputs, Self::Variables) {
        let dx = dout.to_shared().reshape(self.original_x_shape).to_owned();

        (
            record! {
                x: dx
            },
            HNil,
        )
    }
}

pub struct NDimTo2Dim<D> {
    pub phantom: PhantomData<D>,
}

impl<D> NDimTo2Dim<D>
where
    Self: UnconnectedLayer,
{
    pub fn new() -> NDimTo2Dim<D> {
        NDimTo2Dim {
            phantom: PhantomData,
        }
    }
}

impl<D: Dimension> UnconnectedLayer for NDimTo2Dim<D> {
    type Inputs = Record! {
        x: Array<f32, D>
    };
    type Output = Array2<f32>;
    type Backward = NDimTo2DimBackward<D>;
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
        } = inputs);

        let original_x_shape = x.dim();
        let first_len = x.shape()[0];

        let x_len = x.len();
        let out = x
            .to_shared()
            .reshape((first_len, x_len / first_len))
            .to_owned();

        (out, NDimTo2DimBackward { original_x_shape })
    }

    fn initial_variables(&self) -> Self::Variables {
        HNil
    }
}
