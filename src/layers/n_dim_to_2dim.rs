use crate::layer::{Layer, Optimizer, UnconnectedLayer, UnconnectedOptimizer};
use frunk::traits::ToMut;
use frunk::{HCons, HNil};
use ndarray::prelude::*;
use std::marker::PhantomData;

pub struct NDimTo2DimOptimizer<D: Dimension> {
    pub original_x_shape: D::Pattern,
}

impl<D: Dimension> UnconnectedOptimizer for NDimTo2DimOptimizer<D> {
    type Inputs = Record! {
        x: Array<f32, D>
    };
    type Output = Array2<f32>;
    type Variables = HNil;

    fn optimize<'a>(
        self,
        dout: Self::Output,
        variables: <Self::Variables as ToMut<'a>>::Output,
        learning_rate: f32,
    ) -> Self::Inputs {
        let dx = dout.to_shared().reshape(self.original_x_shape).to_owned();

        record! {
            x: dx
        }
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
    type Optimizer = NDimTo2DimOptimizer<D>;
    type Placeholders = HNil;
    type Variables = HNil;

    fn forward(
        &self,
        placeholders: Self::Placeholders,
        variables: Self::Variables,
        inputs: Self::Inputs,
    ) -> (Self::Output, Self::Optimizer) {
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

        (out, NDimTo2DimOptimizer { original_x_shape })
    }
}
