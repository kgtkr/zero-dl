use crate::layer::{Layer, Optimizer, UnconnectedLayer, UnconnectedOptimizer};
use frunk::traits::ToMut;
use frunk::{HCons, HNil};
use ndarray::prelude::*;
use ndarray::Zip;
use std::marker::PhantomData;

pub struct ReluOptimizer<D> {
    pub x: Array<f32, D>,
}

impl<D: Dimension> UnconnectedOptimizer for ReluOptimizer<D> {
    type Inputs = Record! {
        x: Array<f32, D>
    };
    type Output = Array<f32, D>;
    type Variables = HNil;

    fn optimize<'a>(
        self,
        mut dout: Self::Output,
        variables: <Self::Variables as ToMut<'a>>::Output,
        learning_rate: f32,
    ) -> Self::Inputs {
        Zip::from(&mut dout).and(&self.x).apply(|dout_x, &x| {
            if x <= 0. {
                *dout_x = 0.;
            }
        });

        record! {
            x: dout
        }
    }
}

pub struct Relu<D> {
    pub phantom: PhantomData<D>,
}

impl<D: Dimension> Relu<D>
where
    Self: UnconnectedLayer,
{
    pub fn new() -> Self {
        Relu {
            phantom: PhantomData,
        }
    }
}

impl<D: Dimension> UnconnectedLayer for Relu<D> {
    type Inputs = Record! {
        x: Array<f32, D>
    };
    type Output = Array<f32, D>;
    type Optimizer = ReluOptimizer<D>;
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

        let y = x.mapv(|x| x.max(0.));
        (y, ReluOptimizer { x })
    }

    fn initial_variables(&self) -> Self::Variables {
        HNil
    }
}
