use crate::layer::{
    LabelledLayerValues, Layer, LayerValue, Optimizer, UnconnectedLayer, UnconnectedOptimizer,
};
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

    fn optimize(
        self,
        mut dout: <Self::Output as LayerValue>::Grad,
        learning_rate: f32,
    ) -> <Self::Inputs as LabelledLayerValues>::Grads {
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

impl<D: Dimension> Relu<D> {
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

    fn forward(
        &self,
        placeholders: Self::Placeholders,
        inputs: Self::Inputs,
    ) -> (Self::Output, Self::Optimizer) {
        record_dest!({
            x,
        } = inputs);

        let y = x.mapv(|x| x.max(0.));
        (y, ReluOptimizer { x })
    }
}
