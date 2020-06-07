use crate::layer::{Layer, LayerOutput, Optimizer};
use ndarray::prelude::*;
use ndarray::Zip;

pub struct ReluOptimizer<XOpz> {
    pub x: Array1<f32>,
    pub x_optimizer: XOpz,
}

impl<XOpz> Optimizer for ReluOptimizer<XOpz>
where
    XOpz: Optimizer<Output = Array1<f32>>,
{
    type Output = Array1<f32>;

    fn optimize(self, mut dout: <Self::Output as LayerOutput>::Grad, learning_rate: f32) {
        Zip::from(&mut dout).and(&self.x).apply(|dout_x, &x| {
            if x <= 0. {
                *dout_x = 0.;
            }
        });

        self.x_optimizer.optimize(dout, learning_rate);
    }
}

pub struct Relu<XL> {
    pub x_layer: XL,
}

impl<XL> Relu<XL>
where
    Self: Layer,
{
    pub fn new(x_layer: XL) -> Relu<XL> {
        Relu { x_layer }
    }
}

impl<XL> Layer for Relu<XL>
where
    XL: Layer<Output = Array1<f32>>,
    ReluOptimizer<XL::Optimizer>: Optimizer<Output = Array1<f32>>,
{
    type Output = Array1<f32>;
    type Optimizer = ReluOptimizer<XL::Optimizer>;
    type Placeholders = XL::Placeholders;

    fn forward(&self, placeholders: Self::Placeholders) -> (Self::Output, Self::Optimizer) {
        let (x, x_optimizer) = self.x_layer.forward(placeholders);
        let y = x.mapv(|x| x.max(0.));
        (y, ReluOptimizer { x, x_optimizer })
    }
}
