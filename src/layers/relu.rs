use crate::layer::{Layer, LayerValue, Optimizer};
use ndarray::prelude::*;
use ndarray::Zip;
use std::marker::PhantomData;

pub struct ReluOptimizer<XOpz, D> {
    pub x: Array<f32, D>,
    pub x_optimizer: XOpz,
}

impl<XOpz, D: Dimension> Optimizer for ReluOptimizer<XOpz, D>
where
    XOpz: Optimizer<Output = Array<f32, D>>,
{
    type Output = Array<f32, D>;

    fn optimize(self, mut dout: <Self::Output as LayerValue>::Grad, learning_rate: f32) {
        Zip::from(&mut dout).and(&self.x).apply(|dout_x, &x| {
            if x <= 0. {
                *dout_x = 0.;
            }
        });

        self.x_optimizer.optimize(dout, learning_rate);
    }
}

pub struct Relu<XL, D> {
    pub x_layer: XL,
    pub phantom: PhantomData<D>,
}

impl<XL, D: Dimension> Relu<XL, D>
where
    Self: Layer,
{
    pub fn new(x_layer: XL) -> Self {
        Relu {
            x_layer,
            phantom: PhantomData,
        }
    }
}

impl<XL, D: Dimension> Layer for Relu<XL, D>
where
    XL: Layer<Output = Array<f32, D>>,
    ReluOptimizer<XL::Optimizer, D>: Optimizer<Output = Array<f32, D>>,
{
    type Output = Array<f32, D>;
    type Optimizer = ReluOptimizer<XL::Optimizer, D>;
    type Placeholders = XL::Placeholders;

    fn forward(&self, placeholders: Self::Placeholders) -> (Self::Output, Self::Optimizer) {
        let (x, x_optimizer) = self.x_layer.forward(placeholders);
        let y = x.mapv(|x| x.max(0.));
        (y, ReluOptimizer { x, x_optimizer })
    }
}
