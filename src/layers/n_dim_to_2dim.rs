use crate::layer::{Layer, LayerValue, Optimizer};
use ndarray::prelude::*;
use std::marker::PhantomData;

pub struct NDimTo2DimOptimizer<XOpz, D: Dimension> {
    pub original_x_shape: D::Pattern,
    pub x_optimizer: XOpz,
}

impl<XOpz, D: Dimension> Optimizer for NDimTo2DimOptimizer<XOpz, D>
where
    XOpz: Optimizer<Output = Array<f32, D>>,
{
    type Output = Array2<f32>;

    fn optimize(self, dout: <Self::Output as LayerValue>::Grad, learning_rate: f32) {
        let dx = dout.to_shared().reshape(self.original_x_shape).to_owned();

        self.x_optimizer.optimize(dx, learning_rate);
    }
}

pub struct NDimTo2Dim<XL, D> {
    pub x_layer: XL,
    pub phantom: PhantomData<D>,
}

impl<XL, D> NDimTo2Dim<XL, D>
where
    Self: Layer,
{
    pub fn new(x_layer: XL) -> NDimTo2Dim<XL, D> {
        NDimTo2Dim {
            x_layer,
            phantom: PhantomData,
        }
    }
}

impl<XL, D: Dimension> Layer for NDimTo2Dim<XL, D>
where
    XL: Layer<Output = Array<f32, D>>,
    NDimTo2DimOptimizer<XL::Optimizer, D>: Optimizer<Output = Array2<f32>>,
{
    type Output = Array2<f32>;
    type Optimizer = NDimTo2DimOptimizer<XL::Optimizer, D>;
    type Placeholders = XL::Placeholders;

    fn forward(&self, placeholders: Self::Placeholders) -> (Self::Output, Self::Optimizer) {
        let (x, x_optimizer) = self.x_layer.forward(placeholders);

        let original_x_shape = x.dim();
        let first_len = x.shape()[0];

        let x_len = x.len();
        let out = x
            .to_shared()
            .reshape((first_len, x_len / first_len))
            .to_owned();

        (
            out,
            NDimTo2DimOptimizer {
                original_x_shape,
                x_optimizer,
            },
        )
    }
}
