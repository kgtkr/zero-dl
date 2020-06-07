use crate::arr_functions;
use crate::hlist_extra::ConcatAndSplit;
use crate::layer::{Layer, Optimizer};
use ndarray::prelude::*;

pub struct SoftmaxWithLossOptimizer<XOpz, TOpz> {
    pub y: Array1<f32>,
    pub t: Array1<f32>,
    pub x_optimizer: XOpz,
    pub t_optimizer: TOpz,
}

impl<XOpz, TOpz> Optimizer for SoftmaxWithLossOptimizer<XOpz, TOpz>
where
    XOpz: Optimizer<Output = Array1<f32>>,
    TOpz: Optimizer<Output = Array1<f32>>,
{
    type Output = f32;

    fn optimize(self, dout: f32, learning_rate: f32) {
        let d = &self.y - &self.t;

        self.x_optimizer.optimize(d, learning_rate);
        // TODO: 本当はtの微分も考えるべきかも？
    }
}

pub struct SoftmaxWithLoss<XL, TL> {
    pub x_layer: XL,
    pub t_layer: TL,
}

impl<XL, TL> SoftmaxWithLoss<XL, TL>
where
    Self: Layer,
{
    pub fn new(x_layer: XL, t_layer: TL) -> Self {
        SoftmaxWithLoss { x_layer, t_layer }
    }
}

impl<XL, TL> Layer for SoftmaxWithLoss<XL, TL>
where
    XL: Layer<Output = Array1<f32>>,
    TL: Layer<Output = Array1<f32>>,
    SoftmaxWithLossOptimizer<XL::Optimizer, TL::Optimizer>: Optimizer<Output = f32>,
    XL::Placeholders: ConcatAndSplit<TL::Placeholders>,
{
    type Output = f32;
    type Optimizer = SoftmaxWithLossOptimizer<XL::Optimizer, TL::Optimizer>;
    type Placeholders = <XL::Placeholders as ConcatAndSplit<TL::Placeholders>>::Output;

    fn forward(&self, placeholders: Self::Placeholders) -> (Self::Output, Self::Optimizer) {
        let (x_placeholders, t_placeholders) = ConcatAndSplit::split(placeholders);

        let (x, x_optimizer) = self.x_layer.forward(x_placeholders);
        let (t, t_optimizer) = self.t_layer.forward(t_placeholders);

        let y = arr_functions::softmax_arr1(x.view());
        let loss = arr_functions::cross_entropy_error(y.view(), t.view());

        (
            loss,
            SoftmaxWithLossOptimizer {
                t,
                y,
                x_optimizer,
                t_optimizer,
            },
        )
    }
}
