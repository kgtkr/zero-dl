use crate::arr_functions;
use crate::layer::{Layer, LayerOutput, Optimizer};
use ndarray::prelude::*;
use ndarray::Zip;
use ndarray_stats::QuantileExt;

pub struct PoolingOptimizer<XOpz> {
    pub x: Array4<f32>,
    pub x_optimizer: XOpz,
    pub arg_max: Array1<usize>,
    pub pool_h: usize,
    pub pool_w: usize,
    pub stride: usize,
    pub pad: usize,
}

impl<XOpz> Optimizer for PoolingOptimizer<XOpz>
where
    XOpz: Optimizer<Output = Array4<f32>>,
{
    type Output = Array4<f32>;

    fn optimize(self, dout: <Self::Output as LayerOutput>::Grad, learning_rate: f32) {
        let dout = dout.permuted_axes([0, 2, 3, 1]);

        let pool_size = self.pool_h * self.pool_w;
        let mut dmax = Array::zeros((dout.len(), pool_size));

        let dout_flatten = dout.iter().cloned().collect::<Array1<_>>();
        for &i in &self.arg_max {
            dmax.slice_mut(s![0..self.arg_max.len(), i])
                .assign(&dout_flatten);
        }

        let dmax = dmax
            .into_shape({
                let mut tmp = dout.dim();
                tmp.0 += pool_size;
                tmp
            })
            .unwrap();

        let dmax_len = dmax.len();
        let dmax_dim = dmax.dim();
        let dcol = dmax
            .into_shape((
                dmax_dim.0 * dmax_dim.1 * dmax_dim.2,
                dmax_len / dmax_dim.0 / dmax_dim.1 / dmax_dim.2,
            ))
            .unwrap();
        let dx = arr_functions::col2im(
            dcol.view(),
            self.x.dim(),
            self.pool_h,
            self.pool_w,
            self.stride,
            self.pad,
        );

        self.x_optimizer.optimize(dx, learning_rate);
    }
}

pub struct Pooling<XL> {
    pub x_layer: XL,
    pub pool_h: usize,
    pub pool_w: usize,
    pub stride: usize,
    pub pad: usize,
}

impl<XL> Pooling<XL>
where
    Self: Layer,
{
    pub fn new(
        x_layer: XL,
        pool_h: usize,
        pool_w: usize,
        stride: usize,
        pad: usize,
    ) -> Pooling<XL> {
        Pooling {
            x_layer,
            pool_h,
            pool_w,
            stride,
            pad,
        }
    }
}

impl<XL> Layer for Pooling<XL>
where
    XL: Layer<Output = Array4<f32>>,
    PoolingOptimizer<XL::Optimizer>: Optimizer<Output = Array4<f32>>,
{
    type Output = Array4<f32>;
    type Optimizer = PoolingOptimizer<XL::Optimizer>;
    type Placeholders = XL::Placeholders;

    fn forward(&self, placeholders: Self::Placeholders) -> (Self::Output, Self::Optimizer) {
        let (x, x_optimizer) = self.x_layer.forward(placeholders);

        let (N, C, H, W) = x.dim();
        let out_h = 1 + (H - self.pool_h) / self.stride;
        let out_w = 1 + (W - self.pool_w) / self.stride;

        let col =
            arr_functions::im2col(x.view(), self.pool_h, self.pool_w, self.stride, self.pad).0;
        let col_len = col.len();
        let col = col
            .into_shape((
                col_len / self.pool_h / self.pool_w,
                self.pool_h * self.pool_w,
            ))
            .unwrap();

        let arg_max = col.map_axis(Axis(1), |x| x.argmax().unwrap());
        let out = col.map_axis(Axis(1), |x| *x.max().unwrap());
        let out = out
            .into_shape((N, out_h, out_w, C))
            .unwrap()
            .permuted_axes([0, 3, 1, 2]);

        (
            out,
            PoolingOptimizer {
                x,
                x_optimizer,
                arg_max,
                stride: self.stride,
                pool_h: self.pool_h,
                pool_w: self.pool_w,
                pad: self.pad,
            },
        )
    }
}
