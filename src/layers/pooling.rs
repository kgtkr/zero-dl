use crate::arr_functions;
use crate::layer::{UnconnectedBackward, UnconnectedLayer};
use frunk::traits::ToMut;
use frunk::HNil;
use ndarray::prelude::*;

use ndarray_stats::QuantileExt;

pub struct PoolingBackward {
    pub x: Array4<f32>,
    pub arg_max: Array1<usize>,
    pub pool_h: usize,
    pub pool_w: usize,
    pub stride: usize,
    pub pad: usize,
}

impl UnconnectedBackward for PoolingBackward {
    type Inputs = Record! {
        x: Array4<f32>
    };
    type Output = Array4<f32>;
    type Variables = HNil;

    fn backward(self, dout: Self::Output) -> (Self::Inputs, Self::Variables) {
        let dout = dout.permuted_axes([0, 2, 3, 1]);

        let pool_size = self.pool_h * self.pool_w;
        let mut dmax = Array::zeros((dout.len(), pool_size));

        let dout_flatten = dout.iter().cloned().collect::<Array1<_>>();

        for &i in self
            .arg_max
            .iter()
            .collect::<std::collections::HashSet<_>>()
        {
            dmax.slice_mut(s![0..self.arg_max.len(), i])
                .assign(&dout_flatten);
        }

        let dout_dim = dout.dim();
        let dmax = dmax
            .to_shared()
            .reshape((dout_dim.0, dout_dim.1, dout_dim.2, dout_dim.3, pool_size));

        let dmax_len = dmax.len();
        let dmax_dim = dmax.dim();
        let dcol = dmax.to_shared().reshape((
            dmax_dim.0 * dmax_dim.1 * dmax_dim.2,
            dmax_len / dmax_dim.0 / dmax_dim.1 / dmax_dim.2,
        ));

        let dx = arr_functions::col2im(
            dcol.view(),
            self.x.dim(),
            self.pool_h,
            self.pool_w,
            self.stride,
            self.pad,
        );

        (
            record! {
                x: dx
            },
            HNil,
        )
    }
}

pub struct Pooling {
    pub pool_h: usize,
    pub pool_w: usize,
    pub stride: usize,
    pub pad: usize,
}

impl Pooling
where
    Self: UnconnectedLayer,
{
    pub fn new(pool_h: usize, pool_w: usize, stride: usize, pad: usize) -> Pooling {
        Pooling {
            pool_h,
            pool_w,
            stride,
            pad,
        }
    }
}

impl UnconnectedLayer for Pooling {
    type Inputs = Record! {
        x: Array4<f32>
    };
    type Output = Array4<f32>;
    type Backward = PoolingBackward;
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

        let (N, C, H, W) = x.dim();
        let out_h = 1 + (H - self.pool_h) / self.stride;
        let out_w = 1 + (W - self.pool_w) / self.stride;

        let col =
            arr_functions::im2col(x.view(), self.pool_h, self.pool_w, self.stride, self.pad).0;
        let col_len = col.len();
        let col = col.to_shared().reshape((
            col_len / self.pool_h / self.pool_w,
            self.pool_h * self.pool_w,
        ));

        let arg_max = col.map_axis(Axis(1), |x| x.argmax().unwrap());
        let out = col.map_axis(Axis(1), |x| *x.max().unwrap());
        let out = out
            .to_shared()
            .reshape((N, out_h, out_w, C))
            .permuted_axes([0, 3, 1, 2])
            .to_owned();

        (
            out,
            PoolingBackward {
                x,
                arg_max,
                stride: self.stride,
                pool_h: self.pool_h,
                pool_w: self.pool_w,
                pad: self.pad,
            },
        )
    }

    fn initial_variables(&self) -> Self::Variables {
        HNil
    }
}
