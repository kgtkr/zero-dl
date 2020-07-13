use crate::arr_functions;
use crate::layer::UnconnectedBackward;
use crate::layer::UnconnectedLayer;

use frunk::traits::ToMut;
use frunk::HNil;

use ndarray::prelude::*;

pub struct ConvolutionBackward {
    pub weight: Array4<f32>,
    pub bias: Array1<f32>,
    pub x: Array4<f32>,
    pub stride: usize,
    pub pad: usize,
    pub col: Array2<f32>,
    pub col_W: Array2<f32>,
}

impl UnconnectedBackward for ConvolutionBackward {
    type Inputs = Record! {
        x: Array4<f32>,
        weight: Array4<f32>,
        bias: Array1<f32>
    };
    type Output = Array4<f32>;
    type Variables = HNil;

    fn backward(self, dout: Self::Output) -> (Self::Inputs, Self::Variables) {
        let (FN, C, FH, FW) = self.weight.dim();
        let dout_len = dout.len();
        let dout = dout
            .permuted_axes([0, 2, 3, 1])
            .to_shared()
            .reshape((dout_len / FN, FN));

        let db = dout.sum_axis(Axis(0));
        let dW = self.col.t().dot(&dout);
        let dW = dW.t().to_shared().reshape((FN, C, FH, FW)).to_owned();

        let dcol = dout.dot(&self.col_W.t());
        let dx = arr_functions::col2im(dcol.view(), self.x.dim(), FH, FW, self.stride, self.pad);

        (
            record! {
                x: dx,
                weight: dW,
                bias: db
            },
            HNil,
        )
    }
}

pub struct Convolution {
    pub stride: usize,
    pub pad: usize,
}

impl Convolution
where
    Self: UnconnectedLayer,
{
    pub fn new(stride: usize, pad: usize) -> Self {
        Convolution { stride, pad }
    }
}

impl UnconnectedLayer for Convolution {
    type Inputs = Record! {
        x: Array4<f32>,
        weight: Array4<f32>,
        bias: Array1<f32>
    };
    type Output = Array4<f32>;
    type Backward = ConvolutionBackward;
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
            weight,
            bias,
        } = inputs);

        let (FN, _C, FH, FW) = weight.dim();
        let (N, C, H, W) = x.dim();
        let out_h = 1 + (H + 2 * self.pad - FH) / self.stride;
        let out_w = 1 + (W + 2 * self.pad - FW) / self.stride;

        let col = arr_functions::im2col(x.view(), FH, FW, self.stride, self.pad).0;
        let col_W = weight.to_shared().reshape((FN, C * FH * FW)).t().to_owned();

        let out = col.dot(&col_W) + &bias;
        let out_len = out.len();
        let out = out
            .to_shared()
            .reshape((N, out_h, out_w, out_len / N / out_h / out_w))
            .permuted_axes([0, 3, 1, 2])
            .to_owned();

        (
            out,
            ConvolutionBackward {
                weight,
                bias,
                x,
                col,
                col_W,
                stride: self.stride,
                pad: self.pad,
            },
        )
    }

    fn initial_variables(&self) -> Self::Variables {
        HNil
    }
}
