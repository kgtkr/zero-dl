use ndarray::prelude::*;

pub trait LayerOutput {
    // 微分した時の型。変数を除いて基本的にSelfになる
    type Grad;
}

impl LayerOutput for f32 {
    type Grad = Self;
}

impl<Ix> LayerOutput for Array<f32, Ix> {
    type Grad = Self;
}

impl<Ix> LayerOutput for ArcArray<f32, Ix> {
    type Grad = Self;
}

impl<'a, Ix> LayerOutput for ArrayView<'a, f32, Ix> {
    type Grad = Self;
}

pub trait Optimizer {
    type Output: LayerOutput;

    fn optimize(self, dout: <Self::Output as LayerOutput>::Grad, learning_rate: f32);
}

pub trait Layer {
    type Output;
    type Optimizer: Optimizer;
    type Placeholders;

    fn forward(&self, placeholders: Self::Placeholders) -> (Self::Output, Self::Optimizer);
}

impl<T: Layer> Layer for &T {
    type Output = T::Output;
    type Optimizer = T::Optimizer;
    type Placeholders = T::Placeholders;

    fn forward(&self, placeholders: Self::Placeholders) -> (Self::Output, Self::Optimizer) {
        (*self).forward(placeholders)
    }
}
