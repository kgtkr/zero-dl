use ndarray::prelude::*;

// レイヤー間を流れる値
pub trait LayerValue {
    // 微分した時の型。変数を除いて基本的にSelfになる
    type Grad;
}

impl LayerValue for f32 {
    type Grad = Self;
}

impl<Ix> LayerValue for Array<f32, Ix> {
    type Grad = Self;
}

impl<Ix> LayerValue for ArcArray<f32, Ix> {
    type Grad = Self;
}

impl<'a, Ix> LayerValue for ArrayView<'a, f32, Ix> {
    type Grad = Self;
}

pub trait Optimizer {
    type Output: LayerValue;

    fn optimize(self, dout: <Self::Output as LayerValue>::Grad, learning_rate: f32);
}

pub trait Layer {
    type Output: LayerValue;
    type Optimizer: Optimizer<Output = Self::Output>;
    type Placeholders;

    fn forward(&self, placeholders: Self::Placeholders) -> (Self::Output, Self::Optimizer);
}

// 親レイヤーと未接続のOptimizer
// TODO:整備
pub trait UnconnectedOptimizer {
    type Output: LayerValue;

    fn optimize(self, dout: <Self::Output as LayerValue>::Grad);
}

// 親レイヤーと未接続のレイヤー
// TODO:整備
pub trait UnconnectedLayer {
    // 入力の名前と型のLabelled HList
    type Inputs;

    // このレイヤーで必要なプレースフォルダの名前と型のLabelled HList
    type Placeholders;

    // 出力の型
    type Output: LayerValue;

    type Optimizer: UnconnectedOptimizer<Output = Self::Output>;

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
