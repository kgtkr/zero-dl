use crate::layer::{Layer, LayerValue, Optimizer};
use frunk::HNil;

pub trait NetworkVar: Clone + LayerValue {
    fn optimize(&self, grad: &Self::Grad, learning_rate: f32);
}

#[derive(Debug, Clone)]
pub struct VariableOptimizer<V: NetworkVar> {
    pub value: V,
}

impl<V: NetworkVar> Optimizer for VariableOptimizer<V> {
    type Output = V;

    fn optimize(self, dout: <Self::Output as LayerValue>::Grad, learning_rate: f32) {
        &self.value.optimize(&dout, learning_rate);
    }
}

#[derive(Debug, Clone)]
pub struct Variable<V: NetworkVar> {
    pub value: V,
}

impl<V: NetworkVar> Variable<V>
where
    Self: Layer,
{
    pub fn new(value: V) -> Self {
        Variable { value }
    }
}

impl<V: NetworkVar> Layer for Variable<V> {
    type Output = V;

    type Optimizer = VariableOptimizer<V>;

    type Placeholders = HNil;

    fn forward(&self, placeholders: Self::Placeholders) -> (Self::Output, Self::Optimizer) {
        (
            self.value.clone(),
            VariableOptimizer {
                value: self.value.clone(),
            },
        )
    }
}
