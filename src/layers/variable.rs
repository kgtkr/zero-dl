use crate::layer::{
    LabelledLayerValues, Layer, LayerValue, Optimizer, UnconnectedLayer, UnconnectedOptimizer,
};
use frunk::HNil;

pub trait NetworkVar: Clone + LayerValue {
    fn optimize(&self, grad: &Self::Grad, learning_rate: f32);
}

#[derive(Debug, Clone)]
pub struct VariableOptimizer<V: NetworkVar> {
    pub value: V,
}

impl<V: NetworkVar> UnconnectedOptimizer for VariableOptimizer<V> {
    type Inputs = Record! {};
    type Output = V;

    fn optimize(
        self,
        dout: <Self::Output as LayerValue>::Grad,
        learning_rate: f32,
    ) -> <Self::Inputs as LabelledLayerValues>::Grads {
        &self.value.optimize(&dout, learning_rate);

        record! {}
    }
}

#[derive(Debug, Clone)]
pub struct Variable<V: NetworkVar> {
    pub value: V,
}

impl<V: NetworkVar> Variable<V> {
    pub fn new(value: V) -> Self {
        Variable { value }
    }
}

impl<V: NetworkVar> UnconnectedLayer for Variable<V> {
    type Inputs = Record! {};
    type Output = V;
    type Optimizer = VariableOptimizer<V>;
    type Placeholders = HNil;

    fn forward(
        &self,
        placeholders: Self::Placeholders,
        inputs: Self::Inputs,
    ) -> (Self::Output, Self::Optimizer) {
        record_dest!({} = inputs);

        (
            self.value.clone(),
            VariableOptimizer {
                value: self.value.clone(),
            },
        )
    }
}
