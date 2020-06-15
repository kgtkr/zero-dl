use crate::layer::{
    LabelledLayerValues, Layer, LayerValue, Optimizer, UnconnectedLayer, UnconnectedOptimizer,
};
use frunk::labelled::Field;
use frunk::{HCons, HNil};
use std::marker::PhantomData;

#[derive(Debug, Clone)]
pub struct PlaceholderOptimizer<K, V> {
    pub phantom: PhantomData<(K, V)>,
}

impl<K, V: LayerValue> UnconnectedOptimizer for PlaceholderOptimizer<K, V> {
    type Inputs = Record! {};
    type Output = V;

    fn optimize(
        self,
        dout: <Self::Output as LayerValue>::Grad,
        learning_rate: f32,
    ) -> <Self::Inputs as LabelledLayerValues>::Grads {
        record! {}
    }
}

#[derive(Debug, Clone)]
pub struct Placeholder<K, V> {
    pub phantom: PhantomData<(K, V)>,
}

impl<K, V> Placeholder<K, V> {
    pub fn new() -> Self {
        Placeholder {
            phantom: PhantomData,
        }
    }
}

impl<K, V: LayerValue> UnconnectedLayer for Placeholder<K, V> {
    type Inputs = Record! {};
    type Output = V;
    type Optimizer = PlaceholderOptimizer<K, V>;
    type Placeholders = HCons<Field<K, V>, HNil>;

    fn forward(
        &self,
        placeholders: Self::Placeholders,
        inputs: Self::Inputs,
    ) -> (Self::Output, Self::Optimizer) {
        record_dest!({} = inputs);

        (
            placeholders.head.value,
            PlaceholderOptimizer {
                phantom: PhantomData,
            },
        )
    }
}
