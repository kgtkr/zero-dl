use crate::layer::{Layer, LayerValue, Optimizer};
use frunk::labelled::Field;
use frunk::{HCons, HNil};
use std::marker::PhantomData;

#[derive(Debug, Clone)]
pub struct PlaceholderOptimizer<K, V> {
    pub phantom: PhantomData<(K, V)>,
}

impl<K, V: LayerValue> Optimizer for PlaceholderOptimizer<K, V> {
    type Output = V;

    fn optimize(self, dout: <Self::Output as LayerValue>::Grad, learning_rate: f32) {}
}

#[derive(Debug, Clone)]
pub struct Placeholder<K, V> {
    pub phantom: PhantomData<(K, V)>,
}

impl<K, V> Placeholder<K, V>
where
    Self: Layer,
{
    pub fn new() -> Self {
        Placeholder {
            phantom: PhantomData,
        }
    }
}

impl<K, V: LayerValue> Layer for Placeholder<K, V> {
    type Output = V;

    type Optimizer = PlaceholderOptimizer<K, V>;

    type Placeholders = HCons<Field<K, V>, HNil>;

    fn forward(&self, placeholders: Self::Placeholders) -> (Self::Output, Self::Optimizer) {
        (
            placeholders.head.value,
            PlaceholderOptimizer {
                phantom: PhantomData,
            },
        )
    }
}
