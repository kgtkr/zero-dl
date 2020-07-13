use crate::layer::{UnconnectedLayer, UnconnectedOptimizer};
use frunk::labelled::Field;
use frunk::traits::ToMut;
use frunk::{HCons, HNil};
use std::marker::PhantomData;

#[derive(Debug, Clone)]
pub struct PlaceholderOptimizer<K, V> {
    pub phantom: PhantomData<(K, V)>,
}

impl<K, V> UnconnectedOptimizer for PlaceholderOptimizer<K, V> {
    type Inputs = Record! {};
    type Output = V;
    type Variables = HNil;

    fn optimize(self, _dout: Self::Output) -> (Self::Inputs, Self::Variables) {
        (record! {}, HNil)
    }
}

#[derive(Debug, Clone)]
pub struct Placeholder<K, V> {
    pub phantom: PhantomData<(K, V)>,
}

impl<K, V> Placeholder<K, V>
where
    Self: UnconnectedLayer,
{
    pub fn new() -> Self {
        Placeholder {
            phantom: PhantomData,
        }
    }
}

impl<K, V> UnconnectedLayer for Placeholder<K, V> {
    type Inputs = Record! {};
    type Output = V;
    type Optimizer = PlaceholderOptimizer<K, V>;
    type Placeholders = HCons<Field<K, V>, HNil>;
    type Variables = HNil;

    fn forward(
        &self,
        placeholders: Self::Placeholders,
        _variables: Self::Variables,
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

    fn initial_variables(&self) -> Self::Variables {
        HNil
    }
}
