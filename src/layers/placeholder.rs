use crate::layer::{UnconnectedBackward, UnconnectedLayer};
use frunk::labelled::Field;
use frunk::traits::ToMut;
use frunk::{HCons, HNil};
use std::marker::PhantomData;

#[derive(Debug, Clone)]
pub struct PlaceholderBackward<K, V> {
    pub phantom: PhantomData<(K, V)>,
}

impl<K, V> UnconnectedBackward for PlaceholderBackward<K, V> {
    type Inputs = Record! {};
    type Output = V;
    type Variables = HNil;

    fn backward(self, _dout: Self::Output) -> (Self::Inputs, Self::Variables) {
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
    type Backward = PlaceholderBackward<K, V>;
    type Placeholders = HCons<Field<K, V>, HNil>;
    type Variables = HNil;

    fn forward(
        &self,
        placeholders: Self::Placeholders,
        _variables: Self::Variables,
        inputs: Self::Inputs,
    ) -> (Self::Output, Self::Backward) {
        record_dest!({} = inputs);

        (
            placeholders.head.value,
            PlaceholderBackward {
                phantom: PhantomData,
            },
        )
    }

    fn initial_variables(&self) -> Self::Variables {
        HNil
    }
}
