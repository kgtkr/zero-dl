use crate::initializer::Initializer;
use crate::layer::{UnconnectedLayer, UnconnectedOptimizer};
use frunk::field;
use frunk::labelled::Field;
use frunk::traits::ToMut;
use frunk::{HCons, HNil};

use std::marker::PhantomData;

/* pub fn optimize<D: Dimension>(arr: &mut Array<f32, D>, grad: Array<f32, D>, learning_rate: f32) {
    Zip::from(arr)
        .and(&grad)
        .apply(|x, y| *x -= learning_rate * y)
} */

#[derive(Debug, Clone)]
pub struct VariableOptimizer<K, D> {
    pub phantom: PhantomData<(K, D)>,
}

impl<K: 'static, V: 'static> UnconnectedOptimizer for VariableOptimizer<K, V> {
    type Inputs = Record! {};
    type Output = V;
    type Variables = HCons<Field<K, V>, HNil>;

    fn optimize(self, dout: Self::Output) -> (Self::Inputs, Self::Variables) {
        (
            record! {},
            HCons {
                head: field![K, dout],
                tail: HNil,
            },
        )
    }
}

#[derive(Debug, Clone)]
pub struct Variable<K, D, I> {
    pub phantom: PhantomData<(K, D)>,
    pub initializer: I,
}

impl<K, D, I> Variable<K, D, I>
where
    Self: UnconnectedLayer,
{
    pub fn new(initializer: I) -> Self {
        Variable {
            phantom: PhantomData,
            initializer,
        }
    }
}

impl<K: 'static, V: 'static, I: Initializer<Output = V>> UnconnectedLayer for Variable<K, V, I> {
    type Inputs = Record! {};
    type Output = V;
    type Optimizer = VariableOptimizer<K, V>;
    type Placeholders = HNil;
    type Variables = HCons<Field<K, V>, HNil>;

    fn forward(
        &self,
        _placeholders: Self::Placeholders,
        variables: Self::Variables,
        inputs: Self::Inputs,
    ) -> (Self::Output, Self::Optimizer) {
        record_dest!({} = inputs);

        (
            variables.head.value,
            VariableOptimizer {
                phantom: PhantomData,
            },
        )
    }

    fn initial_variables(&self) -> Self::Variables {
        HCons {
            head: field![K, self.initializer.initial_value()],
            tail: HNil,
        }
    }
}
