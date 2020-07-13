use crate::initializer::Initializer;
use crate::layer::{UnconnectedLayer, UnconnectedOptimizer};
use frunk::field;
use frunk::labelled::Field;
use frunk::traits::ToMut;
use frunk::{HCons, HNil};
use ndarray::Array;
use ndarray::Zip;
use ndarray::{Dimension};


use std::marker::PhantomData;

pub fn optimize<D: Dimension>(arr: &mut Array<f32, D>, grad: Array<f32, D>, learning_rate: f32) {
    Zip::from(arr)
        .and(&grad)
        .apply(|x, y| *x -= learning_rate * y)
}

#[derive(Debug, Clone)]
pub struct VariableOptimizer<K, D> {
    pub phantom: PhantomData<(K, D)>,
}

impl<K: 'static, D: Dimension + 'static> UnconnectedOptimizer for VariableOptimizer<K, D> {
    type Inputs = Record! {};
    type Output = Array<f32, D>;
    type Variables = HCons<Field<K, Array<f32, D>>, HNil>;

    fn optimize<'a>(
        self,
        dout: Self::Output,
        variables: <Self::Variables as ToMut<'a>>::Output,
        learning_rate: f32,
    ) -> Self::Inputs {
        optimize(&mut variables.head.value, dout, learning_rate);

        record! {}
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

impl<K: 'static, D: Dimension + 'static, I: Initializer<Output = Array<f32, D>>> UnconnectedLayer
    for Variable<K, D, I>
{
    type Inputs = Record! {};
    type Output = Array<f32, D>;
    type Optimizer = VariableOptimizer<K, D>;
    type Placeholders = HNil;
    type Variables = HCons<Field<K, Array<f32, D>>, HNil>;

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
