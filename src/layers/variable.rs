use crate::layer::{Layer, Optimizer, UnconnectedLayer, UnconnectedOptimizer};
use frunk::labelled::Field;
use frunk::traits::ToMut;
use frunk::{HCons, HNil};
use ndarray::Array;
use ndarray::Zip;
use ndarray::{Dimension, IntoDimension};
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;
use std::marker::PhantomData;

pub fn initialize_zero<D: IntoDimension>(dim: D) -> Array<f32, D::Dim> {
    Array::zeros(dim.into_dimension())
}

pub fn initialize_random<D: IntoDimension>(dim: D) -> Array<f32, D::Dim> {
    let weight_init_std = 0.01;
    Array::random(dim.into_dimension(), Normal::new(0., 1.).unwrap()) * weight_init_std
}

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
pub struct Variable<K, D> {
    pub phantom: PhantomData<(K, D)>,
}

impl<K, D> Variable<K, D>
where
    Self: UnconnectedLayer,
{
    pub fn new() -> Self {
        Variable {
            phantom: PhantomData,
        }
    }
}

impl<K: 'static, D: Dimension + 'static> UnconnectedLayer for Variable<K, D> {
    type Inputs = Record! {};
    type Output = Array<f32, D>;
    type Optimizer = VariableOptimizer<K, D>;
    type Placeholders = HNil;
    type Variables = HCons<Field<K, Array<f32, D>>, HNil>;

    fn forward(
        &self,
        placeholders: Self::Placeholders,
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
}
