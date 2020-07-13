use crate::initializer::Initializer;
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

pub struct Zero<D> {
    dim: D,
}

impl<D: Dimension> Zero<D>
where
    Zero<D>: Initializer,
{
    pub fn new(dim: impl IntoDimension<Dim = D>) -> Self {
        Zero {
            dim: dim.into_dimension(),
        }
    }
}

impl<D: Dimension> Initializer for Zero<D> {
    type Output = Array<f32, D>;

    fn initial_value(&self) -> Self::Output {
        Array::zeros(self.dim.clone())
    }
}
