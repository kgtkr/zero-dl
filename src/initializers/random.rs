use crate::initializer::Initializer;




use ndarray::Array;

use ndarray::{Dimension, IntoDimension};
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;


pub struct Random<D> {
    dim: D,
}

impl<D: Dimension> Random<D>
where
    Random<D>: Initializer,
{
    pub fn new(dim: impl IntoDimension<Dim = D>) -> Self {
        Random {
            dim: dim.into_dimension(),
        }
    }
}

impl<D: Dimension> Initializer for Random<D> {
    type Output = Array<f32, D>;

    fn initial_value(&self) -> Self::Output {
        let weight_init_std = 0.01;
        Array::random(self.dim.clone(), Normal::new(0., 1.).unwrap()) * weight_init_std
    }
}
