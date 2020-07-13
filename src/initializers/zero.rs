use crate::initializer::Initializer;




use ndarray::Array;

use ndarray::{Dimension, IntoDimension};




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
