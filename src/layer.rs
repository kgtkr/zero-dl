use crate::arr_functions;
use crate::hlist_extra::ConcatAndSplit;
use frunk::labelled::{Field, LabelledGeneric, Transmogrifier};
use frunk::{HCons, HNil};
use ndarray::prelude::*;
use ndarray::Zip;
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::marker::PhantomData;
use std::sync::Arc;

pub trait LayerOutput {
    // 微分した時の型。変数を除いて基本的にSelfになる
    type Grad;
}

impl LayerOutput for f32 {
    type Grad = Self;
}

impl<Ix> LayerOutput for Array<f32, Ix> {
    type Grad = Self;
}

impl<Ix> LayerOutput for ArcArray<f32, Ix> {
    type Grad = Self;
}

impl<'a, Ix> LayerOutput for ArrayView<'a, f32, Ix> {
    type Grad = Self;
}

pub trait Optimizer {
    type Output: LayerOutput;

    fn optimize(self, dout: <Self::Output as LayerOutput>::Grad, learning_rate: f32);
}

pub trait Layer {
    type Output;
    type Optimizer: Optimizer;
    type Placeholders;

    fn forward(&self, placeholders: Self::Placeholders) -> (Self::Output, Self::Optimizer);
}

impl<T: Layer> Layer for &T {
    type Output = T::Output;
    type Optimizer = T::Optimizer;
    type Placeholders = T::Placeholders;

    fn forward(&self, placeholders: Self::Placeholders) -> (Self::Output, Self::Optimizer) {
        (*self).forward(placeholders)
    }
}
