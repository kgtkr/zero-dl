use crate::hlist_extra::{Concat, ConcatIsInverseSplit, ConcatToMutIsToMutConcat, Split};
use frunk::labelled::Field;
use frunk::traits::ToMut;
use frunk::{field, HCons, HNil};

// LayerのLabelled HList
pub trait LabelledLayers {
    type Outputs;
    type Optimizers: LabelledOptimizers<Outputs = Self::Outputs, Variables = Self::Variables>;
    type Placeholders;
    type Variables: for<'a> ToMut<'a>;

    fn forward(
        &self,
        placeholders: Self::Placeholders,
        variables: Self::Variables,
    ) -> (Self::Outputs, Self::Optimizers);

    fn initial_variables(&self) -> Self::Variables;
}

impl LabelledLayers for HNil {
    type Outputs = HNil;
    type Optimizers = HNil;
    type Placeholders = HNil;
    type Variables = HNil;

    fn forward(
        &self,
        HNil: Self::Placeholders,
        _variables: Self::Variables,
    ) -> (Self::Outputs, Self::Optimizers) {
        (HNil, HNil)
    }

    fn initial_variables(&self) -> Self::Variables {
        HNil
    }
}

impl<Name, Type: Layer, Tail: LabelledLayers> LabelledLayers for HCons<Field<Name, Type>, Tail>
where
    Type::Placeholders: ConcatIsInverseSplit<Tail::Placeholders>,
    Type::Variables: for<'a> ConcatToMutIsToMutConcat<'a, Tail::Variables>,
{
    type Outputs = HCons<Field<Name, Type::Output>, Tail::Outputs>;
    type Optimizers = HCons<Field<Name, Type::Optimizer>, Tail::Optimizers>;
    type Placeholders = <Type::Placeholders as Concat<Tail::Placeholders>>::Output;
    type Variables = <Type::Variables as Concat<Tail::Variables>>::Output;

    fn forward(
        &self,
        placeholders: Self::Placeholders,
        variables: Self::Variables,
    ) -> (Self::Outputs, Self::Optimizers) {
        let (head_placeholders, tail_placeholders) = placeholders.split();
        let (head_variables, tail_variables) = variables.split();
        let (head_output, head_optimizer) =
            self.head.value.forward(head_placeholders, head_variables);
        let (tail_outputs, tail_optimizers) = self.tail.forward(tail_placeholders, tail_variables);
        (
            HCons {
                head: field![Name, head_output],
                tail: tail_outputs,
            },
            HCons {
                head: field![Name, head_optimizer],
                tail: tail_optimizers,
            },
        )
    }

    fn initial_variables(&self) -> Self::Variables {
        let head_variables = self.head.value.initial_variables();
        let tail_variables = self.tail.initial_variables();
        head_variables.concat(tail_variables)
    }
}

// OptimizerのLabelled HList
pub trait LabelledOptimizers: Sized {
    type Outputs;
    type Variables: for<'a> ToMut<'a>;

    fn optimize<'a>(
        self,
        douts: Self::Outputs,
        variables: <Self::Variables as ToMut<'a>>::Output,
        learning_rate: f32,
    );
}

impl LabelledOptimizers for HNil {
    type Outputs = HNil;
    type Variables = HNil;

    fn optimize<'a>(
        self,
        _douts: Self::Outputs,
        _variables: <Self::Variables as ToMut<'a>>::Output,
        _learning_rate: f32,
    ) {
    }
}

impl<Name, Type: Optimizer, Tail: LabelledOptimizers> LabelledOptimizers
    for HCons<Field<Name, Type>, Tail>
where
    Type::Variables: for<'a> ConcatToMutIsToMutConcat<'a, Tail::Variables>,
{
    type Outputs = HCons<Field<Name, Type::Output>, Tail::Outputs>;
    type Variables = <Type::Variables as Concat<Tail::Variables>>::Output;

    fn optimize<'a>(
        self,
        douts: Self::Outputs,
        variables: <Self::Variables as ToMut<'a>>::Output,
        learning_rate: f32,
    ) {
        let (type_variables, tail_variables) = variables.split();

        self.head
            .value
            .optimize(douts.head.value, type_variables, learning_rate);
        self.tail
            .optimize(douts.tail, tail_variables, learning_rate);
    }
}

pub trait Optimizer {
    type Output;
    type Variables: for<'a> ToMut<'a>;

    fn optimize<'a>(
        self,
        douts: Self::Output,
        variables: <Self::Variables as ToMut<'a>>::Output,
        learning_rate: f32,
    );
}

pub trait Layer {
    type Output;
    type Optimizer: Optimizer<Output = Self::Output, Variables = Self::Variables>;
    type Placeholders;
    type Variables: for<'a> ToMut<'a>;

    fn forward(
        &self,
        placeholders: Self::Placeholders,
        variables: Self::Variables,
    ) -> (Self::Output, Self::Optimizer);

    fn initial_variables(&self) -> Self::Variables;
}

// 親レイヤーと未接続のレイヤー
pub trait UnconnectedLayer: Sized {
    // 入力の名前と型のLabelled HList
    type Inputs;

    // このレイヤーで必要なプレースフォルダの名前と型のLabelled HList
    type Placeholders;

    // このレイヤーで必要な変数の名前と型のLabelled HList
    type Variables: for<'a> ToMut<'a>;

    // 出力の型
    type Output;

    type Optimizer: UnconnectedOptimizer<
        Inputs = Self::Inputs,
        Output = Self::Output,
        Variables = Self::Variables,
    >;

    fn join<I>(self, input_layers: I) -> LayerAdapter<I, Self>
    where
        LayerAdapter<I, Self>: Layer,
    {
        LayerAdapter {
            input_layers,
            layer: self,
        }
    }

    fn forward(
        &self,
        placeholders: Self::Placeholders,
        variables: Self::Variables,
        inputs: Self::Inputs,
    ) -> (Self::Output, Self::Optimizer);

    fn initial_variables(&self) -> Self::Variables;
}

// 親レイヤーと未接続のOptimizer
pub trait UnconnectedOptimizer {
    type Inputs;
    type Output;
    type Variables: for<'a> ToMut<'a>;

    fn optimize<'a>(
        self,
        douts: Self::Output,
        variables: <Self::Variables as ToMut<'a>>::Output,
        // TODO: 本来はこれOptimizerアルゴリズムのパラメーターにするべき
        learning_rate: f32,
    ) -> Self::Inputs;
}

pub struct LayerAdapter<I, L> {
    input_layers: I,
    layer: L,
}

impl<I: LabelledLayers, L: UnconnectedLayer<Inputs = I::Outputs>> Layer for LayerAdapter<I, L>
where
    I::Placeholders: ConcatIsInverseSplit<L::Placeholders>,
    I::Variables: for<'a> ConcatToMutIsToMutConcat<'a, L::Variables>,
    OptimizerAdapter<I::Optimizers, L::Optimizer>:
        Optimizer<Output = L::Output, Variables = <I::Variables as Concat<L::Variables>>::Output>,
{
    type Output = L::Output;
    type Optimizer = OptimizerAdapter<I::Optimizers, L::Optimizer>;
    type Placeholders = <I::Placeholders as Concat<L::Placeholders>>::Output;
    type Variables = <I::Variables as Concat<L::Variables>>::Output;

    fn forward(
        &self,
        placeholders: Self::Placeholders,
        variables: Self::Variables,
    ) -> (Self::Output, Self::Optimizer) {
        let (input_placeholders, layer_placeholders) = placeholders.split();
        let (input_variables, layer_variables) = variables.split();
        let (inputs, input_optimizers) = self
            .input_layers
            .forward(input_placeholders, input_variables);
        let (output, optimizer) = self
            .layer
            .forward(layer_placeholders, layer_variables, inputs);

        (
            output,
            OptimizerAdapter {
                input_optimizers,
                optimizer,
            },
        )
    }

    fn initial_variables(&self) -> Self::Variables {
        let input_variables = self.input_layers.initial_variables();
        let layer_variables = self.layer.initial_variables();
        input_variables.concat(layer_variables)
    }
}

pub struct OptimizerAdapter<I, O> {
    input_optimizers: I,
    optimizer: O,
}

impl<I: LabelledOptimizers, O: UnconnectedOptimizer<Inputs = I::Outputs>> Optimizer
    for OptimizerAdapter<I, O>
where
    I::Variables: for<'a> ConcatToMutIsToMutConcat<'a, O::Variables>,
{
    type Output = O::Output;
    type Variables = <I::Variables as Concat<O::Variables>>::Output;

    fn optimize<'a>(
        self,
        dout: Self::Output,
        variables: <Self::Variables as ToMut<'a>>::Output,
        learning_rate: f32,
    ) {
        let (head_variables, input_variables) = variables.split();

        let grads = self
            .optimizer
            .optimize(dout, input_variables, learning_rate);
        self.input_optimizers
            .optimize(grads, head_variables, learning_rate);
    }
}

impl<T: Layer> Layer for &T {
    type Output = T::Output;
    type Optimizer = T::Optimizer;
    type Placeholders = T::Placeholders;
    type Variables = T::Variables;

    fn forward(
        &self,
        placeholders: Self::Placeholders,
        variables: Self::Variables,
    ) -> (Self::Output, Self::Optimizer) {
        (*self).forward(placeholders, variables)
    }

    fn initial_variables(&self) -> Self::Variables {
        (*self).initial_variables()
    }
}
