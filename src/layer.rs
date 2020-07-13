use crate::hlist_extra::{Concat, ConcatIsInverseSplit, ConcatToMutIsToMutConcat, Split};
use frunk::labelled::Field;
use frunk::traits::ToMut;
use frunk::{field, HCons, HNil};

// LayerのLabelled HList
pub trait LabelledLayers {
    type Outputs;
    type Backwards: LabelledBackwards<Outputs = Self::Outputs, Variables = Self::Variables>;
    type Placeholders;
    type Variables;

    fn forward(
        &self,
        placeholders: Self::Placeholders,
        variables: Self::Variables,
    ) -> (Self::Outputs, Self::Backwards);

    fn initial_variables(&self) -> Self::Variables;
}

impl LabelledLayers for HNil {
    type Outputs = HNil;
    type Backwards = HNil;
    type Placeholders = HNil;
    type Variables = HNil;

    fn forward(
        &self,
        HNil: Self::Placeholders,
        _variables: Self::Variables,
    ) -> (Self::Outputs, Self::Backwards) {
        (HNil, HNil)
    }

    fn initial_variables(&self) -> Self::Variables {
        HNil
    }
}

impl<Name, Type: Layer, Tail: LabelledLayers> LabelledLayers for HCons<Field<Name, Type>, Tail>
where
    Type::Placeholders: ConcatIsInverseSplit<Tail::Placeholders>,
    Type::Variables: ConcatIsInverseSplit<Tail::Variables>,
{
    type Outputs = HCons<Field<Name, Type::Output>, Tail::Outputs>;
    type Backwards = HCons<Field<Name, Type::Backward>, Tail::Backwards>;
    type Placeholders = <Type::Placeholders as Concat<Tail::Placeholders>>::Output;
    type Variables = <Type::Variables as Concat<Tail::Variables>>::Output;

    fn forward(
        &self,
        placeholders: Self::Placeholders,
        variables: Self::Variables,
    ) -> (Self::Outputs, Self::Backwards) {
        let (head_placeholders, tail_placeholders) = placeholders.split();
        let (head_variables, tail_variables) = variables.split();
        let (head_output, head_backward) =
            self.head.value.forward(head_placeholders, head_variables);
        let (tail_outputs, tail_backwards) = self.tail.forward(tail_placeholders, tail_variables);
        (
            HCons {
                head: field![Name, head_output],
                tail: tail_outputs,
            },
            HCons {
                head: field![Name, head_backward],
                tail: tail_backwards,
            },
        )
    }

    fn initial_variables(&self) -> Self::Variables {
        let head_variables = self.head.value.initial_variables();
        let tail_variables = self.tail.initial_variables();
        head_variables.concat(tail_variables)
    }
}

// BackwardのLabelled HList
pub trait LabelledBackwards: Sized {
    type Outputs;
    type Variables;

    fn backward(self, douts: Self::Outputs) -> Self::Variables;
}

impl LabelledBackwards for HNil {
    type Outputs = HNil;
    type Variables = HNil;

    fn backward(self, _douts: Self::Outputs) -> Self::Variables {
        HNil
    }
}

impl<Name, Type: Backward, Tail: LabelledBackwards> LabelledBackwards
    for HCons<Field<Name, Type>, Tail>
where
    Type::Variables: ConcatIsInverseSplit<Tail::Variables>,
{
    type Outputs = HCons<Field<Name, Type::Output>, Tail::Outputs>;
    type Variables = <Type::Variables as Concat<Tail::Variables>>::Output;

    fn backward(self, douts: Self::Outputs) -> Self::Variables {
        let d1 = self.head.value.backward(douts.head.value);
        let d2 = self.tail.backward(douts.tail);

        d1.concat(d2)
    }
}

pub trait Backward {
    type Output;
    type Variables;

    fn backward(self, douts: Self::Output) -> Self::Variables;
}

pub trait Layer {
    type Output;
    type Backward: Backward<Output = Self::Output, Variables = Self::Variables>;
    type Placeholders;
    type Variables;

    fn forward(
        &self,
        placeholders: Self::Placeholders,
        variables: Self::Variables,
    ) -> (Self::Output, Self::Backward);

    fn initial_variables(&self) -> Self::Variables;
}

// 親レイヤーと未接続のレイヤー
pub trait UnconnectedLayer: Sized {
    // 入力の名前と型のLabelled HList
    type Inputs;

    // このレイヤーで必要なプレースフォルダの名前と型のLabelled HList
    type Placeholders;

    // このレイヤーで必要な変数の名前と型のLabelled HList
    type Variables;

    // 出力の型
    type Output;

    type Backward: UnconnectedBackward<
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
    ) -> (Self::Output, Self::Backward);

    fn initial_variables(&self) -> Self::Variables;
}

// 親レイヤーと未接続のBackward
pub trait UnconnectedBackward {
    type Inputs;
    type Output;
    type Variables;

    fn backward(self, douts: Self::Output) -> (Self::Inputs, Self::Variables);
}

pub struct LayerAdapter<I, L> {
    input_layers: I,
    layer: L,
}

impl<I: LabelledLayers, L: UnconnectedLayer<Inputs = I::Outputs>> Layer for LayerAdapter<I, L>
where
    I::Placeholders: ConcatIsInverseSplit<L::Placeholders>,
    I::Variables: ConcatIsInverseSplit<L::Variables>,
    BackwardAdapter<I::Backwards, L::Backward>:
        Backward<Output = L::Output, Variables = <I::Variables as Concat<L::Variables>>::Output>,
{
    type Output = L::Output;
    type Backward = BackwardAdapter<I::Backwards, L::Backward>;
    type Placeholders = <I::Placeholders as Concat<L::Placeholders>>::Output;
    type Variables = <I::Variables as Concat<L::Variables>>::Output;

    fn forward(
        &self,
        placeholders: Self::Placeholders,
        variables: Self::Variables,
    ) -> (Self::Output, Self::Backward) {
        let (input_placeholders, layer_placeholders) = placeholders.split();
        let (input_variables, layer_variables) = variables.split();
        let (inputs, input_backwards) = self
            .input_layers
            .forward(input_placeholders, input_variables);
        let (output, backward) = self
            .layer
            .forward(layer_placeholders, layer_variables, inputs);

        (
            output,
            BackwardAdapter {
                input_backwards,
                backward,
            },
        )
    }

    fn initial_variables(&self) -> Self::Variables {
        let input_variables = self.input_layers.initial_variables();
        let layer_variables = self.layer.initial_variables();
        input_variables.concat(layer_variables)
    }
}

pub struct BackwardAdapter<I, O> {
    input_backwards: I,
    backward: O,
}

impl<I: LabelledBackwards, O: UnconnectedBackward<Inputs = I::Outputs>> Backward
    for BackwardAdapter<I, O>
where
    I::Variables: ConcatIsInverseSplit<O::Variables>,
{
    type Output = O::Output;
    type Variables = <I::Variables as Concat<O::Variables>>::Output;

    fn backward(self, dout: Self::Output) -> Self::Variables {
        let (grads, d1) = self.backward.backward(dout);
        let d2 = self.input_backwards.backward(grads);

        d2.concat(d1)
    }
}

impl<T: Layer> Layer for &T {
    type Output = T::Output;
    type Backward = T::Backward;
    type Placeholders = T::Placeholders;
    type Variables = T::Variables;

    fn forward(
        &self,
        placeholders: Self::Placeholders,
        variables: Self::Variables,
    ) -> (Self::Output, Self::Backward) {
        (*self).forward(placeholders, variables)
    }

    fn initial_variables(&self) -> Self::Variables {
        (*self).initial_variables()
    }
}
