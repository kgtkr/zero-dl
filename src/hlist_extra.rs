use frunk::hlist::{HCons, HList, HNil};
use frunk::indices::{Here, There};
use frunk::labelled::Field;

#[macro_export]
macro_rules! record {
    ( $( $key: ident: $value: expr ),* ) => {
        {
            use frunk_labelled_proc_macro::label;

            hlist![
                $( field![
                    label!($key),
                    $value
                ]),*
            ]
        }
    };
}

#[macro_export]
macro_rules! record_dest {
    ( { $( $ident: ident: $key: ident ),* } = $record: expr ) => {
        use frunk::labelled::ByNameFieldPlucker;
        use frunk::labelled::Field;

        let record = $record;
        $(
            let (Field { value: $ident, .. }, record) =
                ByNameFieldPlucker::<label!($key), _>::pluck_by_name(record);
        )*
        let HNil = record;
    };
}

#[macro_export]
macro_rules! Record {
    ( $( $key: ident: $type: ty ),* ) => {
        Hlist![
            $( Field<label!($key), $type> ),*
        ]
    };
}

pub trait ConcatAndSplit<RHS>: Sized {
    type Out;

    fn concat(self, rhs: RHS) -> Self::Out;
    fn split(output: Self::Out) -> (Self, RHS);
}

impl<RHS> ConcatAndSplit<RHS> for HNil
where
    RHS: HList,
{
    type Out = RHS;

    fn concat(self, rhs: RHS) -> Self::Out {
        rhs
    }

    fn split(output: Self::Out) -> (Self, RHS) {
        (HNil, output)
    }
}

impl<H, T, RHS> ConcatAndSplit<RHS> for HCons<H, T>
where
    T: ConcatAndSplit<RHS>,
    RHS: HList,
{
    type Out = HCons<H, <T as ConcatAndSplit<RHS>>::Out>;

    fn concat(self, rhs: RHS) -> Self::Out {
        HCons {
            head: self.head,
            tail: self.tail.concat(rhs),
        }
    }

    fn split(output: Self::Out) -> (Self, RHS) {
        let (a, b) = ConcatAndSplit::split(output.tail);
        (
            HCons {
                head: output.head,
                tail: a,
            },
            b,
        )
    }
}

pub trait Has<TargetKey, Index> {
    type TargetValue;

    fn get(&self) -> &Self::TargetValue;
}

impl<K, V, Tail> Has<K, Here> for HCons<Field<K, V>, Tail> {
    type TargetValue = V;

    fn get(&self) -> &Self::TargetValue {
        &self.head.value
    }
}

impl<Head, Tail, K, TailIndex> Has<K, There<TailIndex>> for HCons<Head, Tail>
where
    Tail: Has<K, TailIndex>,
{
    type TargetValue = <Tail as Has<K, TailIndex>>::TargetValue;

    fn get(&self) -> &Self::TargetValue {
        self.tail.get()
    }
}
