use frunk::hlist::{HCons, HList, HNil};
use frunk::indices::{Here, There};
use frunk::labelled::Field;
use std::ops::Add;

#[macro_export]
macro_rules! record {
    ( $( $key: ident: $value: expr ),* ) => {
        {
            frunk::hlist![
                $( frunk::field![
                    frunk_labelled_proc_macro::label!($key),
                    $value
                ]),*
            ]
        }
    };
}

#[macro_export]
macro_rules! record_dest {
    ( { } = $record: ident ) => {
        let frunk::hlist::HNil = $record;
    };

    ( { .. } = $record: ident ) => {
        let _ = $record;
    };

    ( { ..$ident: ident } = $record: ident ) => {
        let $ident = $record;
    };

    ( { $key: ident, $( $tt: tt )* } = $record: ident ) => {
        record_dest!({ $key: $key, $( $tt )* } = $record);
    };

    ( { $key: ident : $ident: ident, $( $tt: tt )* } = $record: ident ) => {
        let record = $record;
        let (frunk::labelled::Field { value: $ident, .. }, record) =
        frunk::labelled::ByNameFieldPlucker::<frunk_labelled_proc_macro::label!($key), _>::pluck_by_name(record);
        record_dest!({ $( $tt )* } = record);
    };
}

#[macro_export]
macro_rules! Record {
    ( $( $key: ident: $type: ty ),* ) => {
        frunk::Hlist![
            $( frunk::labelled::Field<frunk_labelled_proc_macro::label!($key), $type> ),*
        ]
    };
}

trait Concat<RHS> = Add<RHS>;

pub trait Split<LHS>: Sized {
    type RHS;

    fn split(self) -> (LHS, Self::RHS);
}

impl<T> Split<HNil> for T {
    type RHS = T;

    fn split(self) -> (HNil, Self::RHS) {
        (HNil, self)
    }
}

impl<Head, LTail, RTail> Split<HCons<Head, LTail>> for HCons<Head, RTail>
where
    RTail: Split<LTail>,
{
    type RHS = <RTail as Split<LTail>>::RHS;

    fn split(self) -> (HCons<Head, LTail>, Self::RHS) {
        let (l, r) = self.tail.split();
        (
            HCons {
                head: self.head,
                tail: l,
            },
            r,
        )
    }
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
