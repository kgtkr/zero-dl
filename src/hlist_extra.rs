use frunk::hlist::{HCons, HList, HNil};
use frunk::indices::{Here, There};
use frunk::labelled::Field;
use frunk::traits::ToMut;

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

pub trait Split<Lhs>: Sized {
    type Rhs;

    fn split(self) -> (Lhs, Self::Rhs);
}

impl<T> Split<HNil> for T {
    type Rhs = T;

    fn split(self) -> (HNil, Self::Rhs) {
        (HNil, self)
    }
}

impl<Head, LTail, RTail> Split<HCons<Head, LTail>> for HCons<Head, RTail>
where
    RTail: Split<LTail>,
{
    type Rhs = <RTail as Split<LTail>>::Rhs;

    fn split(self) -> (HCons<Head, LTail>, Self::Rhs) {
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

pub trait Concat<Rhs>: Sized {
    type Output;

    fn concat(self, rhs: Rhs) -> Self::Output;
}

impl<Rhs> Concat<Rhs> for HNil
where
    Rhs: HList,
{
    type Output = Rhs;

    fn concat(self, rhs: Rhs) -> Rhs {
        rhs
    }
}

impl<H, T, Rhs> Concat<Rhs> for HCons<H, T>
where
    T: Concat<Rhs>,
    Rhs: HList,
{
    type Output = HCons<H, <T as Concat<Rhs>>::Output>;

    fn concat(self, rhs: Rhs) -> Self::Output {
        HCons {
            head: self.head,
            tail: self.tail.concat(rhs),
        }
    }
}

pub trait ConcatIsInverseSplit<R> = where
    Self: Concat<R>,
    <Self as Concat<R>>::Output: Split<Self, Rhs = R>;

pub trait ConcatToMutIsToMutConcat<'a, R> = where
    Self: ConcatIsInverseSplit<R>,
    Self: ToMut<'a>,
    R: ToMut<'a>,
    <Self as ToMut<'a>>::Output: ConcatIsInverseSplit<<R as ToMut<'a>>::Output>,
    <Self as Concat<R>>::Output: ToMut<
        'a,
        Output = <<Self as ToMut<'a>>::Output as Concat<<R as ToMut<'a>>::Output>>::Output,
    >;

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
