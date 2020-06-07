use frunk::field;
use frunk::hlist::Selector;
use frunk::hlist::{HCons, HList, HNil};
use frunk::indices::{Here, There};
use frunk::labelled::{chars, Field, LabelledGeneric, Transmogrifier};

pub trait Has<TargetKey, Index> {
    type TargetValue;
    type Remainder;
    fn get(&self) -> &Self::TargetValue;
}
impl<K, V, Tail> Has<K, Here> for HCons<Field<K, V>, Tail> {
    type TargetValue = V;
    type Remainder = Tail;
    fn get(&self) -> &Self::TargetValue {
        &self.head.value
    }
}
impl<Head, Tail, K, TailIndex> Has<K, There<TailIndex>> for HCons<Head, Tail>
where
    Tail: Has<K, TailIndex>,
{
    type TargetValue = <Tail as Has<K, TailIndex>>::TargetValue;
    type Remainder = HCons<Head, <Tail as Has<K, TailIndex>>::Remainder>;
    fn get(&self) -> &Self::TargetValue {
        <Tail as Has<K, TailIndex>>::get(&self.tail)
    }
}

pub trait BorrowSub<Target> {
    fn borrow_sub(&self) -> Target;
}
