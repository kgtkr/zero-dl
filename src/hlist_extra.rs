use frunk::indices::{Here, There};
use frunk::{HCons, HNil};

pub trait IndexAccess<I> {
    type Output;

    fn get(&self) -> &Self::Output;
    fn get_mut(&mut self) -> &mut Self::Output;
}

impl<H, T> IndexAccess<Here> for HCons<H, T> {
    type Output = H;

    fn get(&self) -> &Self::Output {
        &self.head
    }

    fn get_mut(&mut self) -> &mut Self::Output {
        &mut self.head
    }
}

impl<H, T: IndexAccess<PrevI>, PrevI> IndexAccess<There<PrevI>> for HCons<H, T> {
    type Output = <T as IndexAccess<PrevI>>::Output;

    fn get(&self) -> &Self::Output {
        self.tail.get()
    }

    fn get_mut(&mut self) -> &mut Self::Output {
        self.tail.get_mut()
    }
}

pub trait AList<A> {}

impl<A> AList<A> for HNil {}

impl<H, T: AList<H>> AList<H> for HCons<H, T> {}
