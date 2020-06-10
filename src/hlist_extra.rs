use frunk::hlist::{HCons, HList, HNil};

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
