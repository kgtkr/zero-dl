use frunk::hlist::{HCons, HList, HNil};

pub trait ConcatAndSplit<RHS>: Sized {
    type Output;

    fn concat(self, rhs: RHS) -> Self::Output;
    fn split(output: Self::Output) -> (Self, RHS);
}

impl<RHS> ConcatAndSplit<RHS> for HNil
where
    RHS: HList,
{
    type Output = RHS;

    fn concat(self, rhs: RHS) -> Self::Output {
        rhs
    }

    fn split(output: Self::Output) -> (Self, RHS) {
        (HNil, output)
    }
}

impl<H, T, RHS> ConcatAndSplit<RHS> for HCons<H, T>
where
    T: ConcatAndSplit<RHS>,
    RHS: HList,
{
    type Output = HCons<H, <T as ConcatAndSplit<RHS>>::Output>;

    fn concat(self, rhs: RHS) -> Self::Output {
        HCons {
            head: self.head,
            tail: self.tail.concat(rhs),
        }
    }

    fn split(output: Self::Output) -> (Self, RHS) {
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
