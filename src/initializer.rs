pub trait Initializer {
    type Output;
    fn initial_value(&self) -> Self::Output;
}
