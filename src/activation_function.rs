pub trait ActivationFunction {
    fn run(&self, x: f64) -> f64;
}

#[derive(Debug, Clone)]
pub struct Step(f64);
impl ActivationFunction for Step {
    fn run(&self, x: f64) -> f64 {
        if x >= self.0 {
            1.0
        } else {
            0.0
        }
    }
}

impl Step {
    pub fn new(a: f64) -> Step {
        Step(a)
    }
}
