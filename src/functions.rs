pub fn step(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else {
        0.0
    }
}

pub fn relu(x: f64) -> f64 {
    x.max(0.)
}

pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + std::f64::consts::E.powf(-x))
}

pub fn identity(x: f64) -> f64 {
    x
}

#[test]
fn plot_activation_function() {
    use crate::plot::function_plot;

    function_plot("step", step);
    function_plot("sigmoid", sigmoid);
    function_plot("relu", relu);
}
