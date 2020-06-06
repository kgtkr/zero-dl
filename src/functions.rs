pub fn step(x: f32) -> f32 {
    if x > 0.0 {
        1.0
    } else {
        0.0
    }
}

pub fn relu(x: f32) -> f32 {
    x.max(0.)
}

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + std::f32::consts::E.powf(-x))
}

pub fn identity(x: f32) -> f32 {
    x
}

#[test]
fn plot_activation_function() {
    use crate::plot::function_plot;

    function_plot("step", step);
    function_plot("sigmoid", sigmoid);
    function_plot("relu", relu);
}
