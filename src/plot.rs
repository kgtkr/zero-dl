pub fn function_plot(name: &str, f: impl Fn(f32) -> f32) {
    use plotlib::page::Page;
    use plotlib::repr::Plot;
    use plotlib::style::LineStyle;
    use plotlib::view::ContinuousView;

    let s = Plot::from_function(|x| f(x as f32) as f64, -6., 6.).line_style(LineStyle::new());

    let v = ContinuousView::new()
        .add(s)
        .x_range(-6., 6.)
        .y_range(-0.5, 1.5)
        .x_label("x")
        .y_label("y");
    Page::single(&v)
        .save(format!("plots/{}.svg", name))
        .unwrap();
}
