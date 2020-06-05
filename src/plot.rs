pub fn function_plot(name: &str, f: impl Fn(f64) -> f64) {
    use plotlib::page::Page;
    use plotlib::repr::Plot;
    use plotlib::style::LineStyle;
    use plotlib::view::ContinuousView;

    let s = Plot::from_function(f, -6., 6.).line_style(LineStyle::new());

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
