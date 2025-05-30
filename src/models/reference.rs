// src/models/reference.rs

use plotters::prelude::*;


pub fn reference_profile(t: f64, t_final: f64) -> f64 {
    let sigma: f64 = 5.0;
    let center: f64 = t_final / 2.0;
    1.5 * (-((t - center).powi(2)) / (2.0 * sigma.powi(2))).exp()
}


pub fn plot_reference_profile(
    t_final: f64,
    dt: f64,
    filename: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(filename, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let t_values: Vec<f64> = (0..=(t_final / dt) as usize)
        .map(|i| i as f64 * dt)
        .collect();

    let v_values: Vec<f64> = t_values
        .iter()
        .map(|&t| reference_profile(t, t_final))
        .collect();

    let v_min = v_values.iter().cloned().fold(f64::INFINITY, f64::min);
    let v_max = v_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let mut chart = ChartBuilder::on(&root)
        .caption("Reference Profile v_ref(t)", ("sans-serif", 30))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(0.0..t_final, v_min..v_max)?;

    chart.configure_mesh().draw()?;

    chart.draw_series(LineSeries::new(
        t_values.iter().zip(v_values.iter()).map(|(&t, &v)| (t, v)),
        &GREEN,
    ))?
    .label("v_ref(t)")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &GREEN));

    chart.configure_series_labels().border_style(&BLACK).draw()?;
    Ok(())
}
