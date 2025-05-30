// src/optim/gradient.rs

use crate::models::neuron::NeuronState;


/// Computes the gradient of the cost functional
pub fn compute_control_gradient(
    adjoints: &Vec<Vec<[f64; 3]>>, // L x M
    control: &Vec<f64>,            // alpha(t)
    lambda2: f64,
    dt: f64,
) -> Vec<f64> {
    let L = adjoints.len();
    let M = adjoints[0].len();
    let mut gradient = vec![0.0; M];

    for t in 0..M {
        let mean_p: f64 = adjoints.iter().map(|p| p[t][0]).sum::<f64>() / (L as f64);
        gradient[t] = mean_p + 2.0 * lambda2 * control[t];
    }

    gradient
}


/// One gradient descent update: alpha_new = alpha - s * grad
pub fn gradient_step(
    control: &Vec<f64>,
    gradient: &Vec<f64>,
    step_size: f64,
) -> Vec<f64> {
    control
        .iter()
        .zip(gradient.iter())
        .map(|(a, g)| a - step_size * g)
        .collect()
}


/// Evaluate the cost functional J(alpha)
pub fn evaluate_cost(
    sim: &Vec<Vec<NeuronState>>,
    control: &Vec<f64>,
    y_target: [f64; 3],
    gamma: f64,
    lambda2: f64,
    c_t: f64,
    dt: f64,
) -> f64 {
    let l = sim.len();
    let m = sim[0].len();

    let mut running_cost = 0.0;
    let mut terminal_cost = 0.0;
    let mut control_cost = 0.0;

    for t in 0..m {
        // Mean v at time t
        let mean_v: f64 = sim.iter().map(|traj| traj[t].v).sum::<f64>() / (l as f64);
        let y_ref = y_target[0]; // reference voltage
        running_cost += gamma * (mean_v - y_ref).powi(2) * dt;
        control_cost += lambda2 * control[t].powi(2) * dt;
    }

    let mean_v_T: f64 = sim.iter().map(|traj| traj[m - 1].v).sum::<f64>() / (l as f64);
    terminal_cost = c_t * gamma * (mean_v_T - y_target[0]).powi(2);

    running_cost + terminal_cost + control_cost
}


use plotters::prelude::*;

/// Plot the cost vs iteration curve
pub fn plot_cost_trace(
    cost: &Vec<f64>,
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(path, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let iter_max = cost.len() as f64;
    let cost_min = cost.iter().cloned().fold(f64::INFINITY, f64::min);
    let cost_max = cost.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let mut chart = ChartBuilder::on(&root)
        .caption("Cost Evolution", ("sans-serif", 30))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(0.0..iter_max, cost_min..cost_max)?;

    chart.configure_mesh().draw()?;

    chart.draw_series(LineSeries::new(
        cost.iter().enumerate().map(|(i, j)| (i as f64, *j)),
        &BLUE,
    ))?
    .label("J(α)")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    chart.configure_series_labels().border_style(&BLACK).draw()?;
    Ok(())
}


/// Plot the final optimal control α(t)
pub fn plot_control(
    control: &Vec<f64>,
    dt: f64,
    filename: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    use plotters::prelude::*;

    let root = BitMapBackend::new(filename, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let t_max = dt * control.len() as f64;
    let alpha_min = control.iter().cloned().fold(f64::INFINITY, f64::min);
    let alpha_max = control.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let mut chart = ChartBuilder::on(&root)
        .caption("Optimal Control α(t)", ("sans-serif", 30))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(0.0..t_max, alpha_min..alpha_max)?;

    chart.configure_mesh().draw()?;

    chart.draw_series(LineSeries::new(
        control.iter().enumerate().map(|(i, a)| (i as f64 * dt, *a)),
        &RED,
    ))?
    .label("α(t)")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart.configure_series_labels().border_style(&BLACK).draw()?;
    Ok(())
}
