// src/simulations/forward.rs

use plotters::style::full_palette::PURPLE;
use serde::Serialize;
use rand_distr::{Distribution, Normal};
use crate::models::neuron::{NeuronState, FhnParameters};
use plotters::prelude::*;


/// One row of output (for 1 neuron at 1 time step)
#[derive(Serialize)]
pub struct SimulationRow {
    pub neuron_id: usize,
    pub time: f64,
    pub v: f64,
    pub w: f64,
    pub y: f64,
}

pub fn save_simulation_to_csv(
    sim: &Vec<Vec<NeuronState>>,
    dt: f64,
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut wtr = csv::Writer::from_path(path)?;

    for (neuron_id, trajectory) in sim.iter().enumerate() {
        for (step, state) in trajectory.iter().enumerate() {
            let row = SimulationRow {
                neuron_id,
                time: step as f64 * dt,
                v: state.v,
                w: state.w,
                y: state.y,
            };
            wtr.serialize(row)?;
        }
    }

    wtr.flush()?;
    Ok(())
}


/// Simulates L neurons over M time steps of size dt.
/// Returns: Vec of trajectories, each of length M.
pub fn simulate_fhn_population(
    L: usize,
    M: usize,
    dt: f64,
    params: &FhnParameters,
    sigma_ext: f64,
    initial: NeuronState,
) -> Vec<Vec<NeuronState>> {
    let sqrt_dt = dt.sqrt();
    let normal = Normal::new(0.0, 1.0).unwrap();

    // Initialize all neurons with the same state
    let mut trajectories = vec![vec![initial; M]; L];

    for t in 1..M {
        // Compute mean gate value across population
        let mean_y: f64 = trajectories.iter()
            .map(|traj| traj[t - 1].y)
            .sum::<f64>() / (L as f64);

        // Update each neuron
        for traj in trajectories.iter_mut() {
            let prev = *traj.get(t - 1).unwrap(); // idiomatic way of cloning

            // Drift part
            let drift = prev.drift(mean_y, params);

            // Diffusion part (only on v)
            let noise = sigma_ext * sqrt_dt * normal.sample(&mut rand::thread_rng());

            traj[t] = NeuronState {
                v: prev.v + dt * drift.v + noise,
                w: prev.w + dt * drift.w,
                y: prev.y + dt * drift.y,
            };
        }
    }

    trajectories
}


/// Plots the average membrane potential (v) over time
pub fn plot_local_field_potential(
    sim: &Vec<Vec<NeuronState>>,
    dt: f64,
    filename: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let M = sim[0].len();
    let L = sim.len();

    // Compute mean voltage at each time step
    let mean_v: Vec<f64> = (0..M)
        .map(|t| {
            sim.iter().map(|traj| traj[t].v).sum::<f64>() / (L as f64)
        })
        .collect();

    let root = BitMapBackend::new(filename, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let time_max = M as f64 * dt;
    let v_min = mean_v.iter().cloned().fold(f64::INFINITY, f64::min);
    let v_max = mean_v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let mut chart = ChartBuilder::on(&root)
        .caption("Local Field Potential", ("sans-serif", 30))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0.0..time_max, v_min..v_max)?;

    chart.configure_mesh().draw()?;

    chart.draw_series(LineSeries::new(
        mean_v.iter().enumerate().map(|(i, v)| (i as f64 * dt, *v)),
        &RED,
    ))?
    .label("mean(v)")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart.configure_series_labels().border_style(&BLACK).draw()?;

    Ok(())
}


/// Plots the v(t) trajectory of a few individual neurons
pub fn plot_individual_neurons(
    sim: &Vec<Vec<NeuronState>>,
    dt: f64,
    filename: &str,
    count: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let M = sim[0].len();
    let L = sim.len();
    let count = count.min(L); // clamp to number of available neurons

    let root = BitMapBackend::new(filename, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let time_max = M as f64 * dt;

    let v_min = sim.iter()
        .take(count)
        .flat_map(|traj| traj.iter().map(|s| s.v))
        .fold(f64::INFINITY, f64::min);
    
    let v_max = sim.iter()
        .take(count)
        .flat_map(|traj| traj.iter().map(|s| s.v))
        .fold(f64::NEG_INFINITY, f64::max);

    let mut chart = ChartBuilder::on(&root)
        .caption("Neuron Voltages", ("sans-serif", 30))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0.0..time_max, v_min..v_max)?;

    chart.configure_mesh().draw()?;

    let colors = [
        &RED, &BLUE, &GREEN, &MAGENTA, &CYAN, &BLACK, &YELLOW, &PURPLE,
    ];

    for i in 0..count {
        let color = colors[i % colors.len()];
        chart
            .draw_series(LineSeries::new(
                sim[i].iter().enumerate().map(|(t, s)| (t as f64 * dt, s.v)),
                color,
            ))?
            .label(format!("Neuron {}", i))
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color));
    }

    chart.configure_series_labels().border_style(&BLACK).draw()?;
    Ok(())
}
