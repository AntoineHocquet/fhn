// src/simulations/adjoint.rs

use crate::models::neuron::{FhnParameters, NeuronState};
use plotters::prelude::*;

/// One adjoint trajectory corresponding to a neuron
pub type AdjointTrajectory = Vec<[f64; 3]>;

/// Solves the adjoint equation backward in time
pub fn compute_adjoint(
    sim: &Vec<Vec<NeuronState>>,
    params: &FhnParameters,
    y_target: [f64; 3],
    gamma: f64,
    cT: f64,
    dt: f64,
) -> Vec<AdjointTrajectory> {
    let L = sim.len();
    let M = sim[0].len();

    let mut adjoints = vec![vec![[0.0; 3]; M]; L];

    // Terminal condition
    for i in 0..L {
        let x_final = sim[i][M - 1];
        adjoints[i][M - 1][0] = 2.0 * cT * gamma * (x_final.v - y_target[0]);
        // [1] and [2] remain 0
    }

    // Backward loop
    for r in (0..M - 1).rev() {
        // Compute mean field quantities at step r+1
        let mean_v: f64 = sim.iter().map(|traj| traj[r + 1].v).sum::<f64>() / (L as f64);
        let mean_y: f64 = sim.iter().map(|traj| traj[r + 1].y).sum::<f64>() / (L as f64);

        for i in 0..L {
            let x = sim[i][r];
            let p_next = adjoints[i][r + 1];

            // Backward Euler step for p(t)
            let dp0 = dt * (
                (1.0 - x.v.powi(2)) * p_next[0]
                - p_next[1]
                - 2.0 * gamma * (x.v - mean_v)
            );
            let dp1 = dt * (params.c * p_next[0] - params.c * params.b * p_next[1]);
            let dp2 = 0.0; // no running cost on y

            adjoints[i][r][0] = p_next[0] + dp0;
            adjoints[i][r][1] = p_next[1] + dp1;
            adjoints[i][r][2] = p_next[2] + dp2;
        }
    }

    adjoints
}


pub fn plot_adjoint_trajectories(
    adj: &Vec<Vec<[f64; 3]>>,
    dt: f64,
    filename: &str,
    count: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let m = adj[0].len();
    let l = adj.len();
    let count = count.min(l);

    let root = BitMapBackend::new(filename, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let t_max = m as f64 * dt;

    let p_min = adj.iter()
        .take(count)
        .flat_map(|traj| traj.iter().map(|p| p[0]))
        .fold(f64::INFINITY, f64::min);

    let p_max = adj.iter()
        .take(count)
        .flat_map(|traj| traj.iter().map(|p| p[0]))
        .fold(f64::NEG_INFINITY, f64::max);

    let mut chart = ChartBuilder::on(&root)
        .caption("Adjoint State p⁰(t) for Select Neurons", ("sans-serif", 30))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(0.0..t_max, p_min..p_max)?;

    chart.configure_mesh().draw()?;

    let colors = [
        &RED, &BLUE, &GREEN, &CYAN, &MAGENTA, &BLACK, &YELLOW,
    ];

    for i in 0..count {
        let color = colors[i % colors.len()];
        chart
            .draw_series(LineSeries::new(
                adj[i].iter().enumerate().map(|(t, p)| (t as f64 * dt, p[0])),
                color,
            ))?
            .label(format!("Neuron {}", i))
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], *color));
    }

    chart.configure_series_labels().border_style(&BLACK).draw()?;
    Ok(())
}
