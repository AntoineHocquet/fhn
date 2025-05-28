// src/simulations/forward.rs

use serde::Serialize;
use rand_distr::{Distribution, Normal};
use crate::models::neuron::{NeuronState, FhnParameters};

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
