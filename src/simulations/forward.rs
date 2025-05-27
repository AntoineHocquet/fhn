// src/simulations/forward.rs

use rand_distr::{Distribution, Normal};
use crate::models::neuron::{NeuronState, FhnParameters};

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
