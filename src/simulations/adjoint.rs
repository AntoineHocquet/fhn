use crate::models::neuron::{FhnParameters, NeuronState};

/// One adjoint trajectory corresponding to a neuron
pub type AdjointTrajectory = Vec<[f64; 3]>;

/// Solves the adjoint equation backward in time
pub fn compute_adjoint(
    sim: &Vec<Vec<NeuronState>>, // forward solution: L x M
    y_target: [f64; 3],          // desired final state
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
            let dp1 = dt * (x.c * p_next[0] - x.c * x.b * p_next[1]);
            let dp2 = 0.0; // no running cost on y

            adjoints[i][r][0] = p_next[0] + dp0;
            adjoints[i][r][1] = p_next[1] + dp1;
            adjoints[i][r][2] = p_next[2] + dp2;
        }
    }

    adjoints
}
