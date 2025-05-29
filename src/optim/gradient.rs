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

