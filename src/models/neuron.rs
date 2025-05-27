// src/models/neuron.rs
use ndarray::Array1;

#[derive(Debug,Clone, Copy)]
pub struct NeuronState {
    pub v: f64, // Membrane potential
    pub w: f64, // Recovery variable
    pub y: f64, // Synaptic gate
}
 
/// Define the Parameters Struct
/// This holds all constants from the FHN model and synapse dynamics 

#[derive(Debug, Clone, Copy)]
pub struct FhnParameters {
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub Vrev: f64,
    pub ar: f64,
    pub ad: f64,
    pub Tmax: f64,
    pub lambda: f64,
    pub VT: f64,
    pub J: f64,      // coupling strength
    pub Iext: f64,   // external current
}
/// Implement the Drift function:
/// Add a method to compute the deterministic drift 
/// of a neuron's state. The mean_y variable is the average from the
///  other neurons (mean field term)

impl NeuronState {
    pub fn drift(&self, mean_y: f64, params: &FhnParameters) -> NeuronState {
        let v = self.v;
        let w = self.w;
        let y = self.y;

        // External current + mean-field coupling
        let input = params.Iext - params.J * (v - params.Vrev) * mean_y;

        let dv = v - (v.powi(3) / 3.0) - w + input;
        let dw = params.c * (v + params.a - params.b * w);
        let s = params.Tmax / (1.0 + (-params.lambda * (v - params.VT)).exp());
        let dy = params.ar * s * (1.0 - y) - params.ad * y;

        NeuronState { v: dv, w: dw, y: dy }
    }
}
