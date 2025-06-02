// src/models/neuron.rs

#[derive(Debug,Clone, Copy)]
pub struct NeuronState {
    pub v: f64, // Membrane potential
    pub w: f64, // Recovery variable
    pub y: f64, // Synaptic gate
}
 
/// We now define the Parameters Struct, which holds all constants from the FHN model.
/// This struct will be passed to the simulation functions.
/// 
/// The significance of each parameter is roughly described here (see the AMOP paper for details):
/// a,b,c,I,sigex = parameter of non coupled FitzHugh-Nagumo model
/// J, sigJ = synaptical weights
/// Iext = external current (i.e. the control)
/// Vrev = reversal potential for synaptic gates
/// ar,ad = transition rates for opening and closing of synaptic gates
/// lambda, VT =parameters for sigmoid correlation between potential and release of neurotransmitter
/// Tmax = time constant for neurotransmitter release
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
    pub J: f64,
    pub Iext: f64,
}

/// The next step is to implement the drift function:
/// Add a method to compute the deterministic drift of a neuron's state. 
/// The mean_y variable is the average from the other neurons (mean field term)
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
