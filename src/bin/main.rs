use fhn::models::neuron::{NeuronState, FhnParameters};
use fhn::simulations::forward::simulate_fhn_population;

fn main() {
    let initial = NeuronState { v: -1.0, w: 0.0, y: 0.5 };

    let params = FhnParameters {
        a: 0.7,
        b: 0.8,
        c: 0.08,
        Vrev: 1.2,
        ar: 1.0,
        ad: 0.3,
        Tmax: 1.0,
        lambda: 0.1,
        VT: 2.0,
        J: 0.46,
        Iext: 0.5,
    };

    let L = 10;
    let M = 1000;
    let dt = 0.1;
    let sigma_ext = 0.04;

    let sim = simulate_fhn_population(L, M, dt, &params, sigma_ext, initial);

    println!("v(t) of first neuron:");
    for (i, s) in sim[0].iter().enumerate().step_by(100) {
        println!("t = {:.1}, v = {:.3}", i as f64 * dt, s.v);
    }
}
