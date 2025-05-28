// src/bin/main.rs

use clap::{Parser, Subcommand};
use fhn::models::neuron::{FhnParameters, NeuronState};
use fhn::simulations::forward::simulate_fhn_population;
use fhn::simulations::forward::save_simulation_to_csv;

/// CLI for the FitzHugh–Nagumo control project
#[derive(Parser)]
#[command(name = "fhn")]
#[command(about = "Simulate and control FitzHugh-Nagumo neuron networks", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run a forward simulation of the neuron network
    Simulate {
        /// Number of neurons (L)
        #[arg(short, long, default_value_t = 10)]
        neurons: usize,

        /// Number of time steps (M)
        #[arg(short, long, default_value_t = 1000)]
        steps: usize,

        /// Time step size (dt)
        #[arg(short, long, default_value_t = 0.1)]
        dt: f64,
    },
}

fn main() {
    let cli = Cli::parse();

    match &cli.command {
        Commands::Simulate {
            neurons,
            steps,
            dt,
        } => {
            println!("Running simulation with L = {neurons}, M = {steps}, dt = {dt}");

            let initial = NeuronState {
                v: -1.0,
                w: 0.0,
                y: 0.5,
            };

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

            let sigma_ext = 0.04;

            let sim = simulate_fhn_population(*neurons, *steps, *dt, &params, sigma_ext, initial);
            // Printing some values
            println!("First neuron's v(t):");
            for (i, state) in sim[0].iter().enumerate().step_by(steps / 10) {
                println!("t = {:.1}, v = {:.3}", i as f64 * dt, state.v);
            }
            // Saving to csv
            match fhn::simulations::forward::save_simulation_to_csv(&sim, *dt, "output/simulation.csv") {
                Ok(_) => println!("✅ Saved to output/simulation.csv"),
                Err(e) => eprintln!("❌ Failed to save CSV: {}", e),
            }

        }
    }
}
