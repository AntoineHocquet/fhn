// src/bin/main.rs

use clap::{Parser, Subcommand};
use fhn::models::neuron::{FhnParameters, NeuronState};
use fhn::simulations::forward::{simulate_fhn_population, plot_local_field_potential, plot_individual_neurons, plot_average_potential, save_simulation_to_csv, simulate_with_control};
use fhn::simulations::adjoint::{compute_adjoint, plot_adjoint_trajectories};
use fhn::optim::gradient::{evaluate_cost, compute_control_gradient, gradient_step, plot_cost_trace, plot_control};
use fhn::models::reference::plot_reference_profile;
use std::fs::File;
use std::io::Write;






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
    Optimize {
        #[arg(short, long, default_value_t = 100)]
        neurons: usize,
        #[arg(short, long, default_value_t = 1000)]
        steps: usize,
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
            
            // Initial condition used for the simulations in the paper
            let initial = NeuronState {
                v: -0.8275021695916729,
                w: 0.1391607698173808,
                y: 0.589165868968053
            };
            
            // Simulation parameters used for the simulations in the paper
            let params = FhnParameters {
                a: 0.7,
                b: 0.8,
                c: 0.08,
                // excitatory synapses since rev. potential higher then rest. potential:
                Vrev: 1.2,
                // fast excitatory conductance i.e. fast activation and deactivation:
                ar: 1.0,
                ad: 0.3,
                Tmax: 1.0,
                lambda: 0.1,
                // threshold for presynaptic neuron for opening of synaptic gates to postsynaptic neuron:
                VT: 2.0,
                // Coupling strength:
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
            // Plotting the average value of the potential
            match fhn::simulations::forward::plot_local_field_potential(&sim, *dt, "figures/lfp.png") {
                Ok(_) => println!("✅ Plot saved to figures/lfp.png"),
                Err(e) => eprintln!("❌ Plotting error: {}", e),
            }
            // Plotting individual trajectories
            match fhn::simulations::forward::plot_individual_neurons(&sim, *dt, "figures/neurons.png", 5) {
                Ok(_) => println!("✅ Individual neuron plot saved to figures/neurons.png"),
                Err(e) => eprintln!("❌ Neuron plot error: {}", e),
            }
        }
        Commands::Optimize { neurons, steps, dt } => {
            let initial = NeuronState {
                v: -0.8275021695916729,
                w: 0.1391607698173808,
                y: 0.589165868968053
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

            let mut control = vec![0.5; *steps];
            let mut cost_trace = Vec::new();
            let max_iters = 20;
            let step_size = 0.005;
            let lambda2 = 0.01;
            let gamma = 1.0;
            let c_t = 1.0;
            let y_target = [0.0, 0.0, 0.0];
            
            // Introduce adjoint profile
            let mut last_adj = vec![];

            // Optimization loop
            for iter in 0..max_iters {
                let sim = simulate_fhn_population(*neurons, *steps, *dt, &params, sigma_ext, initial);
                let cost = evaluate_cost(&sim, &control, y_target, gamma, lambda2, c_t, *dt);
                cost_trace.push(cost);

                let adj = compute_adjoint(&sim, &params, y_target, gamma, c_t, *dt);
                last_adj = adj.clone(); // save last adjoint for plotting later

                let grad = compute_control_gradient(&adj, &control, lambda2, *dt);
                control = gradient_step(&control, &grad, step_size);

                println!("Iter {:>2}: J(α) = {:.6}", iter, cost);
            }

            // save optimal control to csv
            let mut file = File::create("output/control.csv").expect("Failed to create control.csv");
            writeln!(file, "t,alpha").unwrap();
            for (i, alpha) in control.iter().enumerate() {
                writeln!(file, "{:.4},{}", i as f64 * dt, alpha).unwrap();
            }
            println!("✅ Saved final control to output/control.csv");


            // save cost to csv
            let mut file = File::create("output/cost.csv").expect("Failed to create cost.csv");
            writeln!(file, "iter,cost").unwrap();
            for (i, j) in cost_trace.iter().enumerate() {
                writeln!(file, "{},{}", i, j).unwrap();
            }
            println!("✅ Saved cost trace to output/cost.csv");
            
            // Plot control
            match plot_control(&control, *dt, "figures/control.png") {
                Ok(_) => println!("✅ Control plot saved to figures/control.png"),
                Err(e) => eprintln!("❌ Failed to plot control: {}", e),
            }

            // Plot cost
            match plot_cost_trace(&cost_trace, "figures/cost.png") {
                Ok(_) => println!("✅ Cost plot saved to figures/cost.png"),
                Err(e) => eprintln!("❌ Failed to plot cost: {}", e),
            }

            // Plot adjoint
            match plot_adjoint_trajectories(&last_adj, *dt, "figures/adjoint.png", 5) {
                Ok(_) => println!("✅ Adjoint plot saved to figures/adjoint.png"),
                Err(e) => eprintln!("❌ Failed to plot adjoint: {}", e),
            }

            // Plot reference profile
            match plot_reference_profile(*dt * *steps as f64, *dt, "figures/reference.png") {
                Ok(_) => println!("✅ Reference profile plot saved to figures/reference.png"),
                Err(e) => eprintln!("❌ Failed to plot reference profile: {}", e),
            }
            
            // Plot controlled profile
            let final_sim = simulate_with_control(*neurons, *steps, *dt, &params, &control, initial);
            match plot_average_potential(&final_sim, *dt, "figures/potential.png") {
                Ok(_) => println!("✅ Average potential plot saved to figures/potential.png"),
                Err(e) => eprintln!("❌ Failed to plot potential: {}", e),
            }

        }
    }
}
