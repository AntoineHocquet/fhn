# fhn — Optimal Control of FitzHugh–Nagumo Neuron Networks in Rust

> A fast and modular Rust reimplementation of the core numerical experiment from  
> *"Optimal Control of Mean Field Equations with Monotone Coefficients and Applications in Neuroscience"*  
> by Antoine Hocquet and Alexander Vogler, *Applied Mathematics & Optimization*, 2021  
> [https://doi.org/10.1007/s00245-021-09816-1](https://doi.org/10.1007/s00245-021-09816-1)

---

## 🧠 Overview

This repository is part of a broader research and engineering effort to bridge optimal control theory, stochastic PDEs, and modern AI tools for neuroscience-inspired models.

More precisely, the simulation and control of large networks of FitzHugh–Nagumo neurons is implemented, where:
- Each neuron is subject to stochastic noise and mean-field interactions,
- A **common deterministic control** drives the population toward a desired reference trajectory,
- The **cost functional** measures the deviation from a target profile using a quadratic penalty.

We use **gradient descent** on this cost, via the adjoint method, to find the optimal control.

---

## 🔧 Technical Highlights

- Language: **Rust** 🦀 for performance and safety.
- Modular architecture with CLI (`clap`) and plotting (`plotters`).
- Reimplementation of the numerical algorithm from the AMOP 2021 paper:contentReference[oaicite:0]{index=0}.
- No Python or external dependencies — 100% native.

---

## 📊 Core Features

- ✅ Forward simulation via Euler–Maruyama scheme
- ✅ Adjoint equation solved backward in time
- ✅ Cost and gradient computation
- ✅ Gradient descent optimization
- ✅ All figures reproduced:
  - Optimal control \( \alpha(t) \)
  - Adjoint trajectories \( p^{(0)}_i(t) \)
  - Reference profile \( v_{\text{ref}}(t) \)
  - Controlled population potential \( \bar{v}(t) \)

---

## 📁 Folder Structure

.
├── src/
│ ├── bin/ # CLI entry point
│ ├── models/ # Neuron model, reference profiles
│ ├── simulations/ # Forward and adjoint simulation
│ └── optim/ # Cost, gradient, descent
├── figures/ # PNG plots (control, adjoint, etc.)
├── output/ # CSV data (control.csv, cost.csv)
├── Cargo.toml
└── README.md


---

## ▶️ How to Use

### 1. Build the project

```bash
cargo build --release
```

### 2. Run a simulation

```bash
cargo run --bin main -- simulate
```

### 3. Run optimization loop (with all plots)

```bash
cargo run --bin main -- optimize --neurons 100 --steps 1000 --dt 0.1
```

All results are written to figures/ and output/.

## Acknowledgements
The original Python codebase was developed by **Alexander Vogler** (GitHub: `alexander19a`).

The current repository is maintained and extended by **Antoine Hocquet**. It is part of an effort to showcase applied mathematics and numerical control in fast, modern systems languages.

The whole project was supported by the research collaboration between TU Berlin and the DFG-funded CRC 910.

## License
This code is released under the MIT License.
The original paper and its figures are © the authors and Springer, 2021.

---

