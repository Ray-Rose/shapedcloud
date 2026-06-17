# shapedcloud — SCvx powered-descent trajectory solver

[![CI](https://github.com/Ray-Rose/shapedcloud/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Ray-Rose/shapedcloud/actions/workflows/ci.yml)
[![MSRV](https://img.shields.io/badge/MSRV-1.94-blue.svg)](Cargo.toml)
[![License](https://img.shields.io/badge/license-Apache--2.0%20OR%20MIT-blue.svg)](#license)

A **Successive-Convexification (SCvx)** trajectory-optimization solver for
**3-DoF powered descent** (rocket / lander guidance), written as **flight-grade,
`no_std` Rust**. The engineering discipline is the point: static memory, no heap,
no panics, no `unsafe` outside the C-FFI boundary, **bounded worst-case execution
time**, and **bit-deterministic** results (via `libm` transcendentals).

> **Status: research-grade, flight-*shaped*.** It demonstrates the disciplines a
> flight build needs and is verified end-to-end on representative problems — it is
> **not** certified flight software. See `HANDOFF.md` for the full, candid
> engineering log (what works, what's measured, and the open research frontier).

## What it does

Given an initial state and a soft-landing target, it finds a fuel-optimal thrust
trajectory respecting nonlinear 3-DoF dynamics (aerodynamic drag, log-mass
parameterization, optional free-final-time) and the operational cones: thrust
magnitude bounds, pointing angle, glide slope, and a mass floor.

The outer **SCvx** loop linearizes the dynamics about a reference, assembles a
convex **SOCP** subproblem, solves it with a custom interior-point method, and
updates the reference via a Levenberg-Marquardt trust-region ρ-ratio until the
dynamics defect (virtual control) vanishes.

## Workspace

| Crate | Role |
|-------|------|
| `scvx-core` | Shared types, parameters, status enums |
| `scvx-dynamics` | 3-DoF model, analytic Jacobians, FOH+RK4 discretization |
| `scvx-ipm` | SOCP interior-point method (AHO + Nesterov-Todd directions), cone primitives |
| `scvx-solver` | SCvx outer loop, SOCP assembly, preconditioning, O(N) structured KKT, high-level API |
| `scvx-ffi` | C-ABI wrapper for embedding in flight C/C++ (cross-compiles to bare-metal ARM) |

## Build & test

```sh
cargo test --workspace                  # full test suite
cargo clippy --all-targets -- -D warnings
cargo run --release --example mars_descent   # end-to-end Mars descent demo
# bare-metal cross-compile (no_std):
cargo build --release --target thumbv7em-none-eabihf -p scvx-solver
```

CI (`.github/workflows/ci.yml`) runs this matrix on every push. The solver core
cross-compiles to `thumbv7em-none-eabihf`; the C-FFI layer builds both a host
library and a `no_std` bare-metal static library.

## Solver notes (honest)

- **AHO is the production direction** — it converges to machine-precision dynamics
  feasibility across Mars / active-drag / lunar regimes with column
  preconditioning + a trust-region retry.
- The **Nesterov-Todd (NT)** direction is opt-in with graceful AHO fallback. It is
  *correct* on well-conditioned problems (verified against CVXPY/Clarabel and
  Julia/Clarabel oracles) but **diverges on flight-scale subproblems** where the
  relaxation cones vanish at the optimum — a documented limit of symmetric NT
  scaling, not a bug. `HANDOFF.md` records the full investigation.
- A **homogeneous self-dual (HSD) embedded** direction (`solve_socp_hsd`)
  **cracks that NT limit** — the production-solver approach (ECOS/Clarabel). On
  the *same* flight-scale subproblem where plain NT diverges (duality gap → 1e13),
  HSD converges to the external CVXPY/Clarabel + Julia optimum to **~1e-7 relative
  cost in 15 iterations** — tighter *and* faster than even AHO, and it converges
  on cases where AHO itself fails. It is a verified inner solver (not yet wired
  into the SCvx outer loop, and the O(N) structured version is the next lift); see
  `HANDOFF.md` "Phase 26".
- A **block-tridiagonal Schur** primitive provides an O(N) inner solve
  (per-step machine-precision-equivalent to dense).

## External-oracle validation

The Rust IPM is cross-checked against **CVXPY/Clarabel** and **JuMP/Clarabel** on
canonical SOCPs and on a real assembled flight-scale SCvx subproblem (see
`tools/` and `crates/scvx-solver/tests/`). The two external solvers agree to
~1e-9; the Rust solve matches the optimum within the documented tolerances.

## License

Dual-licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
