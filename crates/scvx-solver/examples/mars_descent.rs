//! End-to-end Mars powered-descent example.
//!
//! Solves a small (N=3) Mars landing problem using the high-level
//! [`scvx_solver::solve_powered_descent`] API and prints the resulting
//! trajectory. (N=3 is the validated small-scale sweet spot; see
//! `scvx_converges_larger_n_adaptive_trust` for the larger-N path.)
//!
//! ## Problem
//!
//! - Initial state: 2 m altitude, descending at 0.1 m/s, 400 kg mass.
//! - Terminal target: r = (0, 0, 0), v = (0, 0, 0) — soft landing, ~380 kg.
//! - Free-final-time enabled: solver picks `τ ∈ [tau_lo, tau_hi]` from an
//!   initial guess of 8 s.
//!
//! ## Run
//!
//! ```sh
//! cargo run --release --example mars_descent
//! ```

use nalgebra::SVector;
use scvx_core::{PhysicalParams, SolverStatus, G_MARS};
use scvx_solver::{
    solve_powered_descent, workspace_ncones, workspace_np, workspace_nct,
    PoweredDescentOptions, ScvxWorkspace, TerminalCondition,
};

const N:         usize = 3;
const FREE_TF:   bool  = true;
const NP:        usize = workspace_np(N, FREE_TF);
const NE:        usize = scvx_solver::workspace_ne(N);
const NCT:       usize = workspace_nct(N, FREE_TF);
const NCONES:    usize = workspace_ncones(N, FREE_TF);
const MAX_OUTER: usize = 20;

fn main() {
    // On Windows, the default main-thread stack (~1 MB) is too small
    // for the const-generic SOCP workspace at N≥3. Spawn a 32 MB
    // worker thread to run the demo (the production deployment would
    // place the workspace in a static / pre-allocated arena instead).
    let handle = std::thread::Builder::new()
        .stack_size(32 * 1024 * 1024)
        .spawn(run_demo)
        .expect("spawn worker thread");
    handle.join().expect("worker panicked");
}

fn run_demo() {
    // ---- Mars physical parameters ----
    let phys = PhysicalParams {
        g:             [0.0, 0.0, -G_MARS],   // m/s²
        m_dry:          200.0,                // kg
        m_wet:         1000.0,                // kg
        isp:            225.0,                // s
        g0:             9.80665,              // m/s²
        t_min:         1000.0,                // N
        t_max:         6000.0,                // N
        cos_theta_max: 0.7660444,             // 40° pointing half-angle
        tan_gamma_gs:  1.0,                   // 45° glide slope
        rho:           0.0,                   // no drag for this demo
        cd_a:          0.0,
        tau_lo:        5.0,                   // s
        tau_hi:       50.0,                   // s
    };

    // ---- Initial state ----
    // r = (0, 0, 2), v = (0, 0, -0.1), m = 400 kg
    // (Gentle terminal-approach configuration. N=3 is the validated
    // sweet spot for the current solver; larger problems are
    // possible with looser IPM tolerances but more delicate.)
    let mut initial_state = SVector::<f64, 7>::zeros();
    initial_state[2] = 2.0;
    initial_state[5] = -0.1;
    initial_state[6] = (400.0_f64).ln();

    // ---- Terminal target: soft landing at the origin ----
    let terminal = TerminalCondition {
        r: [0.0; 3],
        v: [0.0; 3],
    };

    // ---- Solver options ----
    // Free-tf small-scale Mars descent. The reference τ is set to
    // 8 s (deliberately suboptimal); the solver should pull it
    // toward a better value within `[tau_lo, tau_hi]`.
    let options = PoweredDescentOptions {
        initial_tau:        8.0,
        target_mass:      380.0,
        use_free_tf:       true,
        use_preconditioning: true,
        use_cone_row_scaling: false,
        max_outer_iters:    15,
        trust_eta0:          5.0,
        trust_eta_max:      20.0,
        trust_eta_min:       1.0e-3,
        ..PoweredDescentOptions::default()
    };

    // ---- Allocate workspace on the heap ----
    // In flight, this would be a static. The Box::default() pattern
    // here mirrors the test infrastructure.
    let mut ws = Box::<ScvxWorkspace<N, NP, NE, NCT, NCONES, MAX_OUTER>>::default();

    println!("\n=== Mars powered-descent demo (N={N}) ===\n");
    println!("Initial state:");
    println!("  r = ({:>8.2}, {:>8.2}, {:>8.2}) m",
             initial_state[0], initial_state[1], initial_state[2]);
    println!("  v = ({:>8.2}, {:>8.2}, {:>8.2}) m/s",
             initial_state[3], initial_state[4], initial_state[5]);
    println!("  m = {:>8.2} kg", initial_state[6].exp());

    println!("\nTerminal target:");
    println!("  r = ({:>8.2}, {:>8.2}, {:>8.2}) m",
             terminal.r[0], terminal.r[1], terminal.r[2]);
    println!("  v = ({:>8.2}, {:>8.2}, {:>8.2}) m/s",
             terminal.v[0], terminal.v[1], terminal.v[2]);

    println!("\nSolver options:");
    println!("  initial_tau     = {:>5.1} s   (τ_lo={}, τ_hi={})",
             options.initial_tau, phys.tau_lo, phys.tau_hi);
    println!("  use_free_tf     = {}", options.use_free_tf);
    println!("  preconditioning = {}", options.use_preconditioning);
    println!("  direction       = {}",
             if options.use_hsd { "HSD (homogeneous self-dual) [default]" }
             else if options.use_nt_scaling { "NT" } else { "AHO" });
    println!("  max_outer_iters = {}", options.max_outer_iters);
    println!();

    // ---- Solve ----
    let status = solve_powered_descent(
        &mut ws, &phys, &initial_state, &terminal, &options,
    );

    // ---- Print SCvx iteration history ----
    let last = ws.iter as usize;
    println!("=== SCvx outer-loop trace ===\n");
    println!("  iter | cost          | trust    | ‖ν‖       | ρ     | accept | ipm");
    println!("  -----+---------------+----------+-----------+-------+--------+-------");
    for i in 0..=last.min(ws.history.len().saturating_sub(1)) {
        let r = &ws.history[i];
        println!(
            "  {:>4} | {:>13.4e} | {:>8.3e} | {:>9.3e} | {:>5.2} | {:>6} | {:>2}/{}",
            r.iter, r.cost, r.trust_eta, r.virt_l1, r.rho_ratio,
            r.accepted, r.ipm_status, r.ipm_iters,
        );
    }

    println!("\n=== Result ===\n");
    let status_str = match status {
        SolverStatus::Converged    => "Converged",
        SolverStatus::OuterIterCap => "OuterIterCap (max outer reached, partial convergence)",
        SolverStatus::InnerFailure => "InnerFailure (inner IPM gave up)",
        SolverStatus::Infeasible   => "Infeasible",
        SolverStatus::BadInput     => "BadInput (caller error)",
    };
    println!("Status: {} ({})", status_str, status as u32);
    println!("τ (final): {:>6.3} s   (initial guess {:>5.1} s — adjusted {:+.3} s)",
             ws.reference.tau, options.initial_tau,
             ws.reference.tau - options.initial_tau);

    // ---- Print trajectory ----
    if matches!(status, SolverStatus::Converged | SolverStatus::OuterIterCap) {
        println!("\n=== Final trajectory ===\n");
        let dt_norm = if N > 1 { 1.0 / (N - 1) as f64 } else { 0.0 };
        let dt_s    = dt_norm * ws.reference.tau;
        println!("  k | t (s)  | r_x       r_y       r_z       | v_x       v_y       v_z       | mass (kg) | u_x       u_y       u_z       | σ (N)");
        println!("  --+--------+---------------------------------+---------------------------------+-----------+---------------------------------+----------");
        for k in 0..N {
            let t = (k as f64) * dt_s;
            println!(
                "  {:>1} | {:>6.3} | {:>+8.3}  {:>+8.3}  {:>+8.3}  | {:>+8.3}  {:>+8.3}  {:>+8.3}  | {:>9.2} | {:>+8.1}  {:>+8.1}  {:>+8.1}  | {:>8.1}",
                k, t,
                ws.reference.x[(0, k)], ws.reference.x[(1, k)], ws.reference.x[(2, k)],
                ws.reference.x[(3, k)], ws.reference.x[(4, k)], ws.reference.x[(5, k)],
                ws.reference.x[(6, k)].exp(),
                ws.reference.u[(0, k)], ws.reference.u[(1, k)], ws.reference.u[(2, k)],
                ws.reference.sigma[k],
            );
        }

        // ---- Quality summary ----
        let final_virt_l1 = ws.history[last].virt_l1;
        let final_cost    = ws.history[last].cost;
        println!("\n=== Quality summary ===");
        println!("  Final cost (Σσ + virt·‖ν‖):  {:.4e}", final_cost);
        println!("  Final ‖ν‖₁ (virtual ctrl):    {:.4e}", final_virt_l1);
        println!("  Outer iters used:             {} of {}", last + 1, MAX_OUTER);
    } else {
        println!("\nSolve did not produce a usable trajectory.");
        std::process::exit(1);
    }

    println!();
}
