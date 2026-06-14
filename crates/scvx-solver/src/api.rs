//! High-level application API for the SCvx powered-descent solver.
//!
//! Hides the workspace construction and SCvx/IPM parameter twiddling
//! behind a single entrypoint, [`solve_powered_descent`]. Callers
//! provide physical parameters, initial state, terminal target, and
//! a few options; the solver builds a linear-interpolation reference,
//! runs SCvx with sensible defaults (AHO + column preconditioning),
//! and writes the converged trajectory back into the workspace.
//!
//! ## Const-generic dim helpers
//!
//! The SCvx workspace requires `NP`, `NE`, `NCT`, `NCONES` const
//! generics that depend on `N` and `use_free_tf`. The
//! [`workspace_np`], [`workspace_ne`], [`workspace_nct`],
//! [`workspace_ncones`] helpers compute them from `N` and the free-tf
//! flag. Callers declare their workspace via:
//!
//! ```ignore
//! const N: usize        = 5;
//! const FREE_TF: bool   = true;
//! const NP: usize       = workspace_np(N, FREE_TF);
//! const NE: usize       = workspace_ne(N);
//! const NCT: usize      = workspace_nct(N, FREE_TF);
//! const NCONES: usize   = workspace_ncones(N, FREE_TF);
//! const MAX_OUTER: usize = 20;
//!
//! let mut ws = Box::<ScvxWorkspace<N, NP, NE, NCT, NCONES, MAX_OUTER>>::default();
//! ```
//!
//! ## Default reference construction
//!
//! [`solve_powered_descent`] builds a linear interpolation between
//! `initial_state` and the terminal target (with mass linearly
//! decreasing to `target_mass`), then sets the per-node thrust to
//! hover-equivalent. This is a coarse initial guess but sufficient
//! for SCvx to converge — the outer loop will refine it.

use libm::{exp, log};
use nalgebra::SVector;
use scvx_core::{
    IpmAlgoParams, PhysicalParams, ScvxAlgoParams, SolverStatus,
};
use scvx_ipm::SocpProblem;

use crate::assemble::{
    ncones_scvx_free_tf, nct_scvx_free_tf, np_scvx_free_tf, TerminalCondition,
    N_CONES_PER_NODE_SCVX, N_CONE_DIM_PER_NODE_SCVX, N_EQ_PER_DYN, N_EQ_TERMINAL,
    N_VARS_PER_NODE_SCVX,
};
use crate::scvx::{solve_scvx, ScvxWorkspace};

// These helpers derive every dimension from the **authoritative** layout
// constants/const-fns in `assemble.rs` (the same ones the assembler's indexing
// math uses) — NOT from bare literals — so the FFI workspace-sizing path
// (`scvx_workspace_size_nN`, which calls `workspace_np`) can never silently
// disagree with the actual layout. Kept `const fn` for the FFI's const context.

/// Const-generic helper: `NP` for the SCvx workspace given problem size
/// `n` and whether free-final-time is enabled.
pub const fn workspace_np(n: usize, use_free_tf: bool) -> usize {
    if use_free_tf { np_scvx_free_tf(n) } else { n * N_VARS_PER_NODE_SCVX }
}

/// Const-generic helper: `NE` for the SCvx workspace (free-tf has no
/// extra equality rows — `δτ` only contributes to existing dynamics
/// rows via a new column).
pub const fn workspace_ne(n: usize) -> usize {
    n * N_EQ_PER_DYN + N_EQ_TERMINAL
}

/// Const-generic helper: `NCT` (total cone slack dim) for the SCvx
/// workspace given problem size `n` and free-tf flag.
pub const fn workspace_nct(n: usize, use_free_tf: bool) -> usize {
    if use_free_tf { nct_scvx_free_tf(n) } else { n * N_CONE_DIM_PER_NODE_SCVX }
}

/// Const-generic helper: `NCONES` (number of cones in the product) for
/// the SCvx workspace given problem size `n` and free-tf flag.
pub const fn workspace_ncones(n: usize, use_free_tf: bool) -> usize {
    if use_free_tf { ncones_scvx_free_tf(n) } else { n * N_CONES_PER_NODE_SCVX }
}

/// Configuration for [`solve_powered_descent`]. All fields have
/// production-tuned defaults; override sparingly.
#[derive(Clone, Copy)]
pub struct PoweredDescentOptions {
    // ---- Problem-shape ----
    /// Initial guess for the time-dilation scalar `τ` (seconds).
    /// When `use_free_tf` is `true`, the solver will adjust this within
    /// `[phys.tau_lo, phys.tau_hi]`; otherwise `τ` stays fixed.
    pub initial_tau:         f64,
    /// Estimated terminal vehicle mass (kg). The reference trajectory
    /// linearly interpolates `log(mass)` from `initial_state[6]` to
    /// `log(target_mass)`. Used only to seed the warm start — the SOCP
    /// doesn't enforce a target mass.
    pub target_mass:         f64,
    /// If `true`, the solver optimizes `τ` as a decision variable.
    /// Default `true` (real powered descent rarely has a fixed
    /// landing time).
    pub use_free_tf:         bool,

    // ---- Preconditioning ----
    /// Default `true` — per-variable column scaling. Required for
    /// flight-scale problems where cone slack magnitudes span ~6
    /// orders of magnitude.
    pub use_preconditioning: bool,
    /// Per-cone slack rescaling. Default `false` for free-tf
    /// (interacts with `δτ` in subtle ways at scale) and `true` for
    /// fixed-tf small problems.
    pub use_cone_row_scaling: bool,
    /// Use NT-direction IPM. Default `false` (AHO is more robust on
    /// the SCvx subproblem; see HANDOFF.md "open todo").
    pub use_nt_scaling:      bool,

    // ---- Termination ----
    /// SCvx outer-loop iteration cap.
    pub max_outer_iters:     u32,
    /// Inner-IPM iteration cap (per outer iteration).
    pub max_inner_iters:     u32,
    /// Outer-loop convergence tolerance on `‖dx‖ + ‖du‖`.
    pub conv_tol_x:          f64,
    /// Outer-loop convergence tolerance on `‖ν‖₁` (virtual control).
    pub conv_tol_virt:       f64,
    /// Inner-IPM tolerance (μ, primal, dual residuals — all the same).
    /// Default `1e-4` (loose enough that AHO doesn't degenerate at the
    /// optimum but tight enough that SCvx outer convergence proceeds).
    pub ipm_tol:             f64,

    // ---- Trust region ----
    pub trust_eta0:          f64,
    pub trust_eta_min:       f64,
    pub trust_eta_max:       f64,
    /// L1 penalty weight on virtual control. Larger values force `ν → 0`
    /// more aggressively but can hurt outer-loop conditioning.
    pub virt_weight:         f64,
}

impl Default for PoweredDescentOptions {
    fn default() -> Self {
        Self {
            initial_tau:          20.0,
            target_mass:         700.0,
            use_free_tf:          true,
            use_preconditioning:  true,
            use_cone_row_scaling: false,
            use_nt_scaling:       false,
            max_outer_iters:       15,
            // Inner IPM cap of 25 matches the existing demos and
            // empirically gives the cleanest convergence. Higher caps
            // can cause the IPM to drift too far at the cone boundary
            // and degrade subsequent outer iters' conditioning.
            max_inner_iters:       25,
            conv_tol_x:           1.0e-3,
            conv_tol_virt:        1.0e-7,
            ipm_tol:              1.0e-4,
            trust_eta0:           20.0,
            trust_eta_min:        1.0e-3,
            trust_eta_max:       100.0,
            virt_weight:          1.0e5,
        }
    }
}

/// Build the SCvx initial reference: linear interpolation in state
/// between `initial_state` and the terminal target, hover-thrust at
/// every node, `τ = options.initial_tau`.
///
/// Stored in `workspace.reference` so the outer loop can read it.
fn seed_linear_reference<
    const N:         usize,
    const NP:        usize,
    const NE:        usize,
    const NCT:       usize,
    const NCONES:    usize,
    const MAX_OUTER: usize,
>(
    ws:            &mut ScvxWorkspace<N, NP, NE, NCT, NCONES, MAX_OUTER>,
    phys:          &PhysicalParams,
    initial_state: &SVector<f64, 7>,
    terminal:      &TerminalCondition,
    options:       &PoweredDescentOptions,
) {
    // Build x_target from terminal + target_mass.
    let mut x_target = SVector::<f64, 7>::zeros();
    for i in 0..3 {
        x_target[i]     = terminal.r[i];
        x_target[3 + i] = terminal.v[i];
    }
    x_target[6] = log(options.target_mass.max(phys.m_dry));

    // Hover thrust along +z, using average mass over the trajectory.
    let m_avg = (exp(initial_state[6]) + options.target_mass) * 0.5;
    let u_hover_z = -m_avg * phys.g[2]; // g[2] < 0 ⇒ u_hover_z > 0

    for k in 0..N {
        let alpha = if N > 1 { k as f64 / (N - 1) as f64 } else { 0.0 };
        for i in 0..7 {
            ws.reference.x[(i, k)] = (1.0 - alpha) * initial_state[i] + alpha * x_target[i];
        }
        // Pure +z hover thrust at every node.
        ws.reference.u[(0, k)] = 0.0;
        ws.reference.u[(1, k)] = 0.0;
        ws.reference.u[(2, k)] = u_hover_z;
        ws.reference.sigma[k]  = u_hover_z;
    }
    ws.reference.tau = options.initial_tau;
}

/// High-level powered-descent solver. Builds a linear initial reference,
/// runs SCvx with the given options, and writes the converged trajectory
/// into `workspace.reference`. Returns the [`SolverStatus`].
///
/// # Const generics
///
/// - `N`: number of temporal nodes.
/// - `NP`, `NE`, `NCT`, `NCONES`: workspace dims; use [`workspace_np`],
///   [`workspace_ne`], [`workspace_nct`], [`workspace_ncones`] with the
///   appropriate `use_free_tf` flag to compute them.
/// - `MAX_OUTER`: maximum outer SCvx iterations (compile-time cap).
///
/// # Workspace
///
/// The caller provides a pre-allocated `ScvxWorkspace`. After return:
/// - On `Converged` or `OuterIterCap`: `workspace.reference` holds the
///   solution trajectory (state + control + σ + τ).
/// - On `InnerFailure` or `BadInput`: `workspace.reference` may be stale
///   or partially updated; do not consume.
///
/// # Example
///
/// ```ignore
/// const N: usize = 10;
/// const NP: usize = workspace_np(N, true);
/// const NE: usize = workspace_ne(N);
/// const NCT: usize = workspace_nct(N, true);
/// const NCONES: usize = workspace_ncones(N, true);
/// const MAX_OUTER: usize = 20;
///
/// let mut ws = Box::<ScvxWorkspace<N, NP, NE, NCT, NCONES, MAX_OUTER>>::default();
/// let phys = mars_params();
/// let initial_state = SVector::<f64, 7>::from_column_slice(&[
///     0.0, 0.0, 100.0,   // r: 100m altitude
///     0.0, 0.0,  -10.0,  // v: -10 m/s descent
///     800.0_f64.ln(),    // log mass
/// ]);
/// let terminal = TerminalCondition { r: [0.0; 3], v: [0.0; 3] };
/// let options = PoweredDescentOptions::default();
///
/// let status = solve_powered_descent(&mut ws, &phys, &initial_state, &terminal, &options);
/// ```
#[allow(clippy::too_many_arguments)]
pub fn solve_powered_descent<
    const N:         usize,
    const NP:        usize,
    const NE:        usize,
    const NCT:       usize,
    const NCONES:    usize,
    const MAX_OUTER: usize,
>(
    workspace:     &mut ScvxWorkspace<N, NP, NE, NCT, NCONES, MAX_OUTER>,
    phys:          &PhysicalParams,
    initial_state: &SVector<f64, 7>,
    terminal:      &TerminalCondition,
    options:       &PoweredDescentOptions,
) -> SolverStatus {
    // Pre-flight: the const-generic workspace dims (NP/NCT/NCONES) are chosen
    // by the caller INDEPENDENTLY of the runtime `use_free_tf` flag. A mismatch
    // (e.g. fixed-tf dims with `use_free_tf = true`) indexes out of bounds in
    // the assembler / preconditioner / extractor — a release-mode abort under
    // `panic = "abort"`, or UB if this crate is linked `panic = "unwind"` (as
    // it can be through the FFI staticlib). `debug_assert!` is a no-op in
    // release, so this MUST be a live runtime check. Reject with `BadInput`.
    // The comparisons are const-foldable (const generics vs `N`-derived
    // consts), so this costs nothing at runtime.
    if NP     != workspace_np(N, options.use_free_tf)
        || NE     != workspace_ne(N)
        || NCT    != workspace_nct(N, options.use_free_tf)
        || NCONES != workspace_ncones(N, options.use_free_tf)
    {
        return SolverStatus::BadInput;
    }

    // Seed linear-interpolation reference.
    seed_linear_reference(workspace, phys, initial_state, terminal, options);

    // Bundle into SCvx + IPM param structs.
    let algo = ScvxAlgoParams {
        max_outer_iters: options.max_outer_iters,
        trust_eta0:      options.trust_eta0,
        trust_eta_min:   options.trust_eta_min,
        trust_eta_max:   options.trust_eta_max,
        virt_weight:     options.virt_weight,
        conv_tol_x:      options.conv_tol_x,
        conv_tol_virt:   options.conv_tol_virt,
        use_free_tf:     options.use_free_tf,
        ..ScvxAlgoParams::default()
    };
    let ipm = IpmAlgoParams {
        max_iters:            options.max_inner_iters,
        tol_mu:               options.ipm_tol,
        tol_primal:           options.ipm_tol,
        tol_dual:             options.ipm_tol,
        tol_gap:              options.ipm_tol,
        use_preconditioning:  options.use_preconditioning,
        use_cone_row_scaling: options.use_cone_row_scaling,
        use_nt_scaling:       options.use_nt_scaling,
        ..IpmAlgoParams::default()
    };

    solve_scvx(workspace, phys, &algo, &ipm, initial_state, terminal)
}

/// Type alias to make the unused-prob-import lint happy. The
/// `SocpProblem` import is needed in the docstring's `ignore` example
/// for type inference but not at the module level.
type _ApiTypeAlias<const NP: usize, const NE: usize, const NCT: usize, const NCONES: usize>
    = SocpProblem<NP, NE, NCT, NCONES>;

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    extern crate std;
    use std::boxed::Box;
    use std::thread;

    use nalgebra::SVector;
    use scvx_core::{PhysicalParams, SolverStatus, G_MARS};

    use super::*;

    use std::eprintln;

    fn run_in_big_stack<F>(f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        thread::Builder::new()
            .stack_size(32 * 1024 * 1024)
            .spawn(f)
            .expect("spawn")
            .join()
            .expect("inner panic");
    }

    fn mars_params() -> PhysicalParams {
        PhysicalParams {
            g:             [0.0, 0.0, -G_MARS],
            m_dry:          200.0,
            m_wet:         1000.0,
            isp:            225.0,
            g0:               9.80665,
            t_min:         1000.0,
            t_max:         6000.0,
            cos_theta_max:    0.7660444,
            tan_gamma_gs:     1.0,
            rho:              0.0,
            cd_a:             0.0,
            tau_lo:           5.0,
            tau_hi:          50.0,
        }
    }

    /// Const-fn helpers must agree with the manual layout used in
    /// existing tests / docs.
    #[test]
    fn workspace_dim_helpers_match_layout_constants() {
        // Fixed-tf: NP = 19·N, NCT = 30·N, NCONES = 8·N.
        assert_eq!(workspace_np    (5, false),  95);
        assert_eq!(workspace_ne    (5),         41);
        assert_eq!(workspace_nct   (5, false), 150);
        assert_eq!(workspace_ncones(5, false),  40);

        // Free-tf: +1 to NP, +2 to NCT, +2 to NCONES.
        assert_eq!(workspace_np    (5, true),   96);
        assert_eq!(workspace_ne    (5),         41); // NE unchanged
        assert_eq!(workspace_nct   (5, true),  152);
        assert_eq!(workspace_ncones(5, true),   42);
    }

    /// **Headline end-to-end test**: `solve_powered_descent` on a tiny
    /// Mars problem (N=3), free-tf enabled. The high-level API must
    /// reach `OuterIterCap` or `Converged` cleanly without the caller
    /// touching `ScvxAlgoParams` or `IpmAlgoParams` directly.
    #[test]
    fn solve_powered_descent_runs_clean_on_small_mars_problem() {
        run_in_big_stack(|| {
            const N:         usize = 3;
            const FREE_TF:   bool  = true;
            const NP:        usize = workspace_np(N, FREE_TF);
            const NE:        usize = workspace_ne(N);
            const NCT:       usize = workspace_nct(N, FREE_TF);
            const NCONES:    usize = workspace_ncones(N, FREE_TF);
            const MAX_OUTER: usize = 15;

            let phys = mars_params();
            let mut initial_state = SVector::<f64, 7>::zeros();
            initial_state[2] = 2.0;          // r_z = 2 m
            initial_state[5] = -0.1;         // v_z = -0.1 m/s
            initial_state[6] = (400.0_f64).ln();

            let terminal = TerminalCondition { r: [0.0; 3], v: [0.0; 3] };
            let options = PoweredDescentOptions {
                initial_tau:         10.0,
                target_mass:        380.0,
                trust_eta0:           5.0,
                trust_eta_max:       20.0,
                ..PoweredDescentOptions::default()
            };

            let mut ws = Box::<ScvxWorkspace<N, NP, NE, NCT, NCONES, MAX_OUTER>>::default();
            let status = solve_powered_descent(
                &mut ws, &phys, &initial_state, &terminal, &options,
            );

            // Headline: clean termination.
            assert!(
                matches!(status, SolverStatus::Converged | SolverStatus::OuterIterCap),
                "high-level API returned unexpected status: {}", status as u32
            );

            // Reference must be in original physical units.
            for k in 0..N {
                for i in 0..7 {
                    assert!(ws.reference.x[(i, k)].is_finite());
                }
            }
            // r_z[0] should still be ~2 m.
            assert!(ws.reference.x[(2, 0)] > 0.5,
                    "reference appears scaled: r_z[0] = {}", ws.reference.x[(2, 0)]);
            // τ should be within bounds.
            assert!(ws.reference.tau >= phys.tau_lo);
            assert!(ws.reference.tau <= phys.tau_hi);
        });
    }

    /// **Fixed-tf path** of the same high-level API.
    #[test]
    fn solve_powered_descent_works_fixed_tf() {
        run_in_big_stack(|| {
            const N:         usize = 3;
            const FREE_TF:   bool  = false;
            const NP:        usize = workspace_np(N, FREE_TF);
            const NE:        usize = workspace_ne(N);
            const NCT:       usize = workspace_nct(N, FREE_TF);
            const NCONES:    usize = workspace_ncones(N, FREE_TF);
            const MAX_OUTER: usize = 15;

            let phys = mars_params();
            let mut initial_state = SVector::<f64, 7>::zeros();
            initial_state[2] = 2.0;
            initial_state[5] = -0.1;
            initial_state[6] = (400.0_f64).ln();

            let terminal = TerminalCondition { r: [0.0; 3], v: [0.0; 3] };
            // Fixed-tf with column preconditioning only. The default
            // `max_inner_iters = 25` works well here; we keep it
            // explicit for clarity in this test.
            let options = PoweredDescentOptions {
                initial_tau:         10.0,
                target_mass:        380.0,
                trust_eta0:           5.0,
                trust_eta_max:       20.0,
                use_free_tf:          false,
                use_cone_row_scaling: false,
                ..PoweredDescentOptions::default()
            };

            let mut ws = Box::<ScvxWorkspace<N, NP, NE, NCT, NCONES, MAX_OUTER>>::default();
            let status = solve_powered_descent(
                &mut ws, &phys, &initial_state, &terminal, &options,
            );
            eprintln!("fixed-tf solve: status = {} after {} outer iters",
                      status as u32, ws.iter + 1);
            for i in 0..=(ws.iter as usize).min(ws.history.len().saturating_sub(1)) {
                let r = &ws.history[i];
                eprintln!(
                    "  outer {:>2}: cost={:>12.4e}  trust={:>9.3e}  ‖ν‖={:>9.3e}  \
                     ρ={:>5.3}  accept={}  ipm={}/{}",
                    r.iter, r.cost, r.trust_eta, r.virt_l1, r.rho_ratio,
                    r.accepted, r.ipm_status, r.ipm_iters,
                );
            }
            assert!(
                matches!(status, SolverStatus::Converged | SolverStatus::OuterIterCap),
                "fixed-tf: got status {}", status as u32
            );
            // τ unchanged from initial.
            assert!((ws.reference.tau - 10.0).abs() < 1e-12,
                    "τ should be preserved in fixed-tf, got {}", ws.reference.tau);
        });
    }
}
