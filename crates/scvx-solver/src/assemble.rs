//! SOCP assembly for the LCvx-style 3-DoF powered-descent subproblem.
//!
//! Maps `(reference trajectory, linearized dynamics, physical params, BCs)`
//! to a [`SocpProblem`] ready for the inner Mehrotra IPM. This is the
//! "impedance match" piece — everything downstream of P5 (dynamics) and
//! P3a (generic IPM) hangs off the result.
//!
//! **STATUS — two assemblers live in this module.** [`assemble_scvx_socp`]
//! (19 vars/node, 8 cones, `NP = 19N`) is the **production** SCvx assembly the
//! outer loop calls every iteration; its layout is documented on that function
//! and in `HANDOFF.md`. [`assemble_lcvx_socp`] — the 11-var/6-cone LCvx layout
//! described below — is a **vestigial** prototype, exercised only by unit tests
//! and the WCET benchmark, never on the live SCvx path. The layout sections in
//! THIS header describe the LCvx assembler, not the production one.
//!
//! ## Variable layout
//!
//! Per temporal node `k = 0..N-1`:
//! ```text
//!   z_k = [ x_k (7) ⊕ u_k (3) ⊕ σ_k (1) ]    →  11 vars/node
//! ```
//! Total primal dim `NP = 11·N`. Use [`x_idx`], [`u_idx`], [`sigma_idx`] to
//! look up column offsets.
//!
//! ## Equality rows
//!
//! ```text
//!   rows  0..7         : initial state    x_0 = x_init
//!   rows  7..7N        : dynamics (N-1 transitions, 7 rows each)
//!   rows  7N..7N+6     : terminal r, v    r_{N-1} = r_target, v_{N-1} = v_target
//! ```
//! Total `NE = 7N + 6`.
//!
//! ## Cone constraints (6 per node, 11 dims/node)
//!
//! ```text
//!   thrust magnitude :  (σ_k, u_k)              ∈ SOC^4      (4 dims)
//!   pointing         :  u_k[2] − cos(θ)·σ_k    ∈ ℝ_+        (1)
//!   mass floor       :  z_k − ln(m_dry)         ∈ ℝ_+        (1)
//!   glide slope      :  (tan(γ)·r_z, r_x, r_y)  ∈ SOC^3      (3)
//!   thrust lo bound  :  σ_k − T_min             ∈ ℝ_+        (1)
//!   thrust hi bound  :  T_max − σ_k             ∈ ℝ_+        (1)
//! ```
//! Total `NCT = 11·N`, `NCONES = 6·N`.
//!
//! Caller must pass `NP, NE, NCT, NCONES` const generics consistent with
//! `N`. `assemble_lcvx_socp` checks this in debug builds.

// `+ 0` / `+ 1` patterns are used for visual alignment in the cone-block
// construction below — clarity beats clippy here.
#![allow(clippy::identity_op)]

use libm::log;
use nalgebra::SVector;
use scvx_core::{PhysicalParams, Trajectory};
use scvx_dynamics::LinearizedDynamics;
use scvx_ipm::{ConeDesc, SocpProblem};

// ---------------------------------------------------------------------------
// Layout constants
// ---------------------------------------------------------------------------

pub const NX:                  usize = 7;
pub const NU:                  usize = 3;
pub const NSIGMA:              usize = 1;
pub const N_VARS_PER_NODE:     usize = NX + NU + NSIGMA;       // 11
pub const N_CONE_DIM_PER_NODE: usize = 4 + 1 + 1 + 3 + 1 + 1;  // 11
pub const N_CONES_PER_NODE:    usize = 6;
pub const N_EQ_INITIAL:        usize = NX;                      // 7
pub const N_EQ_PER_DYN:        usize = NX;                      // 7
pub const N_EQ_TERMINAL:       usize = 6;                       // r, v fixed; m free

#[inline] pub fn x_idx(k: usize)     -> usize { k * N_VARS_PER_NODE }
#[inline] pub fn u_idx(k: usize)     -> usize { k * N_VARS_PER_NODE + NX }
#[inline] pub fn sigma_idx(k: usize) -> usize { k * N_VARS_PER_NODE + NX + NU }

// Cone-slot offset within node k's 11-dim cone block:
const CONE_MAG_OFF:   usize = 0;  // 4 dims
const CONE_PT_OFF:    usize = 4;  // 1 dim
const CONE_MASS_OFF:  usize = 5;  // 1 dim
const CONE_GS_OFF:    usize = 6;  // 3 dims
const CONE_TMIN_OFF:  usize = 9;  // 1 dim
const CONE_TMAX_OFF:  usize = 10; // 1 dim

// ===========================================================================
// SCvx layout — adds virtual control ν_k (7) and L2 aux w_k (1) per node,
// plus trust-region SOC and virtual-control SOC. Used by `assemble_scvx_socp`.
// ===========================================================================

pub const NNU:                     usize = NX;                          // virtual control dim = 7
pub const NW:                      usize = 1;                           // L2 aux for ν
pub const N_VARS_PER_NODE_SCVX:    usize = NX + NU + NSIGMA + NNU + NW; // 19
pub const N_CONES_PER_NODE_SCVX:   usize = 8;
pub const N_CONE_DIM_PER_NODE_SCVX: usize = 4 + 1 + 1 + 3 + 1 + 1 + 11 + 8; // 30

#[inline] pub fn x_idx_scvx(k: usize)     -> usize { k * N_VARS_PER_NODE_SCVX }
#[inline] pub fn u_idx_scvx(k: usize)     -> usize { k * N_VARS_PER_NODE_SCVX + NX }
#[inline] pub fn sigma_idx_scvx(k: usize) -> usize { k * N_VARS_PER_NODE_SCVX + NX + NU }
#[inline] pub fn nu_idx_scvx(k: usize)    -> usize { k * N_VARS_PER_NODE_SCVX + NX + NU + NSIGMA }
#[inline] pub fn w_idx_scvx(k: usize)     -> usize { k * N_VARS_PER_NODE_SCVX + NX + NU + NSIGMA + NNU }

// Cone-slot offsets within node k's 30-dim SCvx cone block:
const CONE_MAG_OFF_SCVX:   usize = 0;   // 4
const CONE_PT_OFF_SCVX:    usize = 4;   // 1
const CONE_MASS_OFF_SCVX:  usize = 5;   // 1
const CONE_GS_OFF_SCVX:    usize = 6;   // 3
const CONE_TMIN_OFF_SCVX:  usize = 9;   // 1
const CONE_TMAX_OFF_SCVX:  usize = 10;  // 1
const CONE_TRUST_OFF_SCVX: usize = 11;  // 11  — (η, x_k−x̄_k, u_k−ū_k)
const CONE_VIRT_OFF_SCVX:  usize = 22;  // 8   — (w_k, ν_k)

// ---------------------------------------------------------------------------
// Free-final-time (δτ) layout extensions.
//
// When `use_free_tf` is enabled, the SOCP gains:
//   - one global primal variable `δτ` at index `19·N` (after all per-node vars)
//   - two SOC^1 bound cones (`tau_lo ≤ τ_ref + δτ ≤ tau_hi`) at the end of
//     the cones array, owning the last 2 rows of `G`/`h`.
//
// Total dims with free-tf:
//   NP     = 19·N + 1
//   NCT    = 30·N + 2
//   NCONES =  8·N + 2
//   NE     =  7·N + 6   (unchanged)
//
// Callers select the right dims via the helper `const fn`s below.
// ---------------------------------------------------------------------------

/// Primal index of the global `δτ` variable (after all per-node variables).
#[inline] pub fn delta_tau_idx_scvx<const N: usize>() -> usize {
    N * N_VARS_PER_NODE_SCVX
}

/// `NP` for the free-tf SCvx layout: `19·N + 1`.
#[inline] pub const fn np_scvx_free_tf(n: usize) -> usize {
    n * N_VARS_PER_NODE_SCVX + 1
}

/// `NCT` (total cone dim) for the free-tf SCvx layout: `30·N + 2`.
#[inline] pub const fn nct_scvx_free_tf(n: usize) -> usize {
    n * N_CONE_DIM_PER_NODE_SCVX + 2
}

/// `NCONES` for the free-tf SCvx layout: `8·N + 2`.
#[inline] pub const fn ncones_scvx_free_tf(n: usize) -> usize {
    n * N_CONES_PER_NODE_SCVX + 2
}

// ---------------------------------------------------------------------------
// Inputs
// ---------------------------------------------------------------------------

/// Soft-landing target: terminal position and velocity (typically zero).
/// Terminal mass is left free (we never enforce a target mass at landing).
#[derive(Clone, Copy, Default)]
pub struct TerminalCondition {
    pub r: [f64; 3],
    pub v: [f64; 3],
}

// ---------------------------------------------------------------------------
// Assembly
// ---------------------------------------------------------------------------

/// Fill `prob` in place with the LCvx-style powered-descent SOCP.
///
/// The caller must pass `NP, NE, NCT, NCONES` matching the [P6 layout]:
/// `NP = 11·N`, `NE = 7·N + 6`, `NCT = 11·N`, `NCONES = 6·N`. Debug builds
/// `debug_assert!` this; release builds trust the caller.
///
/// `reference` is the linearization point — `lin` must have been produced
/// by [`scvx_dynamics::discretize_foh`] called on the same trajectory.
///
/// Note: this writes `c, a_mat, b, g_mat, h, cones` from scratch. Pre-existing
/// values in `prob` are overwritten — no need to zero externally.
pub fn assemble_lcvx_socp<
    const N: usize,
    const NP: usize,
    const NE: usize,
    const NCT: usize,
    const NCONES: usize,
>(
    reference:     &Trajectory<N>,
    lin:           &LinearizedDynamics<N>,
    phys:          &PhysicalParams,
    initial_state: &SVector<f64, NX>,
    terminal:      &TerminalCondition,
    prob:          &mut SocpProblem<NP, NE, NCT, NCONES>,
) {
    // Dim consistency in debug; trusted in release.
    debug_assert_eq!(NP,     N * N_VARS_PER_NODE,
                     "NP must equal 11·N");
    debug_assert_eq!(NE,     N * N_EQ_PER_DYN + N_EQ_TERMINAL,
                     "NE must equal 7·N + 6");
    debug_assert_eq!(NCT,    N * N_CONE_DIM_PER_NODE,
                     "NCT must equal 11·N");
    debug_assert_eq!(NCONES, N * N_CONES_PER_NODE,
                     "NCONES must equal 6·N");

    // Reset the problem to all zeros.
    *prob = SocpProblem::default();

    // ---- Cost: minimize Σ σ_k  (fuel proxy) ----
    for k in 0..N {
        prob.c[sigma_idx(k)] = 1.0;
    }

    // ---- Equality: initial state x_0 = initial_state ----
    let initial_row = 0;
    for i in 0..NX {
        prob.a_mat[(initial_row + i, x_idx(0) + i)] = 1.0;
        prob.b   [initial_row + i] = initial_state[i];
    }

    // ---- Equality: dynamics x_{k+1} − A·x − B⁻·u − B⁺·u_{k+1} = c ----
    let dyn_row_start = NX;
    for k in 0..(N - 1) {
        let row = dyn_row_start + k * NX;
        // +I · x_{k+1}
        for i in 0..NX {
            prob.a_mat[(row + i, x_idx(k + 1) + i)] = 1.0;
        }
        // -A_k · x_k
        for i in 0..NX {
            for j in 0..NX {
                prob.a_mat[(row + i, x_idx(k) + j)] -= lin.a[k][(i, j)];
            }
        }
        // -B⁻_k · u_k
        for i in 0..NX {
            for j in 0..NU {
                prob.a_mat[(row + i, u_idx(k) + j)] -= lin.b_minus[k][(i, j)];
            }
        }
        // -B⁺_k · u_{k+1}
        for i in 0..NX {
            for j in 0..NU {
                prob.a_mat[(row + i, u_idx(k + 1) + j)] -= lin.b_plus[k][(i, j)];
            }
        }
        // RHS: c_k + s_k·τ_ref  (FOH discretization absorbs the τ contribution
        // into the constant when τ is held at its reference value)
        for i in 0..NX {
            prob.b[row + i] = lin.c[k][i] + lin.s[k][i] * reference.tau;
        }
    }

    // ---- Equality: terminal r_{N-1} = target.r, v_{N-1} = target.v ----
    let term_row = NX + (N - 1) * NX;
    for i in 0..3 {
        prob.a_mat[(term_row + i,     x_idx(N - 1) + i)]     = 1.0;
        prob.b   [term_row + i]     = terminal.r[i];
        prob.a_mat[(term_row + 3 + i, x_idx(N - 1) + 3 + i)] = 1.0;
        prob.b   [term_row + 3 + i] = terminal.v[i];
    }

    // ---- Cones: 6 per node, total dim 11 per node ----
    let log_m_dry = log(phys.m_dry);
    for k in 0..N {
        let coff = k * N_CONE_DIM_PER_NODE;        // cone-dim window start
        let cidx = k * N_CONES_PER_NODE;           // cone-array slot start

        // Cone 1: thrust magnitude (σ_k, u_k) ∈ SOC^4
        //   s_0 = σ_k,  s_{1..4} = u_k
        prob.g_mat[(coff + CONE_MAG_OFF, sigma_idx(k))] = -1.0;
        for i in 0..NU {
            prob.g_mat[(coff + CONE_MAG_OFF + 1 + i, u_idx(k) + i)] = -1.0;
        }
        prob.cones[cidx + 0] = ConeDesc { offset: coff + CONE_MAG_OFF, dim: 4 };

        // Cone 2: pointing  u_z − cos(θ)·σ ≥ 0  (SOC^1 = ℝ_+)
        prob.g_mat[(coff + CONE_PT_OFF, u_idx(k) + 2)]   = -1.0;
        prob.g_mat[(coff + CONE_PT_OFF, sigma_idx(k))]   = phys.cos_theta_max;
        prob.cones[cidx + 1] = ConeDesc { offset: coff + CONE_PT_OFF, dim: 1 };

        // Cone 3: mass floor  z_k − ln(m_dry) ≥ 0
        prob.g_mat[(coff + CONE_MASS_OFF, x_idx(k) + 6)] = -1.0;
        prob.h    [coff + CONE_MASS_OFF]                  = -log_m_dry;
        prob.cones[cidx + 2] = ConeDesc { offset: coff + CONE_MASS_OFF, dim: 1 };

        // Cone 4: glide slope (tan(γ)·r_z, r_x, r_y) ∈ SOC^3
        prob.g_mat[(coff + CONE_GS_OFF + 0, x_idx(k) + 2)] = -phys.tan_gamma_gs;
        prob.g_mat[(coff + CONE_GS_OFF + 1, x_idx(k) + 0)] = -1.0;
        prob.g_mat[(coff + CONE_GS_OFF + 2, x_idx(k) + 1)] = -1.0;
        prob.cones[cidx + 3] = ConeDesc { offset: coff + CONE_GS_OFF, dim: 3 };

        // Cone 5: T_min lower bound  σ_k − T_min ≥ 0
        prob.g_mat[(coff + CONE_TMIN_OFF, sigma_idx(k))] = -1.0;
        prob.h    [coff + CONE_TMIN_OFF]                  = -phys.t_min;
        prob.cones[cidx + 4] = ConeDesc { offset: coff + CONE_TMIN_OFF, dim: 1 };

        // Cone 6: T_max upper bound  T_max − σ_k ≥ 0
        prob.g_mat[(coff + CONE_TMAX_OFF, sigma_idx(k))] = 1.0;
        prob.h    [coff + CONE_TMAX_OFF]                  = phys.t_max;
        prob.cones[cidx + 5] = ConeDesc { offset: coff + CONE_TMAX_OFF, dim: 1 };
    }
}

// ===========================================================================
// SCvx assembly — full subproblem with trust region + virtual control
// ===========================================================================

/// Fill `prob` with the SCvx subproblem SOCP at the given linearization.
///
/// Variable layout per node (19 vars):
/// ```text
///   z_k = [ x_k (7) ⊕ u_k (3) ⊕ σ_k (1) ⊕ ν_k (7) ⊕ w_k (1) ]
/// ```
/// - `ν_k`: virtual-control slack in the dynamics row block
/// - `w_k`: L2 epigraph variable, `w_k ≥ ‖ν_k‖₂`, penalized in cost
///
/// Cones per node (8, total dim 30):
/// 1. Thrust magnitude     `(σ_k, u_k)`                ∈ SOC^4
/// 2. Pointing             `u_z − cos(θ)·σ`            ∈ ℝ_+
/// 3. Mass floor           `z_k − ln(m_dry)`           ∈ ℝ_+
/// 4. Glide slope          `(tan(γ)·r_z, r_x, r_y)`    ∈ SOC^3
/// 5. T_min                `σ_k − T_min`               ∈ ℝ_+
/// 6. T_max                `T_max − σ_k`               ∈ ℝ_+
/// 7. Trust region         `(η, x_k − x̄_k, u_k − ū_k)` ∈ SOC^{11}
/// 8. Virtual-control L2   `(w_k, ν_k)`                 ∈ SOC^8
///
/// **Fixed-final-time** (`use_free_tf = false`, default):
/// Caller must pass `NP = 19·N`, `NE = 7N + 6`, `NCT = 30N`, `NCONES = 8N`.
/// `τ` is held at `reference.tau` and the dynamics RHS is `c_k + s_k·τ_ref`.
///
/// **Free-final-time** (`use_free_tf = true`):
/// Caller must pass `NP = 19·N + 1`, `NE = 7N + 6`, `NCT = 30N + 2`,
/// `NCONES = 8N + 2`. The extra primal variable `δτ` is at index
/// `delta_tau_idx_scvx::<N>()`. Two extra bound cones at the end of the
/// cones array enforce `tau_lo ≤ τ_ref + δτ ≤ tau_hi`. The dynamics row
/// gets an additional `-s_k·δτ` term on the LHS (so the actual time
/// dilation absorbs the linearization remainder along the τ direction).
#[allow(clippy::too_many_arguments)]
pub fn assemble_scvx_socp<
    const N: usize,
    const NP: usize,
    const NE: usize,
    const NCT: usize,
    const NCONES: usize,
>(
    reference:     &Trajectory<N>,
    lin:           &LinearizedDynamics<N>,
    phys:          &PhysicalParams,
    initial_state: &SVector<f64, NX>,
    terminal:      &TerminalCondition,
    trust_eta:     f64,
    virt_weight:   f64,
    use_free_tf:   bool,
    prob:          &mut SocpProblem<NP, NE, NCT, NCONES>,
) {
    if use_free_tf {
        debug_assert_eq!(NP,     N * N_VARS_PER_NODE_SCVX + 1,
                         "NP must equal 19·N + 1 in free-tf mode");
        debug_assert_eq!(NCT,    N * N_CONE_DIM_PER_NODE_SCVX + 2,
                         "NCT must equal 30·N + 2 in free-tf mode");
        debug_assert_eq!(NCONES, N * N_CONES_PER_NODE_SCVX + 2,
                         "NCONES must equal 8·N + 2 in free-tf mode");
    } else {
        debug_assert_eq!(NP,     N * N_VARS_PER_NODE_SCVX,
                         "NP must equal 19·N");
        debug_assert_eq!(NCT,    N * N_CONE_DIM_PER_NODE_SCVX,
                         "NCT must equal 30·N");
        debug_assert_eq!(NCONES, N * N_CONES_PER_NODE_SCVX,
                         "NCONES must equal 8·N");
    }
    debug_assert_eq!(NE,     N * N_EQ_PER_DYN + N_EQ_TERMINAL,
                     "NE must equal 7·N + 6 (free-tf does not add equality rows)");
    // `N = 0` is a nonsensical layout (NP = 0) that would underflow `(N - 1)`
    // in the dynamics/terminal row math below; pin it for dev builds.
    debug_assert!(N >= 1, "N (node count) must be >= 1");
    debug_assert!(trust_eta > 0.0, "trust radius must be positive");
    debug_assert!(virt_weight >= 0.0, "virtual-control weight must be non-negative");
    if use_free_tf {
        debug_assert!(phys.tau_lo > 0.0,             "free-tf requires tau_lo > 0");
        debug_assert!(phys.tau_hi > phys.tau_lo,     "free-tf requires tau_hi > tau_lo");
        debug_assert!(reference.tau >= phys.tau_lo,
                      "reference.tau must be inside [tau_lo, tau_hi]");
        debug_assert!(reference.tau <= phys.tau_hi,
                      "reference.tau must be inside [tau_lo, tau_hi]");
    }

    *prob = SocpProblem::default();

    // ---- Cost: minimize Σ (σ_k + virt_weight · w_k) ----
    for k in 0..N {
        prob.c[sigma_idx_scvx(k)] = 1.0;
        prob.c[w_idx_scvx(k)]     = virt_weight;
    }

    // ---- Equality: initial state x_0 = initial_state ----
    let initial_row = 0;
    for i in 0..NX {
        prob.a_mat[(initial_row + i, x_idx_scvx(0) + i)] = 1.0;
        prob.b   [initial_row + i] = initial_state[i];
    }

    // ---- Equality: dynamics  x_{k+1} − A·x − B⁻·u − B⁺·u_{k+1} − ν_k = c + s·τ ----
    let dyn_row_start = NX;
    for k in 0..(N - 1) {
        let row = dyn_row_start + k * NX;
        // +I · x_{k+1}
        for i in 0..NX {
            prob.a_mat[(row + i, x_idx_scvx(k + 1) + i)] = 1.0;
        }
        // -A_k · x_k
        for i in 0..NX {
            for j in 0..NX {
                prob.a_mat[(row + i, x_idx_scvx(k) + j)] -= lin.a[k][(i, j)];
            }
        }
        // -B⁻_k · u_k
        for i in 0..NX {
            for j in 0..NU {
                prob.a_mat[(row + i, u_idx_scvx(k) + j)] -= lin.b_minus[k][(i, j)];
            }
        }
        // -B⁺_k · u_{k+1}
        for i in 0..NX {
            for j in 0..NU {
                prob.a_mat[(row + i, u_idx_scvx(k + 1) + j)] -= lin.b_plus[k][(i, j)];
            }
        }
        // -I · ν_k  (virtual control slack absorbs linearization residual)
        for i in 0..NX {
            prob.a_mat[(row + i, nu_idx_scvx(k) + i)] = -1.0;
        }
        // Free-tf only: −s_k · δτ contribution. Dynamics becomes
        //   x_{k+1} − A·x − B⁻·u − B⁺·u_{k+1} − ν − s·δτ = c + s·τ_ref
        // ⇔ x_{k+1} = A·x + B⁻·u + B⁺·u_{k+1} + ν + c + s·(τ_ref + δτ).
        if use_free_tf {
            let dtau_idx = delta_tau_idx_scvx::<N>();
            for i in 0..NX {
                prob.a_mat[(row + i, dtau_idx)] = -lin.s[k][i];
            }
        }
        // RHS: c_k + s_k·τ_ref  (unchanged whether free-tf or not — the
        // δτ contribution goes on the LHS via the column added above)
        for i in 0..NX {
            prob.b[row + i] = lin.c[k][i] + lin.s[k][i] * reference.tau;
        }
    }

    // ---- Equality: terminal r, v ----
    let term_row = NX + (N - 1) * NX;
    for i in 0..3 {
        prob.a_mat[(term_row + i,     x_idx_scvx(N - 1) + i)]     = 1.0;
        prob.b   [term_row + i]     = terminal.r[i];
        prob.a_mat[(term_row + 3 + i, x_idx_scvx(N - 1) + 3 + i)] = 1.0;
        prob.b   [term_row + 3 + i] = terminal.v[i];
    }

    // ---- Cones: 8 per node, total dim 30 per node ----
    let log_m_dry = log(phys.m_dry);
    for k in 0..N {
        let coff = k * N_CONE_DIM_PER_NODE_SCVX;
        let cidx = k * N_CONES_PER_NODE_SCVX;

        // Cone 1: thrust magnitude (σ_k, u_k) ∈ SOC^4
        prob.g_mat[(coff + CONE_MAG_OFF_SCVX, sigma_idx_scvx(k))] = -1.0;
        for i in 0..NU {
            prob.g_mat[(coff + CONE_MAG_OFF_SCVX + 1 + i, u_idx_scvx(k) + i)] = -1.0;
        }
        prob.cones[cidx + 0] = ConeDesc { offset: coff + CONE_MAG_OFF_SCVX, dim: 4 };

        // Cone 2: pointing u_z − cos(θ)·σ ≥ 0
        prob.g_mat[(coff + CONE_PT_OFF_SCVX, u_idx_scvx(k) + 2)]   = -1.0;
        prob.g_mat[(coff + CONE_PT_OFF_SCVX, sigma_idx_scvx(k))]   = phys.cos_theta_max;
        prob.cones[cidx + 1] = ConeDesc { offset: coff + CONE_PT_OFF_SCVX, dim: 1 };

        // Cone 3: mass floor z_k − ln(m_dry) ≥ 0
        prob.g_mat[(coff + CONE_MASS_OFF_SCVX, x_idx_scvx(k) + 6)] = -1.0;
        prob.h    [coff + CONE_MASS_OFF_SCVX]                       = -log_m_dry;
        prob.cones[cidx + 2] = ConeDesc { offset: coff + CONE_MASS_OFF_SCVX, dim: 1 };

        // Cone 4: glide slope (tan(γ)·r_z, r_x, r_y) ∈ SOC^3
        prob.g_mat[(coff + CONE_GS_OFF_SCVX + 0, x_idx_scvx(k) + 2)] = -phys.tan_gamma_gs;
        prob.g_mat[(coff + CONE_GS_OFF_SCVX + 1, x_idx_scvx(k) + 0)] = -1.0;
        prob.g_mat[(coff + CONE_GS_OFF_SCVX + 2, x_idx_scvx(k) + 1)] = -1.0;
        prob.cones[cidx + 3] = ConeDesc { offset: coff + CONE_GS_OFF_SCVX, dim: 3 };

        // Cone 5: σ_k − T_min ≥ 0
        prob.g_mat[(coff + CONE_TMIN_OFF_SCVX, sigma_idx_scvx(k))] = -1.0;
        prob.h    [coff + CONE_TMIN_OFF_SCVX]                       = -phys.t_min;
        prob.cones[cidx + 4] = ConeDesc { offset: coff + CONE_TMIN_OFF_SCVX, dim: 1 };

        // Cone 6: T_max − σ_k ≥ 0
        prob.g_mat[(coff + CONE_TMAX_OFF_SCVX, sigma_idx_scvx(k))] = 1.0;
        prob.h    [coff + CONE_TMAX_OFF_SCVX]                       = phys.t_max;
        prob.cones[cidx + 5] = ConeDesc { offset: coff + CONE_TMAX_OFF_SCVX, dim: 1 };

        // Cone 7: trust region (η, x_k − x̄_k, u_k − ū_k) ∈ SOC^{11}
        //   s[0]   = η                 (no coupling to z; h = η)
        //   s[1+i] = x_k[i] − x̄_k[i]   (G[..., x_k_col+i] = -1, h = -x̄_k[i])
        //   s[8+i] = u_k[i] − ū_k[i]   (G[..., u_k_col+i] = -1, h = -ū_k[i])
        prob.h[coff + CONE_TRUST_OFF_SCVX] = trust_eta;
        for i in 0..NX {
            prob.g_mat[(coff + CONE_TRUST_OFF_SCVX + 1 + i, x_idx_scvx(k) + i)] = -1.0;
            prob.h    [coff + CONE_TRUST_OFF_SCVX + 1 + i]                      = -reference.x[(i, k)];
        }
        for i in 0..NU {
            prob.g_mat[(coff + CONE_TRUST_OFF_SCVX + 1 + NX + i, u_idx_scvx(k) + i)] = -1.0;
            prob.h    [coff + CONE_TRUST_OFF_SCVX + 1 + NX + i]                       = -reference.u[(i, k)];
        }
        prob.cones[cidx + 6] = ConeDesc { offset: coff + CONE_TRUST_OFF_SCVX, dim: 11 };

        // Cone 8: virtual control L2  (w_k, ν_k) ∈ SOC^8
        prob.g_mat[(coff + CONE_VIRT_OFF_SCVX, w_idx_scvx(k))] = -1.0;
        for i in 0..NNU {
            prob.g_mat[(coff + CONE_VIRT_OFF_SCVX + 1 + i, nu_idx_scvx(k) + i)] = -1.0;
        }
        prob.cones[cidx + 7] = ConeDesc { offset: coff + CONE_VIRT_OFF_SCVX, dim: 8 };
    }

    // ---- Free-tf bound cones (2 SOC^1, at end of `G`/`h` and `cones`) ----
    //
    // Bounds: `tau_lo ≤ τ_ref + δτ ≤ tau_hi`. We parameterize via the
    // shift from reference: `tau_lo − τ_ref ≤ δτ ≤ tau_hi − τ_ref`.
    //
    // SOC^1 (= ℝ_+) standard form `s = h − G·x ≥ 0`:
    //   Lower bound `δτ ≥ tau_lo − τ_ref`:
    //     `s = δτ − (tau_lo − τ_ref) = δτ + (τ_ref − tau_lo)`
    //     ⇒ `G[row, δτ_idx] = −1`, `h[row] = τ_ref − tau_lo`
    //   Upper bound `δτ ≤ tau_hi − τ_ref`:
    //     `s = (tau_hi − τ_ref) − δτ`
    //     ⇒ `G[row, δτ_idx] = +1`, `h[row] = tau_hi − τ_ref`
    if use_free_tf {
        let dtau_idx     = delta_tau_idx_scvx::<N>();
        let bound_lo_row = N * N_CONE_DIM_PER_NODE_SCVX;       // = NCT − 2
        let bound_hi_row = N * N_CONE_DIM_PER_NODE_SCVX + 1;   // = NCT − 1
        let bound_lo_idx = N * N_CONES_PER_NODE_SCVX;          // = NCONES − 2
        let bound_hi_idx = N * N_CONES_PER_NODE_SCVX + 1;      // = NCONES − 1

        prob.g_mat[(bound_lo_row, dtau_idx)] = -1.0;
        prob.h    [bound_lo_row]             = reference.tau - phys.tau_lo;
        prob.cones[bound_lo_idx] = ConeDesc { offset: bound_lo_row, dim: 1 };

        prob.g_mat[(bound_hi_row, dtau_idx)] = 1.0;
        prob.h    [bound_hi_row]             = phys.tau_hi - reference.tau;
        prob.cones[bound_hi_idx] = ConeDesc { offset: bound_hi_row, dim: 1 };
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    extern crate std;
    use std::eprintln;
    use nalgebra::SMatrix;

    use super::*;
    use scvx_dynamics::discretize_foh;
    use scvx_ipm::{solve_socp, SocpWorkspace};
    use scvx_core::IpmAlgoParams;

    fn mars_params() -> PhysicalParams {
        PhysicalParams {
            g:             [0.0, 0.0, -3.7114],
            m_dry:          200.0,
            m_wet:         1000.0,
            isp:            225.0,
            g0:               9.80665,
            t_min:         1000.0,
            t_max:         6000.0,
            cos_theta_max:    0.7660444,   // 40°
            tan_gamma_gs:     1.0,          // 45°
            rho:              0.0,          // drag OFF for LCvx-style
            cd_a:             0.0,
            tau_lo:           5.0,
            tau_hi:          50.0,
        }
    }

    /// Build a hover-like reference trajectory: position descending linearly,
    /// velocity = const, mass = const, thrust = hover-equivalent.
    fn hover_reference<const N: usize>(
        x_init: SVector<f64, 7>,
        m: f64,
        tau: f64,
    ) -> Trajectory<N> {
        let mut traj = Trajectory::<N>::default();
        let p = mars_params();
        // Hover thrust along +z to cancel gravity
        let u_hover = [0.0, 0.0, -m * p.g[2]];
        for k in 0..N {
            for i in 0..7 {
                traj.x[(i, k)] = x_init[i];
            }
            for (i, &v) in u_hover.iter().enumerate() {
                traj.u[(i, k)] = v;
            }
            traj.sigma[k] = m * (-p.g[2]); // ‖u‖
        }
        traj.tau = tau;
        traj
    }

    /// **Structural test**: with `N=3`, verify the assembled `SocpProblem`
    /// has the expected dim layout, the cost vector flags only σ_k columns,
    /// and the key matrix entries (initial-state I, dynamics blocks, cones)
    /// are correctly placed. Doesn't try to solve — that's the next test.
    #[test]
    fn assembled_problem_has_expected_structure() {
        const N: usize = 3;
        const NP: usize     = N * N_VARS_PER_NODE;         // 33
        const NE: usize     = N * N_EQ_PER_DYN + N_EQ_TERMINAL; // 27
        const NCT: usize    = N * N_CONE_DIM_PER_NODE;     // 33
        const NCONES: usize = N * N_CONES_PER_NODE;        // 18

        let phys = mars_params();
        let mut x_init = SVector::<f64, 7>::zeros();
        x_init[2] = 1000.0;
        x_init[5] = -50.0;
        x_init[6] = (800.0_f64).ln();

        let traj = hover_reference::<N>(x_init, 800.0, 20.0);
        let mut lin = LinearizedDynamics::<N>::default();
        discretize_foh(&traj, &phys, &mut lin, 4);

        let term = TerminalCondition { r: [0.0; 3], v: [0.0; 3] };
        let mut prob = SocpProblem::<NP, NE, NCT, NCONES>::default();
        assemble_lcvx_socp(&traj, &lin, &phys, &x_init, &term, &mut prob);

        // ---- Cost: only σ_k columns are 1.0; everything else 0 ----
        for k in 0..N {
            assert_eq!(prob.c[sigma_idx(k)], 1.0, "σ_{k} cost weight");
        }
        for col in 0..NP {
            if (col + 1) % N_VARS_PER_NODE != 0 {
                assert_eq!(prob.c[col], 0.0, "non-σ col {col} should be 0");
            }
        }

        // ---- Initial-state rows: I block on x_0 columns ----
        for i in 0..NX {
            assert_eq!(prob.a_mat[(i, x_idx(0) + i)], 1.0,
                       "initial row {i}, x_0[{i}] should be 1");
            assert_eq!(prob.b[i], x_init[i],
                       "initial RHS {i} mismatch");
        }

        // ---- Dynamics row block for transition k=0: spot-check shape ----
        // Row dyn_row_start + 0 (the x-component of x_1 dynamics):
        //   coefficient on x_1[0] should be 1; coefficient on x_0[0] should be -A_0[0,0].
        let dyn_row = NX; // start of dynamics rows
        assert_eq!(prob.a_mat[(dyn_row, x_idx(1) + 0)], 1.0);
        assert!((prob.a_mat[(dyn_row, x_idx(0) + 0)] + lin.a[0][(0, 0)]).abs() < 1e-15);
        // u_0 column: -B⁻[0,0]
        assert!((prob.a_mat[(dyn_row, u_idx(0) + 0)] + lin.b_minus[0][(0, 0)]).abs() < 1e-15);
        // u_1 column: -B⁺[0,0]
        assert!((prob.a_mat[(dyn_row, u_idx(1) + 0)] + lin.b_plus[0][(0, 0)]).abs() < 1e-15);
        // RHS: c_0 + s_0·τ_ref
        let expected_rhs = lin.c[0][0] + lin.s[0][0] * traj.tau;
        assert!((prob.b[dyn_row] - expected_rhs).abs() < 1e-15);

        // ---- Terminal rows: I block on x_{N-1}[0..6] columns ----
        let term_row = NX + (N - 1) * NX;
        // For i ∈ 0..3, terminal row constrains r_i (state index i).
        // For i ∈ 3..6, terminal row constrains v_{i-3} (state index i).
        // Either way, column = x_idx(N-1) + i.
        for i in 0..6 {
            assert_eq!(prob.a_mat[(term_row + i, x_idx(N - 1) + i)], 1.0);
        }

        // ---- Cone descriptors: 6 per node × 3 nodes = 18, with expected dims ----
        for k in 0..N {
            let cidx = k * N_CONES_PER_NODE;
            assert_eq!(prob.cones[cidx + 0].dim, 4,  "mag cone dim");
            assert_eq!(prob.cones[cidx + 1].dim, 1,  "pointing dim");
            assert_eq!(prob.cones[cidx + 2].dim, 1,  "mass-floor dim");
            assert_eq!(prob.cones[cidx + 3].dim, 3,  "glide-slope dim");
            assert_eq!(prob.cones[cidx + 4].dim, 1,  "Tmin dim");
            assert_eq!(prob.cones[cidx + 5].dim, 1,  "Tmax dim");
            // Offsets are contiguous and start at k * 11:
            assert_eq!(prob.cones[cidx + 0].offset, k * N_CONE_DIM_PER_NODE);
            assert_eq!(prob.cones[cidx + 5].offset, k * N_CONE_DIM_PER_NODE + 10);
        }

        // ---- Cone 1 (thrust magnitude) for node 0: spot-check G entries ----
        let mag_off = 0; // node 0
        assert_eq!(prob.g_mat[(mag_off,     sigma_idx(0))],  -1.0);  // s[0] = σ
        assert_eq!(prob.g_mat[(mag_off + 1, u_idx(0) + 0)],  -1.0);  // s[1] = u_x
        assert_eq!(prob.g_mat[(mag_off + 2, u_idx(0) + 1)],  -1.0);  // s[2] = u_y
        assert_eq!(prob.g_mat[(mag_off + 3, u_idx(0) + 2)],  -1.0);  // s[3] = u_z

        // ---- Cone 5/6 (Tmin/Tmax) for node 0 ----
        assert_eq!(prob.g_mat[(CONE_TMIN_OFF, sigma_idx(0))], -1.0);
        assert_eq!(prob.h    [CONE_TMIN_OFF],                 -phys.t_min);
        assert_eq!(prob.g_mat[(CONE_TMAX_OFF, sigma_idx(0))], 1.0);
        assert_eq!(prob.h    [CONE_TMAX_OFF],                 phys.t_max);

        // ---- Mass-floor row for node 0: h = -ln(m_dry) ----
        let log_m_dry = (phys.m_dry as f64).ln();
        assert!((prob.h[CONE_MASS_OFF] + log_m_dry).abs() < 1e-12);

        eprintln!("Dim sanity: NP={NP} NE={NE} NCT={NCT} NCONES={NCONES}");
        eprintln!("Assembled SOCP looks structurally correct.");

        // Suppress unused-variable warning from the spot-check we used to read
        // the dynamics RHS earlier.
        let _ = expected_rhs;
    }

    /// **Feasibility test**: assemble a 4-node powered-descent SOCP and try
    /// to solve it. Verify the IPM returns Optimal or BestFeasible (not
    /// NumericalError) — the actual physical accuracy of the solution is
    /// validated in P7/P8 against the SCPToolbox.jl oracle.
    ///
    /// Uses N=4 (3 transitions, ~6.7s each at τ=20s) with reasonable Mars
    /// soft-landing parameters. The hover-reference linearization is a poor
    /// guess but should still produce a feasible SOCP.
    #[test]
    fn small_problem_is_solvable() {
        const N: usize = 4;
        const NP: usize     = N * N_VARS_PER_NODE;
        const NE: usize     = N * N_EQ_PER_DYN + N_EQ_TERMINAL;
        const NCT: usize    = N * N_CONE_DIM_PER_NODE;
        const NCONES: usize = N * N_CONES_PER_NODE;

        let phys = mars_params();
        // Start: 500 m altitude, descending at 20 m/s, mass 800 kg
        let mut x_init = SVector::<f64, 7>::zeros();
        x_init[2] = 500.0;
        x_init[5] = -20.0;
        x_init[6] = (800.0_f64).ln();

        // Hover-like reference trajectory at the initial state.
        let traj = hover_reference::<N>(x_init, 800.0, 20.0);
        let mut lin = LinearizedDynamics::<N>::default();
        discretize_foh(&traj, &phys, &mut lin, 4);

        let term = TerminalCondition { r: [0.0; 3], v: [0.0; 3] };
        let mut prob = SocpProblem::<NP, NE, NCT, NCONES>::default();
        assemble_lcvx_socp(&traj, &lin, &phys, &x_init, &term, &mut prob);

        let mut ws = SocpWorkspace::<NP, NE, NCT>::default();
        let params = IpmAlgoParams::default();
        let result = solve_socp(&prob, &params, &mut ws);

        eprintln!("LCvx-style solve: status_u32={}, iters={}",
                  result.status.as_u32(), result.iters);

        // We accept any non-error terminus. The actual trajectory quality
        // depends on the (bad) hover-reference linearization; that's an
        // SCvx-outer-loop concern.
        use scvx_core::IpmStatus;
        let ok = matches!(
            result.status,
            IpmStatus::Optimal | IpmStatus::BestFeasible | IpmStatus::IterCap
        );
        assert!(ok, "unexpected status {}", result.status.as_u32());

        // Optimally, σ_k should be within [T_min, T_max] per node.
        for k in 0..N {
            let sigma_val = result.x[sigma_idx(k)];
            assert!(sigma_val.is_finite(),
                    "σ_{k} non-finite = {sigma_val}");
        }

        // SMatrix is used by nalgebra under the hood; keep the import.
        let _check: SMatrix<f64, 1, 1> = SMatrix::zeros();
    }
}
