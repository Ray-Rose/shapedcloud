//! Diagonal preconditioning for the SCvx subproblem SOCP.
//!
//! Motivation: the SCvx subproblem mixes cones whose magnitudes span ~6
//! orders (trust radius ~ O(1), thrust ~ O(1e3) N). The Nesterov-Todd
//! reduced Hessian `H = Gᵀ·W²·G` inherits this conditioning — eigenvalues
//! spread, the Schur factorization loses precision, and the IPM exits
//! `NumericalError` before centering can settle.
//!
//! Two complementary preconditioning strategies live here:
//!
//! **1. Per-variable column scaling** (`build_scaling_diagonal`,
//! `scale_socp_in_place`). Diagonal `D ≻ 0` on the primal:
//! ```text
//!   x_orig = D · x_scaled
//!   c' = D · c,  A' = A · D,  G' = G · D  (b, h unchanged)
//! ```
//! Cone slacks `s = h − G·x` are **invariant** under this transformation.
//! Useful for keeping `H = G'ᵀ · W² · G' = D · GᵀW²G · D` well-conditioned
//! when primal entries span wide ranges.
//!
//! **2. Per-cone row scaling** (`build_cone_scale_diagonal`,
//! `scale_cone_rows_in_place`). Diagonal `E ≻ 0` on cone slacks:
//! ```text
//!   s_normalized = s / e_c       per cone c
//!   G_c' ← G_c / e_c,  h_c' ← h_c / e_c
//! ```
//! Cone constraint preserved (SOC is positive-homogeneous degree 1).
//! Useful for normalizing slacks across cones — what column scaling can
//! NOT do (slacks are invariant under column scaling). Required for NT
//! convergence when cone slack magnitudes span wide ranges (trust radius
//! ~ O(η) vs thrust ~ O(t_max)).
//!
//! Both strategies commute (row and column scaling are independent
//! operations on different axes). They can be applied together by calling
//! `scale_socp_in_place` and `scale_cone_rows_in_place` in either order.
//!
//! ## Scope
//!
//! These helpers are SCvx-layer concerns. The inner IPM (`scvx-ipm`) stays
//! oblivious — it just solves whichever SocpProblem it's handed. Standalone
//! SOCP tests (oracle_diff, mehrotra regressions) are unaffected.
//!
//! The caller is responsible for: (1) building the diagonals once, (2)
//! scaling the assembled problem in place, (3) seeding the IPM with the
//! scaled-coords warm start, (4) unscaling the solution before any
//! consumer that interprets the primal in physical units.

use nalgebra::SVector;
use scvx_core::PhysicalParams;
use scvx_ipm::SocpProblem;

use crate::assemble::{
    delta_tau_idx_scvx, np_scvx_free_tf, nu_idx_scvx, sigma_idx_scvx, u_idx_scvx,
    w_idx_scvx, x_idx_scvx, N_VARS_PER_NODE_SCVX, NU, NX,
};

/// Minimum scale floor — clamps every diagonal entry to ≥ this value so no
/// component degenerates to zero (which would cause division-by-zero on
/// warm-start re-seed and break the inverse).
const MIN_SCALE: f64 = 1.0;

/// Build the per-variable scaling diagonal `D` for the SCvx subproblem,
/// derived from physical parameters and the initial state.
///
/// Per node `k`, packs scales matching the 19-per-node SCvx layout:
/// ```text
///   z_k = [ x_k(7) ⊕ u_k(3) ⊕ σ_k(1) ⊕ ν_k(7) ⊕ w_k(1) ]
/// ```
///
/// **Scale choices** (all clamped to `≥ MIN_SCALE = 1.0`):
///
/// | Component        | Scale source                                |
/// |------------------|---------------------------------------------|
/// | `r` (3)          | `max(|x_init[0..3]|, 100.0)` — altitude     |
/// | `v` (3)          | `max(|x_init[3..6]|, 10.0)` — velocity      |
/// | `z = ln(m)` (1)  | `1.0` — log-mass naturally O(1)             |
/// | `u` (3)          | `phys.t_max` — thrust magnitude             |
/// | `σ` (1)          | `phys.t_max` — matches `u` (cone coupling)  |
/// | `ν` (7)          | per-component match to `x` scale            |
/// | `w` (1)          | `1.0` — L2 norm of small `ν` is small       |
///
/// **Why u and σ must match:** the thrust-magnitude cone enforces
/// `‖u‖ ≤ σ`. Under per-variable scaling, `(u_orig, σ_orig) ∈ SOC^4` ⇔
/// `(D_u·u_scaled, D_σ·σ_scaled) ∈ SOC^4`. For the scaled iterate to lie
/// in `SOC^4`, we need `D_u = D_σ` (so the cone is invariant under the
/// scaling). Same for ν per-component matching x.
///
/// Returns `D` packed in the 19-per-node layout. Const-generic over `NP`
/// which must equal `19·N` (fixed-tf) or `19·N + 1` (free-tf, with `δτ`
/// scale at the trailing slot).
///
/// **Free-tf δτ scale**: `(tau_hi − tau_lo).max(1.0)`. The δτ variable is
/// bounded by `tau_lo − τ_ref ≤ δτ ≤ tau_hi − τ_ref`, so its magnitude
/// is at most `tau_hi − tau_lo`. Scaling by that range puts the scaled
/// δτ in `[-1, +1]`.
pub fn build_scaling_diagonal<const N: usize, const NP: usize>(
    phys:        &PhysicalParams,
    x_init:      &SVector<f64, 7>,
    use_free_tf: bool,
) -> SVector<f64, NP> {
    if use_free_tf {
        debug_assert_eq!(NP, np_scvx_free_tf(N), "NP must equal np_scvx_free_tf(N) in free-tf mode");
    } else {
        debug_assert_eq!(NP, N * N_VARS_PER_NODE_SCVX, "NP must equal N·N_VARS_PER_NODE_SCVX for SCvx layout");
    }

    // Per-component position scale (max altitude / lateral excursion expected).
    let pos_scale = x_init[0].abs()
        .max(x_init[1].abs())
        .max(x_init[2].abs())
        .max(100.0)
        .max(MIN_SCALE);

    // Per-component velocity scale.
    let vel_scale = x_init[3].abs()
        .max(x_init[4].abs())
        .max(x_init[5].abs())
        .max(10.0)
        .max(MIN_SCALE);

    // Log-mass: small (1-7 range typically). No scaling needed.
    let mass_scale = MIN_SCALE;

    // Thrust scale: use t_max (the upper bound of σ). Fallback to 1.0 if
    // somehow t_max ≤ 0 (caller bug, but defend anyway).
    let thrust_scale = phys.t_max.max(MIN_SCALE);

    // σ matches thrust (cone coupling — see docstring).
    let sigma_scale = thrust_scale;

    let mut d = SVector::<f64, NP>::zeros();
    for k in 0..N {
        // x_k = [r(3), v(3), z(1)]
        let xi = x_idx_scvx(k);
        d[xi    ] = pos_scale;
        d[xi + 1] = pos_scale;
        d[xi + 2] = pos_scale;
        d[xi + 3] = vel_scale;
        d[xi + 4] = vel_scale;
        d[xi + 5] = vel_scale;
        d[xi + 6] = mass_scale;

        // u_k (3) — all components match thrust scale.
        let ui = u_idx_scvx(k);
        for i in 0..NU {
            d[ui + i] = thrust_scale;
        }

        // σ_k (1)
        d[sigma_idx_scvx(k)] = sigma_scale;

        // ν_k (7) — per-component match to x (virtual control absorbs state defect).
        let ni = nu_idx_scvx(k);
        d[ni    ] = pos_scale;
        d[ni + 1] = pos_scale;
        d[ni + 2] = pos_scale;
        d[ni + 3] = vel_scale;
        d[ni + 4] = vel_scale;
        d[ni + 5] = vel_scale;
        d[ni + 6] = mass_scale;

        // w_k (1) — small (L2 norm of small ν is small).
        d[w_idx_scvx(k)] = MIN_SCALE;
    }

    if use_free_tf {
        // δτ at the global index `N·N_VARS_PER_NODE_SCVX` — use the authoritative
        // index helper, not a `N * 19` literal, so it tracks the node layout.
        let dtau_scale = (phys.tau_hi - phys.tau_lo).max(MIN_SCALE);
        d[delta_tau_idx_scvx::<N>()] = dtau_scale;
    }

    debug_assert_eq!(NX + NU + 1 + NX + 1, N_VARS_PER_NODE_SCVX, "SCvx node layout invariant");

    d
}

/// Apply column-wise scaling `D` to the SOCP problem **in place**.
///
/// After this call, the IPM solves for `x_scaled` such that
/// `x_orig = D ⊙ x_scaled` reconstructs the original-coords solution.
///
/// Modifies: `c` (element-wise `*= D`), `a_mat` and `g_mat` (column `j`
/// scaled by `D[j]`). Leaves `b`, `h`, and the cone descriptors alone —
/// they're scaling-invariant.
///
/// **Safety:** every entry of `D` must be strictly positive (a zero
/// component would silently delete that primal variable from the problem).
/// `build_scaling_diagonal` guarantees this via `MIN_SCALE` clamping.
pub fn scale_socp_in_place<
    const NP:     usize,
    const NE:     usize,
    const NCT:    usize,
    const NCONES: usize,
>(
    prob: &mut SocpProblem<NP, NE, NCT, NCONES>,
    d:    &SVector<f64, NP>,
) {
    // c' = D ⊙ c
    for j in 0..NP {
        prob.c[j] *= d[j];
    }

    // A' = A · diag(D): scale column j by D[j]
    for j in 0..NP {
        let scale = d[j];
        for i in 0..NE {
            prob.a_mat[(i, j)] *= scale;
        }
    }

    // G' = G · diag(D): scale column j by D[j]
    for j in 0..NP {
        let scale = d[j];
        for i in 0..NCT {
            prob.g_mat[(i, j)] *= scale;
        }
    }

    // b, h, and cones are scaling-invariant — see module docstring math.
}

/// Convert a scaled-coords solution back to original units:
/// `x_orig = D ⊙ x_scaled` element-wise.
///
/// `x_out` is overwritten; callers can pass a workspace-owned buffer.
pub fn unscale_solution<const NP: usize>(
    x_scaled: &SVector<f64, NP>,
    d:        &SVector<f64, NP>,
    x_out:    &mut SVector<f64, NP>,
) {
    for j in 0..NP {
        x_out[j] = x_scaled[j] * d[j];
    }
}

/// Convert an original-coords vector to scaled coords:
/// `x_scaled = x_orig ⊘ D` element-wise.
///
/// Used to re-seed the IPM warm start after `scale_socp_in_place` has been
/// applied. `x_inout` is divided in place by `D`.
///
/// **Safety:** assumes every entry of `D` is strictly positive (see
/// `scale_socp_in_place` docstring).
pub fn scale_warm_start_in_place<const NP: usize>(
    x_inout: &mut SVector<f64, NP>,
    d:       &SVector<f64, NP>,
) {
    for j in 0..NP {
        x_inout[j] /= d[j];
    }
}

// ===========================================================================
// Per-cone slack rescaling
// ===========================================================================

/// Build the per-cone slack-scale vector `E` for the SCvx subproblem.
///
/// `e[c]` is the positive scale by which cone `c`'s slack vector gets
/// divided: `s_c_normalized = s_c_orig / e[c]`. Since SOC is positive-
/// homogeneous of degree 1, dividing all components of `s_c` by the same
/// positive constant preserves cone membership.
///
/// The SCvx subproblem has 8 cones per node, in this order:
///
/// | # | Cone                                             | Scale source                              |
/// |---|--------------------------------------------------|-------------------------------------------|
/// | 1 | Thrust magnitude `(σ, u) ∈ SOC^4`                | `t_max`                                   |
/// | 2 | Pointing `u_z − cos(θ)·σ ∈ ℝ_+`                  | `t_max`                                   |
/// | 3 | Mass floor `z − ln(m_dry) ∈ ℝ_+`                 | `1`                                       |
/// | 4 | Glide slope `(tan(γ)·r_z, r_x, r_y) ∈ SOC^3`     | `pos_scale`                               |
/// | 5 | T_min `σ − T_min ∈ ℝ_+`                          | `t_max`                                   |
/// | 6 | T_max `T_max − σ ∈ ℝ_+`                          | `t_max`                                   |
/// | 7 | Trust region `(η, x − x̄, u − ū) ∈ SOC^{11}`      | `max(trust_eta, pos_scale, thrust_scale)` |
/// | 8 | Virtual control L2 `(w, ν) ∈ SOC^8`              | `1`                                       |
///
/// **Why trust cone uses `max(trust_eta, pos_scale, thrust_scale)`:** the
/// natural slack magnitude is `trust_eta` (the bar is bounded by `η`),
/// but `prob.h` contains `(η, −x̄, −ū)` with `|x̄|` ~ position scale and
/// `|ū|` ~ thrust scale. The IPM's dual warm-start initializes `ws.y`
/// from `prob.h` directly: with cone-row-scaling by `trust_eta` alone,
/// `ws.y_trust` lands at `(|ū|/trust_eta, 0, ...)` which is enormously
/// imbalanced vs other cones at `(1, 0, ...)`. Scaling by the larger
/// value balances the dual at the cost of a tighter slack — the IPM
/// handles a tight slack better than an imbalanced dual.
///
/// All entries are clamped to `≥ MIN_SCALE = 1.0` so no cone gets a zero
/// or negative scale (which would cause division-by-zero or flip the cone
/// orientation).
///
/// **Note: trust scale depends on `trust_eta`** which changes per outer
/// SCvx iteration. The caller passes the current `trust_eta` so the
/// per-cone scale is correct for *this* iteration's assembled SOCP.
pub fn build_cone_scale_diagonal<const N: usize, const NCONES: usize>(
    phys:        &PhysicalParams,
    x_init:      &SVector<f64, 7>,
    trust_eta:   f64,
    use_free_tf: bool,
) -> SVector<f64, NCONES> {
    if use_free_tf {
        debug_assert_eq!(NCONES, 8 * N + 2, "NCONES must equal 8·N + 2 in free-tf mode");
    } else {
        debug_assert_eq!(NCONES, 8 * N, "NCONES must equal 8·N for SCvx layout");
    }

    let pos_scale = x_init[0].abs()
        .max(x_init[1].abs())
        .max(x_init[2].abs())
        .max(100.0)
        .max(MIN_SCALE);

    let thrust_scale = phys.t_max.max(MIN_SCALE);

    // Trust scale tracks the natural slack magnitude: at the optimum
    // (x ≈ x̄, u ≈ ū), the slack is `(η, 0, ..., 0)` so |slack| ~ η.
    // Using `trust_eta` keeps the normalized trust slack at ~unit
    // magnitude. An alternative `max(trust_eta, pos_scale, thrust_scale)`
    // balances the dual warm-start better but breaks AHO convergence
    // empirically; tradeoff favors small `e_trust` here.
    let trust_scale = trust_eta.max(MIN_SCALE);

    let mut e = SVector::<f64, NCONES>::zeros();
    for k in 0..N {
        let base = k * 8;
        e[base    ] = thrust_scale; // 1: thrust mag (D=4)
        e[base + 1] = thrust_scale; // 2: pointing (D=1)
        e[base + 2] = MIN_SCALE;    // 3: mass floor (D=1)
        e[base + 3] = pos_scale;    // 4: glide slope (D=3)
        e[base + 4] = thrust_scale; // 5: T_min (D=1)
        e[base + 5] = thrust_scale; // 6: T_max (D=1)
        e[base + 6] = trust_scale;  // 7: trust region (D=11)
        e[base + 7] = MIN_SCALE;    // 8: virt control L2 (D=8)
    }
    if use_free_tf {
        // Free-tf bound cones: both slacks have magnitude bounded by
        // `tau_hi − tau_lo`. Scale by that range.
        let dtau_range = (phys.tau_hi - phys.tau_lo).max(MIN_SCALE);
        e[N * 8    ] = dtau_range; // 9: δτ lower bound
        e[N * 8 + 1] = dtau_range; // 10: δτ upper bound
    }
    e
}

/// Apply per-cone slack-scaling **in place**: for each cone `c`, divide
/// the rows of `G` and `h` belonging to that cone by `e[c]`. The
/// in-cone-block primal columns are unchanged (column scaling is a
/// separate operation; see `scale_socp_in_place`).
///
/// After this call, the IPM operates on cones whose slacks are normalized
/// to ~unit magnitude (assuming `build_cone_scale_diagonal` was used).
/// This is what `arrow(s)^{-1}` blow-up at the boundary really cares
/// about — and what primal column scaling alone cannot fix (slacks are
/// invariant under primal scaling).
///
/// **Safety:** every entry of `e` must be strictly positive.
/// `build_cone_scale_diagonal` guarantees this via `MIN_SCALE` clamping.
pub fn scale_cone_rows_in_place<
    const NP:     usize,
    const NE:     usize,
    const NCT:    usize,
    const NCONES: usize,
>(
    prob: &mut SocpProblem<NP, NE, NCT, NCONES>,
    e:    &SVector<f64, NCONES>,
) {
    for c in 0..NCONES {
        let cone   = prob.cones[c];
        let scale  = e[c].max(MIN_SCALE);
        let inv_e  = 1.0 / scale;
        for r in cone.offset..(cone.offset + cone.dim) {
            // Row r of G: divide every column entry by e[c].
            for j in 0..NP {
                prob.g_mat[(r, j)] *= inv_e;
            }
            // Row r of h: divide by e[c].
            prob.h[r] *= inv_e;
        }
    }

    // The `prob.cones` descriptors (offsets and dims) are unchanged.
    // `c`, `A`, `b` are unchanged. Only G rows and h entries on selected
    // rows are modified.
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    extern crate std;
    use std::eprintln;
    use nalgebra::{SMatrix, SVector};

    use super::*;
    use scvx_ipm::{solve_socp, ConeDesc, SocpWorkspace};
    use scvx_core::{IpmAlgoParams, IpmStatus};

    fn mars_params() -> PhysicalParams {
        PhysicalParams {
            g:             [0.0, 0.0, -3.7114],
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

    /// The scaling diagonal must be strictly positive everywhere (else
    /// downstream division blows up).
    #[test]
    fn diagonal_is_strictly_positive() {
        const N: usize = 3;
        const NP: usize = 19 * N;
        let phys = mars_params();
        let mut x_init = SVector::<f64, 7>::zeros();
        x_init[2] = 500.0;
        x_init[5] = -20.0;
        x_init[6] = (800.0_f64).ln();

        let d = build_scaling_diagonal::<N, NP>(&phys, &x_init, false);
        for i in 0..NP {
            assert!(d[i] > 0.0, "d[{i}] = {} is non-positive", d[i]);
            assert!(d[i].is_finite(), "d[{i}] = {} non-finite", d[i]);
        }
    }

    /// Scale choices follow the documented table: pos=100, vel=10, mass=1,
    /// thrust=t_max=6000, ν matches x, w=1.
    #[test]
    fn diagonal_matches_documented_scale_table() {
        const N: usize = 2;
        const NP: usize = 19 * N;
        let phys = mars_params();
        // x_init smaller than the defaults so the floor kicks in.
        let x_init = SVector::<f64, 7>::zeros();
        let d = build_scaling_diagonal::<N, NP>(&phys, &x_init, false);

        // Node 0 layout: x(7) at [0..7], u(3) at [7..10], σ(1) at [10],
        // ν(7) at [11..18], w(1) at [18].
        for k in 0..N {
            let base = k * 19;
            // Position floor
            for i in 0..3 {
                assert_eq!(d[base + i], 100.0, "pos[{i}] node {k}");
            }
            // Velocity floor
            for i in 0..3 {
                assert_eq!(d[base + 3 + i], 10.0, "vel[{i}] node {k}");
            }
            // Log-mass
            assert_eq!(d[base + 6], 1.0, "mass node {k}");
            // Thrust (= t_max)
            for i in 0..3 {
                assert_eq!(d[base + 7 + i], phys.t_max, "thrust[{i}] node {k}");
            }
            // σ (= t_max)
            assert_eq!(d[base + 10], phys.t_max, "sigma node {k}");
            // ν matches x scales (pos, vel, mass) — same indexing within the
            // 7-component block.
            for i in 0..3 {
                assert_eq!(d[base + 11 + i], 100.0, "nu_r[{i}] node {k}");
            }
            for i in 0..3 {
                assert_eq!(d[base + 14 + i], 10.0, "nu_v[{i}] node {k}");
            }
            assert_eq!(d[base + 17], 1.0, "nu_m node {k}");
            // w
            assert_eq!(d[base + 18], 1.0, "w node {k}");
        }
    }

    /// `x_init` large enough that the floor doesn't kick in — verify the
    /// per-component max actually controls.
    #[test]
    fn diagonal_uses_x_init_when_large() {
        const N: usize = 1;
        const NP: usize = 19 * N;
        let phys = mars_params();
        let mut x_init = SVector::<f64, 7>::zeros();
        x_init[2] = 5000.0; // way above 100m floor
        x_init[3] = -250.0; // way above 10 m/s floor
        let d = build_scaling_diagonal::<N, NP>(&phys, &x_init, false);

        // All three position entries take the max (5000.0).
        for i in 0..3 {
            assert_eq!(d[i], 5000.0);
        }
        // All three velocity entries take the max (250.0).
        for i in 0..3 {
            assert_eq!(d[3 + i], 250.0);
        }
    }

    /// Free-tf scale diagonal: `NP = 19·N + 1`, last entry is `δτ`
    /// scaled by `(tau_hi - tau_lo)`. First 19·N entries match the
    /// fixed-tf diagonal exactly.
    #[test]
    fn diagonal_free_tf_appends_dtau_scale() {
        const N: usize = 2;
        const NP_FIXED: usize = 19 * N;
        const NP_FREE:  usize = 19 * N + 1;
        let phys = mars_params();
        // x_init small so the per-component max doesn't kick in.
        let x_init = SVector::<f64, 7>::zeros();

        let d_fixed = build_scaling_diagonal::<N, NP_FIXED>(&phys, &x_init, false);
        let d_free  = build_scaling_diagonal::<N, NP_FREE>(&phys, &x_init, true);

        // First 19·N entries are identical between fixed-tf and free-tf.
        for i in 0..(19 * N) {
            assert_eq!(d_fixed[i], d_free[i],
                       "entry {i} differs between fixed and free-tf diagonals");
        }
        // Last entry of free-tf is `tau_hi - tau_lo`.
        let want = (phys.tau_hi - phys.tau_lo).max(1.0);
        assert!((d_free[19 * N] - want).abs() < 1e-15,
                "δτ scale = {}, want {}", d_free[19 * N], want);
    }

    /// Free-tf scale floors to `MIN_SCALE = 1.0` if `tau_hi - tau_lo < 1`.
    #[test]
    fn diagonal_free_tf_dtau_scale_floors_to_min_scale() {
        const N: usize = 1;
        const NP: usize = 19 * N + 1;
        // Narrow τ band: `tau_hi - tau_lo = 0.5` < MIN_SCALE = 1.
        let phys = PhysicalParams {
            tau_lo: 10.0,
            tau_hi: 10.5,
            ..mars_params()
        };
        let x_init = SVector::<f64, 7>::zeros();
        let d = build_scaling_diagonal::<N, NP>(&phys, &x_init, true);
        assert_eq!(d[19 * N], 1.0,
                   "δτ scale should floor to MIN_SCALE = 1.0, got {}", d[19 * N]);
    }

    /// Free-tf cone scale: `NCONES = 8·N + 2`, last two entries are
    /// `(tau_hi - tau_lo)` (the bound-cone scales).
    #[test]
    fn cone_scale_free_tf_appends_bound_cone_scales() {
        const N: usize = 2;
        const NCONES_FIXED: usize = 8 * N;
        const NCONES_FREE:  usize = 8 * N + 2;
        let phys = mars_params();
        let x_init = SVector::<f64, 7>::zeros();
        let trust_eta = 5.0;

        let e_fixed = build_cone_scale_diagonal::<N, NCONES_FIXED>(
            &phys, &x_init, trust_eta, false,
        );
        let e_free = build_cone_scale_diagonal::<N, NCONES_FREE>(
            &phys, &x_init, trust_eta, true,
        );

        // First 8·N entries match exactly.
        for i in 0..(8 * N) {
            assert_eq!(e_fixed[i], e_free[i],
                       "cone entry {i} differs between fixed and free-tf");
        }
        // Last two entries are `(tau_hi - tau_lo).max(1.0)`.
        let want = (phys.tau_hi - phys.tau_lo).max(1.0);
        assert!((e_free[8 * N    ] - want).abs() < 1e-15,
                "lower-bound cone scale = {}, want {}", e_free[8 * N], want);
        assert!((e_free[8 * N + 1] - want).abs() < 1e-15,
                "upper-bound cone scale = {}, want {}", e_free[8 * N + 1], want);
    }

    /// `A' = A·diag(D)` element-wise: column `j` of `A'` is `D[j] · column j of A`.
    #[test]
    fn scale_socp_in_place_matches_column_scaling() {
        const NP: usize = 4;
        const NE: usize = 2;
        const NCT: usize = 3;
        const NCONES: usize = 1;

        let mut prob = SocpProblem::<NP, NE, NCT, NCONES> {
            c:     SVector::<f64, 4>::from_column_slice(&[1.0, 2.0, 3.0, 4.0]),
            a_mat: SMatrix::<f64, 2, 4>::from_row_slice(&[
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
            ]),
            b:     SVector::<f64, 2>::from_column_slice(&[10.0, 20.0]),
            g_mat: SMatrix::<f64, 3, 4>::from_row_slice(&[
                -1.0,  0.0,  0.0,  0.0,
                 0.0, -2.0,  0.0,  0.0,
                 0.0,  0.0, -3.0,  0.0,
            ]),
            h:     SVector::<f64, 3>::from_column_slice(&[0.5, 1.5, 2.5]),
            cones: [ConeDesc { offset: 0, dim: 3 }],
        };

        let c_orig = prob.c;
        let a_orig = prob.a_mat;
        let g_orig = prob.g_mat;
        let b_orig = prob.b;
        let h_orig = prob.h;

        let d = SVector::<f64, 4>::from_column_slice(&[10.0, 100.0, 1000.0, 0.1]);
        scale_socp_in_place(&mut prob, &d);

        // c' = D ⊙ c
        for j in 0..NP {
            assert!((prob.c[j] - c_orig[j] * d[j]).abs() < 1e-15);
        }
        // A'[i,j] = A[i,j] · D[j]
        for i in 0..NE {
            for j in 0..NP {
                let want = a_orig[(i, j)] * d[j];
                assert!((prob.a_mat[(i, j)] - want).abs() < 1e-15,
                        "A'[{i},{j}] = {}, want {}", prob.a_mat[(i, j)], want);
            }
        }
        // G'[i,j] = G[i,j] · D[j]
        for i in 0..NCT {
            for j in 0..NP {
                let want = g_orig[(i, j)] * d[j];
                assert!((prob.g_mat[(i, j)] - want).abs() < 1e-15,
                        "G'[{i},{j}] = {}, want {}", prob.g_mat[(i, j)], want);
            }
        }
        // b, h unchanged
        for i in 0..NE {
            assert_eq!(prob.b[i], b_orig[i]);
        }
        for i in 0..NCT {
            assert_eq!(prob.h[i], h_orig[i]);
        }
    }

    /// `unscale_solution(D, x_scaled)` recovers `x_orig` from
    /// `x_scaled = x_orig / D`.
    #[test]
    fn unscale_inverts_warm_start_scaling() {
        const NP: usize = 5;
        let d        = SVector::<f64, NP>::from_column_slice(&[1.0, 10.0, 100.0, 1.0, 1000.0]);
        let x_orig   = SVector::<f64, NP>::from_column_slice(&[3.0, 50.0, 250.0, -2.0, 7500.0]);

        let mut x_scaled = x_orig;
        scale_warm_start_in_place(&mut x_scaled, &d);
        // x_scaled = x_orig / D
        for j in 0..NP {
            assert!((x_scaled[j] - x_orig[j] / d[j]).abs() < 1e-15);
        }

        let mut x_recovered = SVector::<f64, NP>::zeros();
        unscale_solution(&x_scaled, &d, &mut x_recovered);
        for j in 0..NP {
            assert!((x_recovered[j] - x_orig[j]).abs() < 1e-12,
                    "x[{j}] = {} vs {}", x_recovered[j], x_orig[j]);
        }
    }

    /// **The headline test**: solve the same SOCP twice — once raw, once
    /// scaled — and verify the optimum and cost agree. Cone constraints
    /// must remain satisfied after unscaling.
    ///
    /// Problem (the 2-cone test from socp.rs):
    ///   min x_3 s.t. x_1 = 1, x_2 = 1, (x_3, x_1, x_2) ∈ SOC^3, x_3 ≥ 2.
    /// Expected: x = (1, 1, 2).
    #[test]
    fn scaled_problem_recovers_same_optimum() {
        const NP: usize     = 3;
        const NE: usize     = 2;
        const NCT: usize    = 4;
        const NCONES: usize = 2;

        let build = || SocpProblem::<NP, NE, NCT, NCONES> {
            c:     SVector::<f64, 3>::from_column_slice(&[0.0, 0.0, 1.0]),
            a_mat: SMatrix::<f64, 2, 3>::from_row_slice(&[
                1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
            ]),
            b:     SVector::<f64, 2>::from_column_slice(&[1.0, 1.0]),
            g_mat: SMatrix::<f64, 4, 3>::from_row_slice(&[
                 0.0,  0.0, -1.0,
                -1.0,  0.0,  0.0,
                 0.0, -1.0,  0.0,
                 0.0,  0.0, -1.0,
            ]),
            h:     SVector::<f64, 4>::from_column_slice(&[0.0, 0.0, 0.0, -2.0]),
            cones: [
                ConeDesc { offset: 0, dim: 3 },
                ConeDesc { offset: 3, dim: 1 },
            ],
        };

        // Unscaled baseline.
        let prob_raw = build();
        let params = IpmAlgoParams::default();
        let mut ws_raw = SocpWorkspace::<NP, NE, NCT>::default();
        let result_raw = solve_socp(&prob_raw, &params, &mut ws_raw);
        assert!(matches!(result_raw.status, IpmStatus::Optimal | IpmStatus::BestFeasible));

        // Scaled: deliberately mismatched per-variable scales (test that the
        // math is right even when the scaling is "wrong" for conditioning).
        let mut prob_scaled = build();
        let d = SVector::<f64, NP>::from_column_slice(&[10.0, 100.0, 1000.0]);
        scale_socp_in_place(&mut prob_scaled, &d);
        let mut ws_scaled = SocpWorkspace::<NP, NE, NCT>::default();
        let result_scaled = solve_socp(&prob_scaled, &params, &mut ws_scaled);
        assert!(matches!(result_scaled.status, IpmStatus::Optimal | IpmStatus::BestFeasible),
                "scaled solve failed: status code {}", result_scaled.status.as_u32());

        // Unscale.
        let mut x_recovered = SVector::<f64, NP>::zeros();
        unscale_solution(&result_scaled.x, &d, &mut x_recovered);

        // Cost agreement (within IPM tolerance ~1e-4 on this problem).
        let cost_raw = prob_raw.c.dot(&result_raw.x);
        // For the scaled cost in original coords, we need c_orig·x_orig,
        // which we recompute from the build closure since prob_scaled.c
        // has been overwritten.
        let prob_for_c = build();
        let cost_unscaled = prob_for_c.c.dot(&x_recovered);
        eprintln!("cost raw     = {:.6e}", cost_raw);
        eprintln!("cost scaled  = {:.6e} (= c'·x_scaled)", prob_scaled.c.dot(&result_scaled.x));
        eprintln!("cost unscaled = {:.6e} (= c·x_recovered)", cost_unscaled);

        // The invariance `c·x = c'·x_scaled` should hold to ~machine precision.
        let cost_invariant = prob_scaled.c.dot(&result_scaled.x);
        assert!((cost_invariant - cost_unscaled).abs() < 1e-10,
                "cost invariance broken: {} vs {}", cost_invariant, cost_unscaled);

        // The two IPM runs should agree on the cost to IPM precision.
        assert!((cost_raw - cost_unscaled).abs() < 1.0e-3,
                "raw cost {} vs scaled cost {}", cost_raw, cost_unscaled);

        // Coordinate-wise agreement on the optimum.
        for j in 0..NP {
            assert!((result_raw.x[j] - x_recovered[j]).abs() < 1.0e-3,
                    "x[{j}]: raw = {}, scaled-unscaled = {}",
                    result_raw.x[j], x_recovered[j]);
        }
    }

    // ---------------------------------------------------------------------
    // Cone-row-scaling tests
    // ---------------------------------------------------------------------

    /// Cone scale diagonal must be strictly positive everywhere.
    #[test]
    fn cone_scale_diagonal_is_strictly_positive() {
        const N: usize = 3;
        const NCONES: usize = 8 * N;
        let phys = mars_params();
        let mut x_init = SVector::<f64, 7>::zeros();
        x_init[2] = 500.0;
        x_init[5] = -20.0;

        let e = build_cone_scale_diagonal::<N, NCONES>(&phys, &x_init, 5.0, false);
        for i in 0..NCONES {
            assert!(e[i] > 0.0, "e[{i}] = {} non-positive", e[i]);
            assert!(e[i].is_finite(), "e[{i}] = {} non-finite", e[i]);
        }
    }

    /// Cone scale per-cone-type matches the documented table (pos for
    /// glide, thrust for thrust-coupled cones, 1 for mass/virt, trust_eta
    /// for the trust cone).
    #[test]
    fn cone_scale_matches_documented_table() {
        const N: usize = 2;
        const NCONES: usize = 8 * N;
        let phys = mars_params();
        let x_init = SVector::<f64, 7>::zeros();
        let trust_eta = 5.0;

        let e = build_cone_scale_diagonal::<N, NCONES>(&phys, &x_init, trust_eta, false);

        for k in 0..N {
            let base = k * 8;
            assert_eq!(e[base    ], phys.t_max, "thrust mag node {k}");
            assert_eq!(e[base + 1], phys.t_max, "pointing node {k}");
            assert_eq!(e[base + 2], 1.0,        "mass floor node {k}");
            assert_eq!(e[base + 3], 100.0,      "glide slope node {k} (default floor)");
            assert_eq!(e[base + 4], phys.t_max, "T_min node {k}");
            assert_eq!(e[base + 5], phys.t_max, "T_max node {k}");
            assert_eq!(e[base + 6], trust_eta,  "trust node {k}");
            assert_eq!(e[base + 7], 1.0,        "virt ctrl node {k}");
        }
    }

    /// `trust_eta ≤ 0` (caller bug) gets clamped to `MIN_SCALE = 1.0` so
    /// division by zero is impossible downstream.
    #[test]
    fn cone_scale_clamps_trust_eta_to_min_scale() {
        const N: usize = 1;
        const NCONES: usize = 8;
        let phys = mars_params();
        let x_init = SVector::<f64, 7>::zeros();

        // Trust eta of 0 (clearly degenerate caller input).
        let e_zero = build_cone_scale_diagonal::<N, NCONES>(&phys, &x_init, 0.0, false);
        assert_eq!(e_zero[6], 1.0, "trust scale must clamp to MIN_SCALE on zero input");

        // Trust eta negative (even more pathological).
        let e_neg = build_cone_scale_diagonal::<N, NCONES>(&phys, &x_init, -100.0, false);
        assert_eq!(e_neg[6], 1.0, "trust scale must clamp on negative input");
    }

    /// `scale_cone_rows_in_place` divides each cone's G rows and h entries
    /// by the corresponding `e[c]`, leaving everything else untouched.
    #[test]
    fn scale_cone_rows_matches_per_cone_row_division() {
        const NP:     usize = 4;
        const NE:     usize = 2;
        const NCT:    usize = 5;
        const NCONES: usize = 2;

        let mut prob = SocpProblem::<NP, NE, NCT, NCONES> {
            c:     SVector::<f64, 4>::from_column_slice(&[1.0, 2.0, 3.0, 4.0]),
            a_mat: SMatrix::<f64, 2, 4>::from_row_slice(&[
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
            ]),
            b:     SVector::<f64, 2>::from_column_slice(&[10.0, 20.0]),
            // 5 cone rows total: cone 0 (rows 0..3, SOC^3), cone 1 (rows 3..5, SOC^2)
            g_mat: SMatrix::<f64, 5, 4>::from_row_slice(&[
                -1.0,  0.0,  0.0,  0.0,
                 0.0, -2.0,  0.0,  0.0,
                 0.0,  0.0, -3.0,  0.0,
                 0.0,  0.0,  0.0, -4.0,
                 0.0,  0.0,  0.0, -5.0,
            ]),
            h:     SVector::<f64, 5>::from_column_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]),
            cones: [
                ConeDesc { offset: 0, dim: 3 },
                ConeDesc { offset: 3, dim: 2 },
            ],
        };

        let c_orig = prob.c;
        let a_orig = prob.a_mat;
        let g_orig = prob.g_mat;
        let b_orig = prob.b;
        let h_orig = prob.h;

        // Pick distinct cone scales so any swap or off-by-one shows up.
        let e = SVector::<f64, NCONES>::from_column_slice(&[10.0, 100.0]);
        scale_cone_rows_in_place(&mut prob, &e);

        // c, A, b unchanged.
        for j in 0..NP {
            assert_eq!(prob.c[j], c_orig[j]);
        }
        for i in 0..NE {
            for j in 0..NP {
                assert_eq!(prob.a_mat[(i, j)], a_orig[(i, j)]);
            }
            assert_eq!(prob.b[i], b_orig[i]);
        }
        // G[rows 0..3] /= e[0] = 10; h[0..3] /= e[0]
        for r in 0..3 {
            for j in 0..NP {
                let want = g_orig[(r, j)] / 10.0;
                assert!((prob.g_mat[(r, j)] - want).abs() < 1e-15,
                        "G[{r},{j}] = {} want {}", prob.g_mat[(r, j)], want);
            }
            let want_h = h_orig[r] / 10.0;
            assert!((prob.h[r] - want_h).abs() < 1e-15,
                    "h[{r}] = {} want {}", prob.h[r], want_h);
        }
        // G[rows 3..5] /= e[1] = 100; h[3..5] /= e[1]
        for r in 3..5 {
            for j in 0..NP {
                let want = g_orig[(r, j)] / 100.0;
                assert!((prob.g_mat[(r, j)] - want).abs() < 1e-15,
                        "G[{r},{j}] = {} want {}", prob.g_mat[(r, j)], want);
            }
            let want_h = h_orig[r] / 100.0;
            assert!((prob.h[r] - want_h).abs() < 1e-15,
                    "h[{r}] = {} want {}", prob.h[r], want_h);
        }
    }

    /// **The headline cone-row test**: solve the same SOCP twice (raw and
    /// cone-row-scaled) and verify the optimum and cost agree.
    #[test]
    fn cone_row_scaled_recovers_same_optimum() {
        const NP: usize     = 3;
        const NE: usize     = 2;
        const NCT: usize    = 4;
        const NCONES: usize = 2;

        let build = || SocpProblem::<NP, NE, NCT, NCONES> {
            c:     SVector::<f64, 3>::from_column_slice(&[0.0, 0.0, 1.0]),
            a_mat: SMatrix::<f64, 2, 3>::from_row_slice(&[
                1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
            ]),
            b:     SVector::<f64, 2>::from_column_slice(&[1.0, 1.0]),
            g_mat: SMatrix::<f64, 4, 3>::from_row_slice(&[
                 0.0,  0.0, -1.0,
                -1.0,  0.0,  0.0,
                 0.0, -1.0,  0.0,
                 0.0,  0.0, -1.0,
            ]),
            h:     SVector::<f64, 4>::from_column_slice(&[0.0, 0.0, 0.0, -2.0]),
            cones: [
                ConeDesc { offset: 0, dim: 3 },
                ConeDesc { offset: 3, dim: 1 },
            ],
        };

        // Unscaled baseline.
        let prob_raw = build();
        let params = IpmAlgoParams::default();
        let mut ws_raw = SocpWorkspace::<NP, NE, NCT>::default();
        let result_raw = solve_socp(&prob_raw, &params, &mut ws_raw);
        assert!(matches!(result_raw.status, IpmStatus::Optimal | IpmStatus::BestFeasible));

        // Cone-row scaling: distinct scales per cone.
        let mut prob_scaled = build();
        let e = SVector::<f64, NCONES>::from_column_slice(&[5.0, 50.0]);
        scale_cone_rows_in_place(&mut prob_scaled, &e);

        let mut ws_scaled = SocpWorkspace::<NP, NE, NCT>::default();
        let result_scaled = solve_socp(&prob_scaled, &params, &mut ws_scaled);
        assert!(matches!(result_scaled.status, IpmStatus::Optimal | IpmStatus::BestFeasible));

        // Primal `x` is NOT affected by cone-row scaling (we didn't touch
        // c, A, or any primal column), so the two runs should give the
        // same `x` to IPM precision.
        let cost_raw    = prob_raw.c.dot(&result_raw.x);
        let cost_scaled = prob_scaled.c.dot(&result_scaled.x);
        eprintln!("cost raw       = {:.6e}", cost_raw);
        eprintln!("cost cone-row  = {:.6e}", cost_scaled);
        assert!((cost_raw - cost_scaled).abs() < 1.0e-3,
                "raw cost {} vs cone-row-scaled cost {}",
                cost_raw, cost_scaled);
        for j in 0..NP {
            assert!((result_raw.x[j] - result_scaled.x[j]).abs() < 1.0e-3,
                    "x[{j}]: raw = {}, scaled = {}",
                    result_raw.x[j], result_scaled.x[j]);
        }
    }

    /// **Composition test**: column scaling AND cone-row scaling applied
    /// together still recover the same primal optimum (after unscaling).
    /// Verifies the two transformations commute and stay consistent.
    #[test]
    fn full_preconditioning_recovers_same_optimum() {
        const NP: usize     = 3;
        const NE: usize     = 1;
        const NCT: usize    = 3;
        const NCONES: usize = 1;

        // The toy SOCP: min x_1 s.t. x_2 + x_3 = 1, (x_1, x_2, x_3) ∈ SOC^3.
        // Optimum: x ≈ (1/√2, 0.5, 0.5).
        let build = || SocpProblem::<NP, NE, NCT, NCONES> {
            c:     SVector::<f64, 3>::from_column_slice(&[1.0, 0.0, 0.0]),
            a_mat: SMatrix::<f64, 1, 3>::from_row_slice(&[0.0, 1.0, 1.0]),
            b:     SVector::<f64, 1>::from_element(1.0),
            g_mat: -SMatrix::<f64, 3, 3>::identity(),
            h:     SVector::<f64, 3>::zeros(),
            cones: [ConeDesc { offset: 0, dim: 3 }],
        };

        let prob_raw = build();
        let params = IpmAlgoParams::default();
        let mut ws_raw = SocpWorkspace::<NP, NE, NCT>::default();
        let result_raw = solve_socp(&prob_raw, &params, &mut ws_raw);
        assert!(matches!(result_raw.status, IpmStatus::Optimal | IpmStatus::BestFeasible));

        // Apply BOTH transformations.
        let mut prob_scaled = build();
        let d = SVector::<f64, NP>::from_column_slice(&[10.0, 100.0, 1000.0]);
        let e = SVector::<f64, NCONES>::from_column_slice(&[7.0]);
        scale_socp_in_place(&mut prob_scaled, &d);
        scale_cone_rows_in_place(&mut prob_scaled, &e);

        let mut ws_scaled = SocpWorkspace::<NP, NE, NCT>::default();
        let result_scaled = solve_socp(&prob_scaled, &params, &mut ws_scaled);
        assert!(matches!(result_scaled.status, IpmStatus::Optimal | IpmStatus::BestFeasible),
                "doubly-scaled solve failed: status {}", result_scaled.status.as_u32());

        // Unscale the primal back to original coords (only column scaling
        // affects the primal — cone-row scaling leaves it alone).
        let mut x_recovered = SVector::<f64, NP>::zeros();
        unscale_solution(&result_scaled.x, &d, &mut x_recovered);

        let cost_raw    = prob_raw.c.dot(&result_raw.x);
        let prob_for_c  = build();
        let cost_double = prob_for_c.c.dot(&x_recovered);
        eprintln!("cost raw           = {:.6e}", cost_raw);
        eprintln!("cost full-precond  = {:.6e}", cost_double);
        assert!((cost_raw - cost_double).abs() < 1.0e-3,
                "raw cost {} vs full-preconditioned cost {}", cost_raw, cost_double);

        for j in 0..NP {
            assert!((result_raw.x[j] - x_recovered[j]).abs() < 1.0e-3,
                    "x[{j}]: raw = {}, full-preconditioned = {}",
                    result_raw.x[j], x_recovered[j]);
        }
    }

    /// **Defense**: scaling preserves both equality (`A·x = b`) and cone
    /// (`G·x + s = h`) feasibility — when applied to a feasible point.
    #[test]
    fn scaling_preserves_feasibility() {
        const NP: usize     = 3;
        const NE: usize     = 1;
        const NCT: usize    = 3;
        const NCONES: usize = 1;

        let mut prob = SocpProblem::<NP, NE, NCT, NCONES> {
            c:     SVector::<f64, 3>::from_column_slice(&[1.0, 0.0, 0.0]),
            a_mat: SMatrix::<f64, 1, 3>::from_row_slice(&[0.0, 1.0, 1.0]),
            b:     SVector::<f64, 1>::from_element(1.0),
            g_mat: -SMatrix::<f64, 3, 3>::identity(),
            h:     SVector::<f64, 3>::zeros(),
            cones: [ConeDesc { offset: 0, dim: 3 }],
        };

        // x = (√0.5, 0.5, 0.5) is the optimum of this SOCP — feasible and
        // on the cone boundary in (x_1, x_2).
        let inv_sqrt2: f64 = (0.5_f64).sqrt();
        let x_orig = SVector::<f64, 3>::from_column_slice(&[inv_sqrt2, 0.5, 0.5]);

        // Verify feasibility in original coords.
        let a_x_orig    = prob.a_mat * x_orig;
        let s_orig      = prob.h - prob.g_mat * x_orig;
        assert!((a_x_orig[0] - prob.b[0]).abs() < 1e-12);
        assert!(s_orig[0] >= (s_orig[1].powi(2) + s_orig[2].powi(2)).sqrt() - 1e-12);

        // Apply scaling.
        let d = SVector::<f64, 3>::from_column_slice(&[10.0, 100.0, 1000.0]);
        scale_socp_in_place(&mut prob, &d);

        // Compute x_scaled = x_orig / D.
        let mut x_scaled = x_orig;
        scale_warm_start_in_place(&mut x_scaled, &d);

        // Equality: A'·x_scaled = A·D·(x_orig/D) = A·x_orig = b. ✓
        let a_scaled_x  = prob.a_mat * x_scaled;
        assert!((a_scaled_x[0] - prob.b[0]).abs() < 1e-12,
                "equality broken: {} vs {}", a_scaled_x[0], prob.b[0]);

        // Cone: s_scaled = h - G'·x_scaled = h - G·x_orig = s_orig. ✓
        let s_scaled = prob.h - prob.g_mat * x_scaled;
        for i in 0..NCT {
            assert!((s_scaled[i] - s_orig[i]).abs() < 1e-12,
                    "s[{i}] changed: {} vs {}", s_scaled[i], s_orig[i]);
        }
    }
}
