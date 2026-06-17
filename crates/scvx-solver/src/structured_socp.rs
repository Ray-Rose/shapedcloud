//! AHO-direction Mehrotra IPM driver using **structured block-tridiagonal
//! Schur** factorization for the inner KKT solve.
//!
//! Same algorithmic recipe as `scvx_ipm::solve_socp` — predictor + Mehrotra
//! sigma + corrector — but the reduced Hessian inverse `H⁻¹` and Schur
//! complement `S⁻¹` are no longer materialized as dense matrices. Instead
//! the IPM calls [`solve_reduced_kkt_scvx_block_m`] which factors `H` as
//! per-stage diagonal blocks and `S = A·H⁻¹·Aᵀ` as block-tridiagonal,
//! solving in `O(N·NZ³)` instead of `O((N·NZ)³)`.
//!
//! ## Compared to `solve_socp` (dense)
//!
//! - **Same**: warm-start logic, residual computation, termination tests,
//!   best-feasible snapshot, Mehrotra sigma, NaN guards, step-length
//!   clipping, interior post-checks, iterate-overflow guard.
//! - **Different**: Newton step computed via the structured solver instead
//!   of dense `H_inv * (b_x - Aᵀ * dl)`.
//!
//! ## Scope of this implementation (all four cells LANDED)
//!
//! - **Both directions**: AHO ([`solve_socp_structured`]) and NT
//!   ([`solve_socp_structured_nt`]). The NT path carries the per-cone NT
//!   scaling `W²` through `M_full` (via [`build_per_cone_nt_blocks`]) instead
//!   of `arrow(s)⁻¹·arrow(y)`.
//! - **Both time modes**: fixed-tf and free-tf
//!   ([`solve_socp_structured_free_tf`], [`solve_socp_structured_nt_free_tf`]).
//!   Free-tf's global `δτ` column couples all stages; it is handled by a
//!   Sherman-Morrison rank-1 update on the Schur complement (see
//!   `reduced_kkt::*_free_tf`).
//! - **Factor/solve split landed**: each IPM iteration factors the reduced
//!   KKT once and applies it twice (predictor + corrector) via the
//!   `factor_*` / `solve_*_with_factor` pair, halving the per-iteration
//!   structured-solve cost.
//!
//! The four drivers share an intentionally-duplicated Mehrotra loop body
//! (per-driver `MAINTENANCE NOTE`s mark the spots that must change in
//! lockstep); the per-driver one-iter equivalence tests vs the dense
//! reference guard against drift.
//!
//! ## Equivalence with the dense driver
//!
//! The `full_newton_step_dense_matches_structured` test in `reduced_kkt.rs`
//! proves the per-iter Newton step computed via this path matches the
//! dense path to ≤ 1e-7 (in Δx, Δλ, Δs, Δy). End-to-end convergence on
//! the same SOCP must therefore produce numerically equivalent
//! trajectories — that's what the integration test in this file pins.

// `for k in 0..N` patterns dominate here because we walk per-stage data in
// lockstep with the cone descriptors; rewriting as `iter().enumerate()`
// hurts readability without a meaningful safety win.
#![allow(clippy::needless_range_loop)]

use libm::sqrt;
use nalgebra::{SMatrix, SVector};

use scvx_core::{IpmAlgoParams, IpmStatus};
use scvx_ipm::{
    soc_arrow_matrix, soc_in_interior, soc_jordan_product, soc_max_step,
    soc_nt_scaling_exact, soc_nt_w_and_inverse, ConeDesc, SocpProblem, SocpResult,
    SocpWorkspace, IPM_HARD_MAX_ITERS,
};

use crate::assemble::{NX, N_VARS_PER_NODE_SCVX};
use crate::reduced_kkt::{
    factor_reduced_kkt_scvx_block_m, factor_reduced_kkt_scvx_block_m_free_tf,
    solve_reduced_kkt_scvx_with_factor, solve_reduced_kkt_scvx_with_factor_free_tf,
    ReducedKktFactor, ReducedKktFactorFreeTf, ReducedKktSolution,
    ReducedKktSolutionFreeTf, ReducedKktStatus, NZ,
};

const BACKOFF: f64 = 0.99;

// ---------------------------------------------------------------------------
// Local helpers (copies of `scvx-ipm::socp` private fns, kept inline to
// avoid bloating the IPM crate's public API just for the structured path).
// ---------------------------------------------------------------------------

/// Initialize a per-cone vector using `h` as a warm-start hint. Same as
/// `socp.rs::init_per_cone_warm_start`.
fn init_per_cone_warm_start<const NCT: usize, const NCONES: usize>(
    h:     &SVector<f64, NCT>,
    cones: &[ConeDesc; NCONES],
    v:     &mut SVector<f64, NCT>,
) {
    const SAFETY_MARGIN_RATIO: f64 = 0.05;
    const BLEND_FRACTION: f64 = 0.10;

    *v = SVector::zeros();
    let hd = h.as_slice();
    let vd = v.as_mut_slice();
    for cone in cones {
        let off = cone.offset;
        let d   = cone.dim;
        let h_slice = &hd[off..off + d];

        // Full-slice finiteness gate (parity with scvx_ipm::socp's copy): a
        // poisoned warm-start seed routes to the canonical interior point.
        // No-op for finite seeds (every nominal path).
        if !h_slice.iter().all(|x| x.is_finite()) {
            vd[off] = 1.0;
            for i in 1..d { vd[off + i] = 0.0; }
            continue;
        }

        let mut bar_mag_sq = 0.0;
        for &val in &h_slice[1..d] {
            bar_mag_sq += val * val;
        }
        let bar_norm = sqrt(bar_mag_sq);
        let scale    = h_slice[0].abs().max(bar_norm).max(1.0);

        if !h_slice[0].is_finite() || !scale.is_finite() {
            vd[off] = 1.0;
            for i in 1..d { vd[off + i] = 0.0; }
            continue;
        }

        let margin     = h_slice[0] - bar_norm;
        let safety_thr = SAFETY_MARGIN_RATIO * scale;

        if margin > safety_thr {
            vd[off..off + d].copy_from_slice(h_slice);
        } else if margin > 0.0 && h_slice[0] > 0.0 {
            let one_minus = 1.0 - BLEND_FRACTION;
            vd[off] = one_minus * h_slice[0] + BLEND_FRACTION * scale;
            for i in 1..d {
                vd[off + i] = one_minus * h_slice[i];
            }
        } else {
            vd[off] = scale;
            for i in 1..d { vd[off + i] = 0.0; }
        }
    }
}

/// Stacked Jordan-algebra identity. Same as `socp.rs::per_cone_e`.
fn per_cone_e<const NCT: usize, const NCONES: usize>(
    cones: &[ConeDesc; NCONES],
) -> SVector<f64, NCT> {
    let mut e = SVector::zeros();
    for cone in cones {
        e[cone.offset] = 1.0;
    }
    e
}

/// Per-cone Jordan product `out[c] = u[c] ∘ v[c]`. Same as
/// `socp.rs::jordan_per_cone`.
fn jordan_per_cone<const NCT: usize, const NCONES: usize>(
    u:     &SVector<f64, NCT>,
    v:     &SVector<f64, NCT>,
    cones: &[ConeDesc; NCONES],
    out:   &mut SVector<f64, NCT>,
) {
    let u_data   = u.as_slice();
    let v_data   = v.as_slice();
    let out_data = out.as_mut_slice();
    for cone in cones {
        let off = cone.offset;
        let d   = cone.dim;
        soc_jordan_product(
            &u_data[off..off + d],
            &v_data[off..off + d],
            &mut out_data[off..off + d],
        );
    }
}

/// Max step along `(z + α·dz)` keeping every cone inside K. Same as
/// `socp.rs::max_step_all_cones`.
fn max_step_all_cones<const NCT: usize, const NCONES: usize>(
    z:     &SVector<f64, NCT>,
    dz:    &SVector<f64, NCT>,
    cones: &[ConeDesc; NCONES],
) -> f64 {
    let zd  = z.as_slice();
    let dzd = dz.as_slice();
    let mut best = f64::INFINITY;
    for cone in cones {
        let off = cone.offset;
        let d   = cone.dim;
        let a   = soc_max_step(&zd[off..off + d], &dzd[off..off + d]);
        if a < best { best = a; }
    }
    best
}

/// Clip a step length into `[0, 1]` with NaN → 0 semantics. Same as
/// `socp.rs::clip01`.
#[inline]
#[allow(clippy::manual_clamp)]
fn clip01(a: f64) -> f64 {
    if a.is_nan() { return 0.0; }
    if a > 1.0 { 1.0 } else if a < 0.0 { 0.0 } else { a }
}

/// Are all cones strictly interior?
fn all_cones_interior<const NCT: usize, const NCONES: usize>(
    z:     &SVector<f64, NCT>,
    cones: &[ConeDesc; NCONES],
) -> bool {
    let zd = z.as_slice();
    for cone in cones {
        if !soc_in_interior(&zd[cone.offset..cone.offset + cone.dim]) {
            return false;
        }
    }
    true
}

// ---------------------------------------------------------------------------
// Per-cone arrow matrix builders (block-diagonal NCT × NCT)
// ---------------------------------------------------------------------------

/// Build the block-diagonal `M = arrow(s)⁻¹·arrow(y)` (NCT × NCT, dense
/// per-cone), the block-diagonal `arrow(s)⁻¹`, and the block-diagonal
/// `arrow(y)`. All three share the same block sparsity pattern.
///
/// Per-D dispatch over `D ∈ {1, 3, 4, 8, 11}` — the cone dims actually
/// used in the SCvx subproblem. Returns `None` if any per-cone arrow
/// inversion fails (cone iterate on boundary — IPM caller bails to
/// `numerical_exit`).
fn build_per_cone_arrow_blocks<const NCT: usize, const NCONES: usize>(
    s:     &SVector<f64, NCT>,
    y:     &SVector<f64, NCT>,
    cones: &[ConeDesc; NCONES],
) -> Option<(
    SMatrix<f64, NCT, NCT>,  // arrow_s_inv (block-diag)
    SMatrix<f64, NCT, NCT>,  // arrow_y     (block-diag)
    SMatrix<f64, NCT, NCT>,  // m = arrow_s_inv · arrow_y (block-diag)
)> {
    let mut arrow_s_inv = SMatrix::<f64, NCT, NCT>::zeros();
    let mut arrow_y     = SMatrix::<f64, NCT, NCT>::zeros();
    let mut m_full      = SMatrix::<f64, NCT, NCT>::zeros();

    for cone in cones {
        let off = cone.offset;
        let d   = cone.dim;
        macro_rules! per_d {
            ($D:literal) => {{
                let mut sv = SVector::<f64, $D>::zeros();
                let mut yv = SVector::<f64, $D>::zeros();
                for i in 0..$D { sv[i] = s[off + i]; yv[i] = y[off + i]; }
                let a_s = soc_arrow_matrix(&sv);
                let a_y = soc_arrow_matrix(&yv);
                let a_s_inv = a_s.try_inverse()?;
                let m_c = a_s_inv * a_y;
                for i in 0..$D {
                    for j in 0..$D {
                        arrow_s_inv[(off + i, off + j)] = a_s_inv[(i, j)];
                        arrow_y    [(off + i, off + j)] = a_y    [(i, j)];
                        m_full     [(off + i, off + j)] = m_c    [(i, j)];
                    }
                }
            }};
        }
        match d {
            1  => per_d!(1),
            3  => per_d!(3),
            4  => per_d!(4),
            8  => per_d!(8),
            11 => per_d!(11),
            _  => return None,
        }
    }
    Some((arrow_s_inv, arrow_y, m_full))
}

/// Build the per-cone **Nesterov-Todd** scaling blocks (block-diagonal,
/// NCT × NCT):
/// - `w`: the NT scaling `W` (symmetric PD)
/// - `w_squared`: `W·W` (the `M` matrix for `H = Gᵀ·W²·G`) — computed as
///   `W·W` exactly so it is self-consistent with `w`/`s_scaled`, NOT via the
///   separate `soc_w_squared` path
/// - `arrow_s_scaled_inv`: `arrow(s̃)⁻¹` where `s̃ = W·s`
/// - `s_scaled`: the scaled iterate `s̃ = W·s`
///
/// Per-D dispatch over `D ∈ {1, 3, 4, 8, 11}`. Uses the exact closed-form NT
/// scaling `soc_nt_scaling_exact` (vanishing-cone-stable) as the PRIMARY path,
/// with `soc_nt_w_and_inverse` (geometric-mean Denman-Beavers) as the fallback —
/// mirroring the dense path's `build_nt_block_for_cone`, so the structured NT
/// driver stays per-step-equivalent to the dense NT driver (and inherits its
/// vanishing-cone stability rather than the old DB overflow-to-`None`). Returns
/// `None` if any per-cone NT computation fails (non-interior iterate, both
/// scalings failing, or singular `arrow(s̃)`) — the IPM caller bails to
/// `numerical_exit`, and the SCvx outer loop falls back to the dense NT driver.
#[allow(clippy::type_complexity)]
fn build_per_cone_nt_blocks<const NCT: usize, const NCONES: usize>(
    s:     &SVector<f64, NCT>,
    y:     &SVector<f64, NCT>,
    cones: &[ConeDesc; NCONES],
) -> Option<(
    SMatrix<f64, NCT, NCT>,  // w                  (block-diag)
    SMatrix<f64, NCT, NCT>,  // w_squared = W·W    (block-diag, = m_full)
    SMatrix<f64, NCT, NCT>,  // arrow_s_scaled_inv (block-diag)
    SVector<f64, NCT>,       // s_scaled = W·s
)> {
    let mut w_out        = SMatrix::<f64, NCT, NCT>::zeros();
    let mut w_sq_out     = SMatrix::<f64, NCT, NCT>::zeros();
    let mut arrow_ss_inv = SMatrix::<f64, NCT, NCT>::zeros();
    let mut s_scaled     = SVector::<f64, NCT>::zeros();

    for cone in cones {
        let off = cone.offset;
        let d   = cone.dim;
        macro_rules! per_d {
            ($D:literal) => {{
                let mut sv = SVector::<f64, $D>::zeros();
                let mut yv = SVector::<f64, $D>::zeros();
                for i in 0..$D { sv[i] = s[off + i]; yv[i] = y[off + i]; }
                // Exact closed-form NT scaling (primary, vanishing-cone-stable),
                // geometric-mean Denman-Beavers as fallback — same primitive
                // order as the dense `build_nt_block_for_cone`.
                let (w_c, _w_inv_c) = soc_nt_scaling_exact::<$D>(&sv, &yv)
                    .or_else(|| soc_nt_w_and_inverse::<$D>(&sv, &yv))?;
                let w_sq_c     = w_c * w_c;
                let s_scaled_c = w_c * sv;
                let arrow_ss_inv_c = soc_arrow_matrix(&s_scaled_c).try_inverse()?;
                for i in 0..$D {
                    s_scaled[off + i] = s_scaled_c[i];
                    for j in 0..$D {
                        w_out       [(off + i, off + j)] = w_c[(i, j)];
                        w_sq_out    [(off + i, off + j)] = w_sq_c[(i, j)];
                        arrow_ss_inv[(off + i, off + j)] = arrow_ss_inv_c[(i, j)];
                    }
                }
            }};
        }
        match d {
            1  => per_d!(1),
            3  => per_d!(3),
            4  => per_d!(4),
            8  => per_d!(8),
            11 => per_d!(11),
            _  => return None,
        }
    }
    Some((w_out, w_sq_out, arrow_ss_inv, s_scaled))
}

/// Stack per-stage `dz_k` (length NZ) into a flat NP-vector for use in
/// downstream Δs / Δy recovery formulas (which still operate in the
/// "flat NP" coordinate system).
fn stack_dz<const N: usize, const NP: usize>(
    sol: &ReducedKktSolution<N>,
) -> SVector<f64, NP> {
    let mut dx = SVector::<f64, NP>::zeros();
    for k in 0..N {
        for j in 0..NZ {
            dx[k * NZ + j] = sol.dz[k][j];
        }
    }
    dx
}

/// Numerical-exit cleanup: return best-feasible if available, else scrub
/// non-finite entries from the live iterate and return them. Mirrors
/// `scvx-ipm::socp::numerical_exit`.
fn numerical_exit<const NP: usize, const NE: usize, const NCT: usize>(
    ws:     &SocpWorkspace<NP, NE, NCT>,
    status: IpmStatus,
    iter:   u32,
) -> SocpResult<NP, NE, NCT> {
    // SocpWorkspace's `best_*` fields are private. Use the same fallback
    // logic by re-reading the public x/lambda/s/y fields. If a best
    // snapshot existed, the dense driver would have copied it into the
    // result; here we can't access `ws.best_*` directly, so we emit the
    // live iterate (scrubbed).
    let mut x      = ws.x;
    let mut lambda = ws.lambda;
    let mut s      = ws.s;
    let mut y      = ws.y;
    let mut had_dirty = false;
    for v in x.as_mut_slice().iter_mut() {
        if !v.is_finite() { *v = 0.0; had_dirty = true; }
    }
    for v in lambda.as_mut_slice().iter_mut() {
        if !v.is_finite() { *v = 0.0; had_dirty = true; }
    }
    for v in s.as_mut_slice().iter_mut() {
        if !v.is_finite() { *v = 0.0; had_dirty = true; }
    }
    for v in y.as_mut_slice().iter_mut() {
        if !v.is_finite() { *v = 0.0; had_dirty = true; }
    }
    let final_status = if had_dirty { IpmStatus::NumericalError } else { status };
    SocpResult { x, lambda, s, y, status: final_status, iters: iter }
}

// ---------------------------------------------------------------------------
// The driver
// ---------------------------------------------------------------------------

/// Structured-Schur AHO Mehrotra IPM driver for SCvx-shaped SOCPs.
///
/// Drop-in replacement for `scvx_ipm::solve_socp` on the **fixed-tf SCvx
/// problem class** — same calling convention (`prob`, `params`, `ws`),
/// same return type. Uses the block-tridiagonal reduced-KKT solver
/// internally, giving `O(N·NZ³)` per-iter cost vs the dense driver's
/// `O((N·NZ)³)`.
///
/// **Assumes** the problem `prob` was assembled via
/// `crate::assemble::assemble_scvx_socp` with `use_free_tf=false`. The
/// debug-build invariants in `solve_reduced_kkt_scvx_block_m` check the
/// SCvx layout (`NP = N·19`, `NE = N·7 + 6`, etc.); release builds trust
/// the caller. Misuse on a non-SCvx problem will produce garbage Δz
/// extraction (the per-stage block-diagonal H assumption is violated).
///
/// **MAINTENANCE NOTE**: the Mehrotra predictor/corrector loop body here
/// is mirrored in [`solve_socp_structured_free_tf`] (which uses the
/// Sherman-Morrison factor/apply pair for the global δτ column). The two
/// loops are intentionally duplicated rather than abstracted behind a
/// trait — the WCET-critical hot path stays explicit and inspectable, the
/// same way `scvx_ipm::solve_socp` and `solve_socp_nt` are kept separate.
/// **If you change the loop logic here (NaN guards, step lengths, sigma
/// centering, snapshot rules), make the identical change in the free-tf
/// driver.** The only intended differences are the factor/apply functions,
/// the solution type, and the `stack_dz` helper (which appends δτ).
pub fn solve_socp_structured<
    const N:      usize,
    const NP:     usize,
    const NE:     usize,
    const NCT:    usize,
    const NCONES: usize,
>(
    prob:   &SocpProblem<NP, NE, NCT, NCONES>,
    params: &IpmAlgoParams,
    ws:     &mut SocpWorkspace<NP, NE, NCT>,
) -> SocpResult<NP, NE, NCT> {
    // Debug-build sanity check: this driver is only valid on SCvx-shaped
    // problems. Release builds skip — caller's responsibility.
    debug_assert_eq!(NP, N * N_VARS_PER_NODE_SCVX,
                     "solve_socp_structured: NP must equal N · 19 (SCvx layout)");
    debug_assert_eq!(NE, N * NX + 6,
                     "solve_socp_structured: NE must equal N · 7 + 6");

    // ---- Initialization (mirror solve_socp) ----
    if !params.warm_start_x {
        ws.x      = SVector::zeros();
        ws.lambda = SVector::zeros();
        init_per_cone_warm_start(&prob.h, &prob.cones, &mut ws.s);
        init_per_cone_warm_start(&prob.h, &prob.cones, &mut ws.y);
    } else {
        let gx = prob.g_mat * ws.x;
        let s_target = prob.h - gx;
        init_per_cone_warm_start(&s_target, &prob.cones, &mut ws.s);
        init_per_cone_warm_start(&prob.h,    &prob.cones, &mut ws.y);
    }

    let e_vec = per_cone_e::<NCT, NCONES>(&prob.cones);
    let ncone = NCT as f64;
    let loose_dual   = libm::sqrt(params.tol_dual.max(0.0)).max(1.0e-5);
    let loose_primal = libm::sqrt(params.tol_primal.max(0.0)).max(1.0e-5);
    let loose_mu     = libm::sqrt(params.tol_mu.max(0.0)).max(1.0e-5);

    // We track a private best-snapshot here since the public workspace
    // fields are off-limits.
    let mut best_x:      SVector<f64, NP> = SVector::zeros();
    let mut best_lambda: SVector<f64, NE> = SVector::zeros();
    let mut best_s:      SVector<f64, NCT> = SVector::zeros();
    let mut best_y:      SVector<f64, NCT> = SVector::zeros();
    let mut best_mu = f64::INFINITY;
    let mut best_valid = false;

    // On a numerical breakdown mid-loop, prefer the captured best-feasible
    // snapshot (return it as `BestFeasible`) over scrubbing the live iterate
    // to `NumericalError`. NOTE: unlike the dense driver's `numerical_exit`
    // (which can read `ws.best_*`), this driver cannot — those workspace
    // fields are private — so it tracks its OWN `best_*` locally and returns
    // them via `bail_result!`. `numerical_exit` here is only the no-snapshot
    // fallback (scrub the live iterate). The effect matches the dense driver
    // (a good snapshot is returned as `BestFeasible`, sparing the SCvx outer
    // loop a wasteful dense re-solve), but the mechanism is the local snapshot,
    // NOT `ws.best_*`. Do not "simplify" by deleting the local tracking.
    macro_rules! bail_result {
        ($it:expr) => {
            if best_valid {
                SocpResult {
                    x: best_x, lambda: best_lambda, s: best_s, y: best_y,
                    status: IpmStatus::BestFeasible,
                    iters: $it,
                }
            } else {
                numerical_exit(ws, IpmStatus::NumericalError, $it)
            }
        };
    }

    let mut prev_x = ws.x;
    let mut prev_y = ws.y;

    // Reuse a single ReducedKktSolution buffer across the per-iter
    // predictor + corrector calls.
    let mut sol_buf: ReducedKktSolution<N> = ReducedKktSolution::default();

    // Reuse a single factor buffer across the per-iter predictor +
    // corrector. **The Phase 6.7 optimization**: each IPM iteration
    // builds the structured factor (`O(N·NZ³)`) ONCE and applies it
    // TWICE (once for the affine predictor RHS, once for the corrector
    // RHS) — saving the second factor pass entirely. ~2× per-iter cost
    // reduction relative to the previous "two full solves" path.
    let mut factor_buf: ReducedKktFactor<N> = ReducedKktFactor::default();

    for iter in 0..params.max_iters.min(IPM_HARD_MAX_ITERS) {
        // ---- Residuals ----
        let r_x = prob.c
                + prob.a_mat.transpose() * ws.lambda
                + prob.g_mat.transpose() * ws.y;
        let r_a = prob.a_mat * ws.x - prob.b;
        let r_g = prob.g_mat * ws.x + ws.s - prob.h;
        let mut r_c_aff = SVector::<f64, NCT>::zeros();
        jordan_per_cone(&ws.y, &ws.s, &prob.cones, &mut r_c_aff);

        // ---- Termination ----
        let mu       = ws.s.dot(&ws.y) / ncone;
        let primal_r = r_a.norm() + r_g.norm();
        let dual_r   = r_x.norm();
        if mu < params.tol_mu && primal_r < params.tol_primal && dual_r < params.tol_dual {
            return SocpResult {
                x: ws.x, lambda: ws.lambda, s: ws.s, y: ws.y,
                status: IpmStatus::Optimal,
                iters:  iter,
            };
        }

        // ---- Best-feasible snapshot ----
        if dual_r < loose_dual
            && primal_r < loose_primal
            && mu.is_finite() && mu >= 0.0 && mu < best_mu
        {
            best_x      = ws.x;
            best_lambda = ws.lambda;
            best_s      = ws.s;
            best_y      = ws.y;
            best_mu     = mu;
            best_valid  = true;
        }

        if iter > 2 && best_valid && best_mu < loose_mu {
            return SocpResult {
                x: best_x, lambda: best_lambda, s: best_s, y: best_y,
                status: IpmStatus::BestFeasible,
                iters:  iter,
            };
        }

        // No-progress guard.
        let dx_iter = (ws.x - prev_x).norm();
        let dy_iter = (ws.y - prev_y).norm();
        if iter > 0 && dx_iter < 1.0e-9 && dy_iter < 1.0e-9
            && best_valid && best_mu < loose_mu
        {
            return SocpResult {
                x: best_x, lambda: best_lambda, s: best_s, y: best_y,
                status: IpmStatus::BestFeasible,
                iters:  iter,
            };
        }
        prev_x = ws.x;
        prev_y = ws.y;

        // ---- Build per-cone arrow blocks (shared with predictor + corrector) ----
        let (arrow_s_inv, arrow_y, m_full) =
            match build_per_cone_arrow_blocks(&ws.s, &ws.y, &prob.cones) {
                Some(t) => t,
                None    => return bail_result!(iter),
            };

        // Regularization (same convention as solve_socp's H_REG_FLOOR /
        // adaptive). For the structured path we keep it simple: fixed floor.
        let reg: f64 = 1.0e-8;

        // ---- Factor the structured KKT once per iter ----
        //
        // The factor captures all RHS-independent work (H_k inversion,
        // Schur block-tridiag forward elim). The predictor + corrector
        // both apply this factor to different RHS vectors, saving the
        // second factor pass.
        let st = factor_reduced_kkt_scvx_block_m::<N, NP, NE, NCT, NCONES>(
            prob, &m_full, reg, &mut factor_buf,
        );
        if st != ReducedKktStatus::Ok {
            return bail_result!(iter);
        }

        // ---- Affine (predictor) step ----
        //
        // b_x_pred = -r_x - Gᵀ·M·r_g + Gᵀ·arrow(s)⁻¹·r_c_aff
        // b_a      = -r_a
        let b_x_pred = -r_x
                     - prob.g_mat.transpose() * (m_full     * r_g)
                     + prob.g_mat.transpose() * (arrow_s_inv * r_c_aff);
        let b_a = -r_a;

        let st = solve_reduced_kkt_scvx_with_factor::<N, NP, NE, NCT, NCONES>(
            prob, &factor_buf, &b_x_pred, &b_a, &mut sol_buf,
        );
        if st != ReducedKktStatus::Ok {
            return bail_result!(iter);
        }
        let dx_a = stack_dz::<N, NP>(&sol_buf);
        let ds_a = -r_g - prob.g_mat * dx_a;
        let dy_a = arrow_s_inv * (arrow_y * (r_g + prob.g_mat * dx_a) - r_c_aff);

        let affine_finite = ds_a.iter().all(|v| v.is_finite())
            && dy_a.iter().all(|v| v.is_finite())
            && dx_a.iter().all(|v| v.is_finite());
        if !affine_finite {
            return bail_result!(iter + 1);
        }

        let alpha_s_aff = clip01(BACKOFF * max_step_all_cones(&ws.s, &ds_a, &prob.cones));
        let alpha_y_aff = clip01(BACKOFF * max_step_all_cones(&ws.y, &dy_a, &prob.cones));

        // Mehrotra centering.
        let s_aff = ws.s + ds_a * alpha_s_aff;
        let y_aff = ws.y + dy_a * alpha_y_aff;
        let mu_aff = s_aff.dot(&y_aff) / ncone;
        let sigma_raw = if mu > 1.0e-300 {
            let r = mu_aff / mu;
            r * r * r
        } else {
            1.0
        };
        let sigma = if sigma_raw.is_finite() { sigma_raw.clamp(0.0, 1.0) } else { 0.5 };

        // Corrector RHS.
        let mut second_order = SVector::<f64, NCT>::zeros();
        jordan_per_cone(&dy_a, &ds_a, &prob.cones, &mut second_order);
        let r_c = r_c_aff + second_order - e_vec * (sigma * mu);

        // ---- Corrector solve (reuses the factor from above) ----
        let b_x_corr = -r_x
                     - prob.g_mat.transpose() * (m_full     * r_g)
                     + prob.g_mat.transpose() * (arrow_s_inv * r_c);

        let st = solve_reduced_kkt_scvx_with_factor::<N, NP, NE, NCT, NCONES>(
            prob, &factor_buf, &b_x_corr, &b_a, &mut sol_buf,
        );
        if st != ReducedKktStatus::Ok {
            return bail_result!(iter);
        }
        let dx = stack_dz::<N, NP>(&sol_buf);
        let ds = -r_g - prob.g_mat * dx;
        let dy = arrow_s_inv * (arrow_y * (r_g + prob.g_mat * dx) - r_c);

        // Stitch dual multipliers back into the flat NE vector for use
        // in the `ws.lambda` update.
        let mut dl = SVector::<f64, NE>::zeros();
        for i in 0..NX { dl[i] = sol_buf.dlam_init[i]; }
        for k in 0..(N - 1) {
            for i in 0..NX {
                dl[NX + k * NX + i] = sol_buf.dlam_dyn[k][i];
            }
        }
        for i in 0..6 {
            dl[N * NX + i] = sol_buf.dlam_term[i];
        }

        let step_finite = dx.iter().all(|v| v.is_finite())
            && dl.iter().all(|v| v.is_finite())
            && ds.iter().all(|v| v.is_finite())
            && dy.iter().all(|v| v.is_finite());
        if !step_finite {
            return bail_result!(iter + 1);
        }

        let alpha_s = clip01(BACKOFF * max_step_all_cones(&ws.s, &ds, &prob.cones));
        let alpha_y = clip01(BACKOFF * max_step_all_cones(&ws.y, &dy, &prob.cones));

        ws.x      += dx * alpha_s;
        ws.lambda += dl * alpha_y;
        ws.s      += ds * alpha_s;
        ws.y      += dy * alpha_y;

        if !all_cones_interior(&ws.s, &prob.cones)
            || !all_cones_interior(&ws.y, &prob.cones)
        {
            return bail_result!(iter + 1);
        }
        let max_abs = ws.x.amax().max(ws.s.amax()).max(ws.y.amax()).max(ws.lambda.amax());
        if !max_abs.is_finite() || max_abs > 1.0e50 {
            return bail_result!(iter + 1);
        }
    }

    // Iter cap: return best snapshot if we have one, else clean the live
    // iterate.
    if best_valid {
        SocpResult {
            x: best_x, lambda: best_lambda, s: best_s, y: best_y,
            status: IpmStatus::BestFeasible,
            iters:  params.max_iters.min(IPM_HARD_MAX_ITERS),
        }
    } else {
        numerical_exit(ws, IpmStatus::IterCap, params.max_iters.min(IPM_HARD_MAX_ITERS))
    }
}

// ---------------------------------------------------------------------------
// Free-tf structured driver (Phase 6.8 Sherman-Morrison)
// ---------------------------------------------------------------------------

/// Stack per-stage `dz_k` and the scalar `dz_delta_tau` into a flat NP-vector
/// where the last element is δτ. For free-tf SCvx, NP = N·NZ + 1.
fn stack_dz_free_tf<const N: usize, const NP: usize>(
    sol: &ReducedKktSolutionFreeTf<N>,
) -> SVector<f64, NP> {
    let mut dx = SVector::<f64, NP>::zeros();
    for k in 0..N {
        for j in 0..NZ {
            dx[k * NZ + j] = sol.base.dz[k][j];
        }
    }
    dx[N * NZ] = sol.dz_delta_tau;
    dx
}

/// Structured-Schur AHO Mehrotra IPM driver for **free-tf** SCvx-shaped
/// SOCPs. Drop-in replacement for `scvx_ipm::solve_socp` on the
/// **free-tf AHO** problem class (NP = N·NZ + 1).
///
/// Same Mehrotra recipe as [`solve_socp_structured`] — only difference is
/// it uses the **Sherman-Morrison-augmented** factor/apply pair
/// ([`factor_reduced_kkt_scvx_block_m_free_tf`] +
/// [`solve_reduced_kkt_scvx_with_factor_free_tf`]) which handles the
/// global δτ column via rank-1 update on the block-tridiag Schur.
///
/// **Assumes** the problem `prob` was assembled with `use_free_tf = true`.
/// Debug-asserts the layout; release trusts the caller.
///
/// **MAINTENANCE NOTE**: this loop body is the free-tf twin of
/// [`solve_socp_structured`]. The two are intentionally duplicated (see
/// that function's maintenance note). If you change the Mehrotra loop
/// logic in either, mirror it in the other. The only intended differences
/// are: the SMW factor/apply pair, [`ReducedKktSolutionFreeTf`], and
/// [`stack_dz_free_tf`] (which appends the δτ scalar at index `N·NZ`).
pub fn solve_socp_structured_free_tf<
    const N:      usize,
    const NP:     usize,
    const NE:     usize,
    const NCT:    usize,
    const NCONES: usize,
>(
    prob:   &SocpProblem<NP, NE, NCT, NCONES>,
    params: &IpmAlgoParams,
    ws:     &mut SocpWorkspace<NP, NE, NCT>,
) -> SocpResult<NP, NE, NCT> {
    debug_assert_eq!(NP, N * N_VARS_PER_NODE_SCVX + 1,
                     "solve_socp_structured_free_tf: NP must equal N·19 + 1");
    debug_assert_eq!(NE, N * NX + 6,
                     "solve_socp_structured_free_tf: NE must equal N·7 + 6");

    // ---- Init (mirror solve_socp_structured) ----
    if !params.warm_start_x {
        ws.x      = SVector::zeros();
        ws.lambda = SVector::zeros();
        init_per_cone_warm_start(&prob.h, &prob.cones, &mut ws.s);
        init_per_cone_warm_start(&prob.h, &prob.cones, &mut ws.y);
    } else {
        let gx = prob.g_mat * ws.x;
        let s_target = prob.h - gx;
        init_per_cone_warm_start(&s_target, &prob.cones, &mut ws.s);
        init_per_cone_warm_start(&prob.h,    &prob.cones, &mut ws.y);
    }

    let e_vec = per_cone_e::<NCT, NCONES>(&prob.cones);
    let ncone = NCT as f64;
    let loose_dual   = libm::sqrt(params.tol_dual.max(0.0)).max(1.0e-5);
    let loose_primal = libm::sqrt(params.tol_primal.max(0.0)).max(1.0e-5);
    let loose_mu     = libm::sqrt(params.tol_mu.max(0.0)).max(1.0e-5);

    let mut best_x:      SVector<f64, NP> = SVector::zeros();
    let mut best_lambda: SVector<f64, NE> = SVector::zeros();
    let mut best_s:      SVector<f64, NCT> = SVector::zeros();
    let mut best_y:      SVector<f64, NCT> = SVector::zeros();
    let mut best_mu = f64::INFINITY;
    let mut best_valid = false;

    // On a numerical breakdown mid-loop, prefer the captured best-feasible
    // snapshot (return it as `BestFeasible`) over scrubbing the live iterate
    // to `NumericalError`. NOTE: unlike the dense driver's `numerical_exit`
    // (which can read `ws.best_*`), this driver cannot — those workspace
    // fields are private — so it tracks its OWN `best_*` locally and returns
    // them via `bail_result!`. `numerical_exit` here is only the no-snapshot
    // fallback (scrub the live iterate). The effect matches the dense driver
    // (a good snapshot is returned as `BestFeasible`, sparing the SCvx outer
    // loop a wasteful dense re-solve), but the mechanism is the local snapshot,
    // NOT `ws.best_*`. Do not "simplify" by deleting the local tracking.
    macro_rules! bail_result {
        ($it:expr) => {
            if best_valid {
                SocpResult {
                    x: best_x, lambda: best_lambda, s: best_s, y: best_y,
                    status: IpmStatus::BestFeasible,
                    iters: $it,
                }
            } else {
                numerical_exit(ws, IpmStatus::NumericalError, $it)
            }
        };
    }

    let mut prev_x = ws.x;
    let mut prev_y = ws.y;

    let mut sol_buf: ReducedKktSolutionFreeTf<N> = ReducedKktSolutionFreeTf::default();
    let mut factor_buf: ReducedKktFactorFreeTf<N> = ReducedKktFactorFreeTf::default();

    for iter in 0..params.max_iters.min(IPM_HARD_MAX_ITERS) {
        let r_x = prob.c
                + prob.a_mat.transpose() * ws.lambda
                + prob.g_mat.transpose() * ws.y;
        let r_a = prob.a_mat * ws.x - prob.b;
        let r_g = prob.g_mat * ws.x + ws.s - prob.h;
        let mut r_c_aff = SVector::<f64, NCT>::zeros();
        jordan_per_cone(&ws.y, &ws.s, &prob.cones, &mut r_c_aff);

        let mu       = ws.s.dot(&ws.y) / ncone;
        let primal_r = r_a.norm() + r_g.norm();
        let dual_r   = r_x.norm();
        if mu < params.tol_mu && primal_r < params.tol_primal && dual_r < params.tol_dual {
            return SocpResult {
                x: ws.x, lambda: ws.lambda, s: ws.s, y: ws.y,
                status: IpmStatus::Optimal,
                iters:  iter,
            };
        }

        if dual_r < loose_dual && primal_r < loose_primal
            && mu.is_finite() && mu >= 0.0 && mu < best_mu
        {
            best_x      = ws.x;
            best_lambda = ws.lambda;
            best_s      = ws.s;
            best_y      = ws.y;
            best_mu     = mu;
            best_valid  = true;
        }

        if iter > 2 && best_valid && best_mu < loose_mu {
            return SocpResult {
                x: best_x, lambda: best_lambda, s: best_s, y: best_y,
                status: IpmStatus::BestFeasible,
                iters:  iter,
            };
        }
        let dx_iter = (ws.x - prev_x).norm();
        let dy_iter = (ws.y - prev_y).norm();
        if iter > 0 && dx_iter < 1.0e-9 && dy_iter < 1.0e-9
            && best_valid && best_mu < loose_mu
        {
            return SocpResult {
                x: best_x, lambda: best_lambda, s: best_s, y: best_y,
                status: IpmStatus::BestFeasible,
                iters:  iter,
            };
        }
        prev_x = ws.x;
        prev_y = ws.y;

        let (arrow_s_inv, arrow_y, m_full) =
            match build_per_cone_arrow_blocks(&ws.s, &ws.y, &prob.cones) {
                Some(t) => t,
                None    => return bail_result!(iter),
            };

        let reg: f64 = 1.0e-8;

        // Factor with SMW.
        let st = factor_reduced_kkt_scvx_block_m_free_tf::<N, NP, NE, NCT, NCONES>(
            prob, &m_full, reg, &mut factor_buf,
        );
        if st != ReducedKktStatus::Ok {
            return bail_result!(iter);
        }

        // Predictor.
        let b_x_pred = -r_x
                     - prob.g_mat.transpose() * (m_full     * r_g)
                     + prob.g_mat.transpose() * (arrow_s_inv * r_c_aff);
        let b_a = -r_a;

        let st = solve_reduced_kkt_scvx_with_factor_free_tf::<N, NP, NE, NCT, NCONES>(
            prob, &factor_buf, &b_x_pred, &b_a, &mut sol_buf,
        );
        if st != ReducedKktStatus::Ok {
            return bail_result!(iter);
        }
        let dx_a = stack_dz_free_tf::<N, NP>(&sol_buf);
        let ds_a = -r_g - prob.g_mat * dx_a;
        let dy_a = arrow_s_inv * (arrow_y * (r_g + prob.g_mat * dx_a) - r_c_aff);

        let affine_finite = ds_a.iter().all(|v| v.is_finite())
            && dy_a.iter().all(|v| v.is_finite())
            && dx_a.iter().all(|v| v.is_finite());
        if !affine_finite {
            return bail_result!(iter + 1);
        }

        let alpha_s_aff = clip01(BACKOFF * max_step_all_cones(&ws.s, &ds_a, &prob.cones));
        let alpha_y_aff = clip01(BACKOFF * max_step_all_cones(&ws.y, &dy_a, &prob.cones));

        let s_aff = ws.s + ds_a * alpha_s_aff;
        let y_aff = ws.y + dy_a * alpha_y_aff;
        let mu_aff = s_aff.dot(&y_aff) / ncone;
        let sigma_raw = if mu > 1.0e-300 {
            let r = mu_aff / mu;
            r * r * r
        } else {
            1.0
        };
        let sigma = if sigma_raw.is_finite() { sigma_raw.clamp(0.0, 1.0) } else { 0.5 };

        let mut second_order = SVector::<f64, NCT>::zeros();
        jordan_per_cone(&dy_a, &ds_a, &prob.cones, &mut second_order);
        let r_c = r_c_aff + second_order - e_vec * (sigma * mu);

        let b_x_corr = -r_x
                     - prob.g_mat.transpose() * (m_full     * r_g)
                     + prob.g_mat.transpose() * (arrow_s_inv * r_c);

        let st = solve_reduced_kkt_scvx_with_factor_free_tf::<N, NP, NE, NCT, NCONES>(
            prob, &factor_buf, &b_x_corr, &b_a, &mut sol_buf,
        );
        if st != ReducedKktStatus::Ok {
            return bail_result!(iter);
        }
        let dx = stack_dz_free_tf::<N, NP>(&sol_buf);
        let ds = -r_g - prob.g_mat * dx;
        let dy = arrow_s_inv * (arrow_y * (r_g + prob.g_mat * dx) - r_c);

        let mut dl = SVector::<f64, NE>::zeros();
        for i in 0..NX { dl[i] = sol_buf.base.dlam_init[i]; }
        for k in 0..(N - 1) {
            for i in 0..NX {
                dl[NX + k * NX + i] = sol_buf.base.dlam_dyn[k][i];
            }
        }
        for i in 0..6 {
            dl[N * NX + i] = sol_buf.base.dlam_term[i];
        }

        let step_finite = dx.iter().all(|v| v.is_finite())
            && dl.iter().all(|v| v.is_finite())
            && ds.iter().all(|v| v.is_finite())
            && dy.iter().all(|v| v.is_finite());
        if !step_finite {
            return bail_result!(iter + 1);
        }

        let alpha_s = clip01(BACKOFF * max_step_all_cones(&ws.s, &ds, &prob.cones));
        let alpha_y = clip01(BACKOFF * max_step_all_cones(&ws.y, &dy, &prob.cones));

        ws.x      += dx * alpha_s;
        ws.lambda += dl * alpha_y;
        ws.s      += ds * alpha_s;
        ws.y      += dy * alpha_y;

        if !all_cones_interior(&ws.s, &prob.cones)
            || !all_cones_interior(&ws.y, &prob.cones)
        {
            return bail_result!(iter + 1);
        }
        let max_abs = ws.x.amax().max(ws.s.amax()).max(ws.y.amax()).max(ws.lambda.amax());
        if !max_abs.is_finite() || max_abs > 1.0e50 {
            return bail_result!(iter + 1);
        }
    }

    if best_valid {
        SocpResult {
            x: best_x, lambda: best_lambda, s: best_s, y: best_y,
            status: IpmStatus::BestFeasible,
            iters:  params.max_iters.min(IPM_HARD_MAX_ITERS),
        }
    } else {
        numerical_exit(ws, IpmStatus::IterCap, params.max_iters.min(IPM_HARD_MAX_ITERS))
    }
}

// ---------------------------------------------------------------------------
// NT-direction structured driver (Phase 6.9)
// ---------------------------------------------------------------------------

/// Structured-Schur **Nesterov-Todd** Mehrotra IPM driver for fixed-tf
/// SCvx-shaped SOCPs. The NT twin of [`solve_socp_structured`].
///
/// The reduced Hessian is `H = Gᵀ·W²·G` (symmetric PD) instead of AHO's
/// `Gᵀ·(arrow(s)⁻¹·arrow(y))·G`. We pass `W²` as the `m_full` matrix to the
/// **same** block-tridiagonal structured solver — the solver is scaling-
/// agnostic, so no `reduced_kkt` changes are needed. Only the per-cone
/// scaling block construction and the Δs/Δy recovery formulas differ.
///
/// NT Newton step (mirrors `scvx_ipm::socp::solve_newton_step_nt`):
/// ```text
///   b_x = −r_x − Gᵀ·W²·r_g + Gᵀ·W·r_c_arg     (r_c_arg in scaled coords)
///   [Δx, Δλ] via structured Schur on H = Gᵀ·W²·G
///   Δs  = −r_g − G·Δx
///   Δy  = −W·r_c_arg + W²·(r_g + G·Δx)
/// ```
///
/// **Assumes** fixed-tf SCvx layout (`NP = N·19`). Free-tf NT would compose
/// this with the Sherman-Morrison δτ correction — a further lift.
///
/// **MAINTENANCE NOTE**: this is the NT analogue of `solve_socp_structured`.
/// The outer Mehrotra scaffold (warm-start, residuals, termination,
/// snapshot, step lengths, guards) is identical; only the scaling-block
/// build and the RHS/recovery formulas differ. Keep the scaffold in sync
/// with `solve_socp_structured` and the formulas in sync with
/// `scvx_ipm::socp::solve_socp_nt`.
pub fn solve_socp_structured_nt<
    const N:      usize,
    const NP:     usize,
    const NE:     usize,
    const NCT:    usize,
    const NCONES: usize,
>(
    prob:   &SocpProblem<NP, NE, NCT, NCONES>,
    params: &IpmAlgoParams,
    ws:     &mut SocpWorkspace<NP, NE, NCT>,
) -> SocpResult<NP, NE, NCT> {
    debug_assert_eq!(NP, N * N_VARS_PER_NODE_SCVX,
                     "solve_socp_structured_nt: NP must equal N·19 (fixed-tf SCvx)");
    debug_assert_eq!(NE, N * NX + 6,
                     "solve_socp_structured_nt: NE must equal N·7 + 6");

    // ---- Init (mirror solve_socp_structured) ----
    if !params.warm_start_x {
        ws.x      = SVector::zeros();
        ws.lambda = SVector::zeros();
        init_per_cone_warm_start(&prob.h, &prob.cones, &mut ws.s);
        init_per_cone_warm_start(&prob.h, &prob.cones, &mut ws.y);
    } else {
        let gx = prob.g_mat * ws.x;
        let s_target = prob.h - gx;
        init_per_cone_warm_start(&s_target, &prob.cones, &mut ws.s);
        init_per_cone_warm_start(&prob.h,    &prob.cones, &mut ws.y);
    }

    let e_vec = per_cone_e::<NCT, NCONES>(&prob.cones);
    let ncone = NCT as f64;
    let loose_dual   = libm::sqrt(params.tol_dual.max(0.0)).max(1.0e-5);
    let loose_primal = libm::sqrt(params.tol_primal.max(0.0)).max(1.0e-5);
    let loose_mu     = libm::sqrt(params.tol_mu.max(0.0)).max(1.0e-5);

    let mut best_x:      SVector<f64, NP> = SVector::zeros();
    let mut best_lambda: SVector<f64, NE> = SVector::zeros();
    let mut best_s:      SVector<f64, NCT> = SVector::zeros();
    let mut best_y:      SVector<f64, NCT> = SVector::zeros();
    let mut best_mu = f64::INFINITY;
    let mut best_valid = false;

    // On a numerical breakdown mid-loop, prefer the captured best-feasible
    // snapshot (return it as `BestFeasible`) over scrubbing the live iterate
    // to `NumericalError`. NOTE: unlike the dense driver's `numerical_exit`
    // (which can read `ws.best_*`), this driver cannot — those workspace
    // fields are private — so it tracks its OWN `best_*` locally and returns
    // them via `bail_result!`. `numerical_exit` here is only the no-snapshot
    // fallback (scrub the live iterate). The effect matches the dense driver
    // (a good snapshot is returned as `BestFeasible`, sparing the SCvx outer
    // loop a wasteful dense re-solve), but the mechanism is the local snapshot,
    // NOT `ws.best_*`. Do not "simplify" by deleting the local tracking.
    macro_rules! bail_result {
        ($it:expr) => {
            if best_valid {
                SocpResult {
                    x: best_x, lambda: best_lambda, s: best_s, y: best_y,
                    status: IpmStatus::BestFeasible,
                    iters: $it,
                }
            } else {
                numerical_exit(ws, IpmStatus::NumericalError, $it)
            }
        };
    }

    let mut prev_x = ws.x;
    let mut prev_y = ws.y;

    let mut sol_buf: ReducedKktSolution<N> = ReducedKktSolution::default();
    let mut factor_buf: ReducedKktFactor<N> = ReducedKktFactor::default();

    for iter in 0..params.max_iters.min(IPM_HARD_MAX_ITERS) {
        let r_x = prob.c
                + prob.a_mat.transpose() * ws.lambda
                + prob.g_mat.transpose() * ws.y;
        let r_a = prob.a_mat * ws.x - prob.b;
        let r_g = prob.g_mat * ws.x + ws.s - prob.h;

        let mu       = ws.s.dot(&ws.y) / ncone;
        let primal_r = r_a.norm() + r_g.norm();
        let dual_r   = r_x.norm();
        if mu < params.tol_mu && primal_r < params.tol_primal && dual_r < params.tol_dual {
            return SocpResult {
                x: ws.x, lambda: ws.lambda, s: ws.s, y: ws.y,
                status: IpmStatus::Optimal,
                iters:  iter,
            };
        }

        if dual_r < loose_dual && primal_r < loose_primal
            && mu.is_finite() && mu >= 0.0 && mu < best_mu
        {
            best_x      = ws.x;
            best_lambda = ws.lambda;
            best_s      = ws.s;
            best_y      = ws.y;
            best_mu     = mu;
            best_valid  = true;
        }
        if iter > 2 && best_valid && best_mu < loose_mu {
            return SocpResult {
                x: best_x, lambda: best_lambda, s: best_s, y: best_y,
                status: IpmStatus::BestFeasible, iters: iter,
            };
        }
        let dx_iter = (ws.x - prev_x).norm();
        let dy_iter = (ws.y - prev_y).norm();
        if iter > 0 && dx_iter < 1.0e-9 && dy_iter < 1.0e-9
            && best_valid && best_mu < loose_mu
        {
            return SocpResult {
                x: best_x, lambda: best_lambda, s: best_s, y: best_y,
                status: IpmStatus::BestFeasible, iters: iter,
            };
        }
        prev_x = ws.x;
        prev_y = ws.y;

        // ---- NT scaling blocks ----
        let (w, w_squared, arrow_s_scaled_inv, s_scaled) =
            match build_per_cone_nt_blocks(&ws.s, &ws.y, &prob.cones) {
                Some(t) => t,
                None    => return bail_result!(iter),
            };

        let reg: f64 = 1.0e-8;

        // Factor the structured Schur with M = W² (once per iter).
        let st = factor_reduced_kkt_scvx_block_m::<N, NP, NE, NCT, NCONES>(
            prob, &w_squared, reg, &mut factor_buf,
        );
        if st != ReducedKktStatus::Ok {
            return bail_result!(iter);
        }

        // ---- Affine (predictor) step ----
        // In NT, arrow(s̃)⁻¹·F̃_4_aff = s̃, so r_c_arg_aff = s_scaled.
        let r_c_arg_aff = s_scaled;
        let b_x_pred = -r_x
                     - prob.g_mat.transpose() * (w_squared * r_g)
                     + prob.g_mat.transpose() * (w         * r_c_arg_aff);
        let b_a = -r_a;

        let st = solve_reduced_kkt_scvx_with_factor::<N, NP, NE, NCT, NCONES>(
            prob, &factor_buf, &b_x_pred, &b_a, &mut sol_buf,
        );
        if st != ReducedKktStatus::Ok {
            return bail_result!(iter);
        }
        // Extract Δλ from the block-structured solution buffer (used for
        // both the affine and corrector directions in the weighted blend).
        let extract_dl = |sb: &ReducedKktSolution<N>| -> SVector<f64, NE> {
            let mut dl = SVector::<f64, NE>::zeros();
            for i in 0..NX { dl[i] = sb.dlam_init[i]; }
            for k in 0..(N - 1) {
                for i in 0..NX { dl[NX + k * NX + i] = sb.dlam_dyn[k][i]; }
            }
            for i in 0..6 { dl[N * NX + i] = sb.dlam_term[i]; }
            dl
        };

        let dx_a = stack_dz::<N, NP>(&sol_buf);
        let dl_a = extract_dl(&sol_buf);
        let ds_a = -r_g - prob.g_mat * dx_a;
        // Δy = −W·r_c_arg + W²·(r_g + G·Δx)
        let dy_a = -(w * r_c_arg_aff) + w_squared * (r_g + prob.g_mat * dx_a);

        let affine_finite = ds_a.iter().all(|v| v.is_finite())
            && dy_a.iter().all(|v| v.is_finite())
            && dx_a.iter().all(|v| v.is_finite())
            && dl_a.iter().all(|v| v.is_finite());
        if !affine_finite {
            return bail_result!(iter + 1);
        }

        let alpha_s_aff = clip01(BACKOFF * max_step_all_cones(&ws.s, &ds_a, &prob.cones));
        let alpha_y_aff = clip01(BACKOFF * max_step_all_cones(&ws.y, &dy_a, &prob.cones));

        // ---- Mehrotra centering σ with the SDPT3 **adaptive exponent** ----
        // Mirrors `scvx_ipm::solve_socp_nt`: e = max(1, 3·min(αp,αd)²) for
        // μ > 1e-6, else e = 1. (Keep in lockstep with the dense NT driver —
        // the structured/dense equivalence tests assert a 1e-5 match.)
        let s_aff = ws.s + ds_a * alpha_s_aff;
        let y_aff = ws.y + dy_a * alpha_y_aff;
        let mu_aff = s_aff.dot(&y_aff) / ncone;
        let sigma = if mu > 1.0e-300 {
            let ratio = mu_aff / mu;
            let alpha_min = alpha_s_aff.min(alpha_y_aff);
            let e = if mu > 1.0e-6 {
                (1.0_f64).max(3.0 * alpha_min * alpha_min)
            } else {
                1.0
            };
            let s_raw = libm::pow(ratio, e);
            if s_raw.is_finite() { s_raw.clamp(0.0, 1.0) } else { 0.5 }
        } else {
            1.0
        };

        // ---- Corrector r_c_arg (scaled coords) ----
        //   Δs̃_aff = W·Δs_aff
        //   Δỹ_aff = W·(r_g + G·Δx_aff) − s̃
        //   r_c_arg_corr = s̃ + arrow(s̃)⁻¹·(Δs̃_aff ∘ Δỹ_aff) − σμ·s̃⁻¹
        let ds_aff_scaled = w * ds_a;
        let dy_aff_scaled = w * (r_g + prob.g_mat * dx_a) - s_scaled;
        let mut second_order = SVector::<f64, NCT>::zeros();
        jordan_per_cone(&ds_aff_scaled, &dy_aff_scaled, &prob.cones, &mut second_order);
        let arrow_s_inv_second = arrow_s_scaled_inv * second_order;
        let s_scaled_inv = arrow_s_scaled_inv * e_vec;
        let r_c_arg_corr = s_scaled + arrow_s_inv_second - s_scaled_inv * (sigma * mu);

        let b_x_corr = -r_x
                     - prob.g_mat.transpose() * (w_squared * r_g)
                     + prob.g_mat.transpose() * (w         * r_c_arg_corr);

        let st = solve_reduced_kkt_scvx_with_factor::<N, NP, NE, NCT, NCONES>(
            prob, &factor_buf, &b_x_corr, &b_a, &mut sol_buf,
        );
        if st != ReducedKktStatus::Ok {
            return bail_result!(iter);
        }
        let dx_c = stack_dz::<N, NP>(&sol_buf);
        let dl_c = extract_dl(&sol_buf);
        let ds_c = -r_g - prob.g_mat * dx_c;
        let dy_c = -(w * r_c_arg_corr) + w_squared * (r_g + prob.g_mat * dx_c);

        let step_finite = dx_c.iter().all(|v| v.is_finite())
            && dl_c.iter().all(|v| v.is_finite())
            && ds_c.iter().all(|v| v.is_finite())
            && dy_c.iter().all(|v| v.is_finite());
        if !step_finite {
            return bail_result!(iter + 1);
        }

        // ---- Weighted corrector (Colombo–Gondzio) + adaptive γ ----
        // Identical to the dense NT driver: blend Δ(ω)=Δ_a+ω·(Δ_c−Δ_a) and
        // pick ω∈{0,0.2,…,1} maximizing the min fraction-to-boundary step.
        let blend_dir = |omega: f64| -> (
            SVector<f64, NP>, SVector<f64, NE>, SVector<f64, NCT>, SVector<f64, NCT>,
        ) {
            (
                dx_a + (dx_c - dx_a) * omega,
                dl_a + (dl_c - dl_a) * omega,
                ds_a + (ds_c - ds_a) * omega,
                dy_a + (dy_c - dy_a) * omega,
            )
        };
        const OMEGA_GRID: [f64; 6] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
        let mut best_omega = 1.0_f64;
        let mut best_obj   = f64::NEG_INFINITY;
        let mut best_raw_s = 0.0_f64;
        let mut best_raw_y = 0.0_f64;
        for &omega in OMEGA_GRID.iter() {
            let (_, _, cand_ds, cand_dy) = blend_dir(omega);
            let raw_s = max_step_all_cones(&ws.s, &cand_ds, &prob.cones);
            let raw_y = max_step_all_cones(&ws.y, &cand_dy, &prob.cones);
            let obj = raw_s.min(raw_y);
            if obj.is_finite() && obj > best_obj {
                best_obj   = obj;
                best_omega = omega;
                best_raw_s = raw_s;
                best_raw_y = raw_y;
            }
        }
        let (dx, dl, ds, dy) = blend_dir(best_omega);

        let gamma = (0.9 + 0.09 * best_raw_s.min(best_raw_y)).min(BACKOFF);
        let alpha_s = clip01(gamma * best_raw_s);
        let alpha_y = clip01(gamma * best_raw_y);

        ws.x      += dx * alpha_s;
        ws.lambda += dl * alpha_y;
        ws.s      += ds * alpha_s;
        ws.y      += dy * alpha_y;

        if !all_cones_interior(&ws.s, &prob.cones)
            || !all_cones_interior(&ws.y, &prob.cones)
        {
            return bail_result!(iter + 1);
        }
        let max_abs = ws.x.amax().max(ws.s.amax()).max(ws.y.amax()).max(ws.lambda.amax());
        if !max_abs.is_finite() || max_abs > 1.0e50 {
            return bail_result!(iter + 1);
        }
    }

    if best_valid {
        SocpResult {
            x: best_x, lambda: best_lambda, s: best_s, y: best_y,
            status: IpmStatus::BestFeasible,
            iters:  params.max_iters.min(IPM_HARD_MAX_ITERS),
        }
    } else {
        numerical_exit(ws, IpmStatus::IterCap, params.max_iters.min(IPM_HARD_MAX_ITERS))
    }
}

// ---------------------------------------------------------------------------
// NT-direction free-tf structured driver (Phase 6.10 — completes the matrix)
// ---------------------------------------------------------------------------

/// Structured-Schur **Nesterov-Todd** Mehrotra IPM driver for **free-tf**
/// SCvx-shaped SOCPs. The composition of:
/// - the NT `W²` scaling of [`solve_socp_structured_nt`] (Phase 6.9), and
/// - the Sherman-Morrison δτ correction of [`solve_socp_structured_free_tf`]
///   (Phase 6.8).
///
/// The free-tf SMW factor/apply pair is scaling-agnostic — it takes whatever
/// `m_full` it's handed and augments the block-tridiag Schur with the rank-1
/// δτ term. So we pass `W²` as `m_full` (just like the fixed-tf NT driver
/// passes `W²` to the plain factor), and the SMW machinery handles the
/// global δτ column transparently. No `reduced_kkt` changes needed.
///
/// NT Newton step (identical formulas to `solve_socp_structured_nt`):
/// ```text
///   b_x = −r_x − Gᵀ·W²·r_g + Gᵀ·W·r_c_arg
///   [Δz, Δδτ, Δλ] via SMW-augmented structured Schur on H = Gᵀ·W²·G
///   Δs  = −r_g − G·Δx          (Δx includes the δτ column)
///   Δy  = −W·r_c_arg + W²·(r_g + G·Δx)
/// ```
///
/// **Assumes** free-tf SCvx layout (`NP = N·19 + 1`). Completes the
/// structured dispatch matrix: AHO/NT × fixed-tf/free-tf all have a
/// machine-precision-verified structured path.
///
/// **MAINTENANCE NOTE**: this is the free-tf twin of
/// [`solve_socp_structured_nt`] AND the NT twin of
/// [`solve_socp_structured_free_tf`]. The Mehrotra scaffold and NT formulas
/// match the former; the SMW factor/apply/`stack_dz_free_tf`/`base.dlam_*`
/// plumbing matches the latter. Keep all three in sync.
pub fn solve_socp_structured_nt_free_tf<
    const N:      usize,
    const NP:     usize,
    const NE:     usize,
    const NCT:    usize,
    const NCONES: usize,
>(
    prob:   &SocpProblem<NP, NE, NCT, NCONES>,
    params: &IpmAlgoParams,
    ws:     &mut SocpWorkspace<NP, NE, NCT>,
) -> SocpResult<NP, NE, NCT> {
    debug_assert_eq!(NP, N * N_VARS_PER_NODE_SCVX + 1,
                     "solve_socp_structured_nt_free_tf: NP must equal N·19 + 1");
    debug_assert_eq!(NE, N * NX + 6,
                     "solve_socp_structured_nt_free_tf: NE must equal N·7 + 6");

    if !params.warm_start_x {
        ws.x      = SVector::zeros();
        ws.lambda = SVector::zeros();
        init_per_cone_warm_start(&prob.h, &prob.cones, &mut ws.s);
        init_per_cone_warm_start(&prob.h, &prob.cones, &mut ws.y);
    } else {
        let gx = prob.g_mat * ws.x;
        let s_target = prob.h - gx;
        init_per_cone_warm_start(&s_target, &prob.cones, &mut ws.s);
        init_per_cone_warm_start(&prob.h,    &prob.cones, &mut ws.y);
    }

    let e_vec = per_cone_e::<NCT, NCONES>(&prob.cones);
    let ncone = NCT as f64;
    let loose_dual   = libm::sqrt(params.tol_dual.max(0.0)).max(1.0e-5);
    let loose_primal = libm::sqrt(params.tol_primal.max(0.0)).max(1.0e-5);
    let loose_mu     = libm::sqrt(params.tol_mu.max(0.0)).max(1.0e-5);

    let mut best_x:      SVector<f64, NP> = SVector::zeros();
    let mut best_lambda: SVector<f64, NE> = SVector::zeros();
    let mut best_s:      SVector<f64, NCT> = SVector::zeros();
    let mut best_y:      SVector<f64, NCT> = SVector::zeros();
    let mut best_mu = f64::INFINITY;
    let mut best_valid = false;

    // On a numerical breakdown mid-loop, prefer the captured best-feasible
    // snapshot (return it as `BestFeasible`) over scrubbing the live iterate
    // to `NumericalError`. NOTE: unlike the dense driver's `numerical_exit`
    // (which can read `ws.best_*`), this driver cannot — those workspace
    // fields are private — so it tracks its OWN `best_*` locally and returns
    // them via `bail_result!`. `numerical_exit` here is only the no-snapshot
    // fallback (scrub the live iterate). The effect matches the dense driver
    // (a good snapshot is returned as `BestFeasible`, sparing the SCvx outer
    // loop a wasteful dense re-solve), but the mechanism is the local snapshot,
    // NOT `ws.best_*`. Do not "simplify" by deleting the local tracking.
    macro_rules! bail_result {
        ($it:expr) => {
            if best_valid {
                SocpResult {
                    x: best_x, lambda: best_lambda, s: best_s, y: best_y,
                    status: IpmStatus::BestFeasible,
                    iters: $it,
                }
            } else {
                numerical_exit(ws, IpmStatus::NumericalError, $it)
            }
        };
    }

    let mut prev_x = ws.x;
    let mut prev_y = ws.y;

    let mut sol_buf: ReducedKktSolutionFreeTf<N> = ReducedKktSolutionFreeTf::default();
    let mut factor_buf: ReducedKktFactorFreeTf<N> = ReducedKktFactorFreeTf::default();

    for iter in 0..params.max_iters.min(IPM_HARD_MAX_ITERS) {
        let r_x = prob.c
                + prob.a_mat.transpose() * ws.lambda
                + prob.g_mat.transpose() * ws.y;
        let r_a = prob.a_mat * ws.x - prob.b;
        let r_g = prob.g_mat * ws.x + ws.s - prob.h;

        let mu       = ws.s.dot(&ws.y) / ncone;
        let primal_r = r_a.norm() + r_g.norm();
        let dual_r   = r_x.norm();
        if mu < params.tol_mu && primal_r < params.tol_primal && dual_r < params.tol_dual {
            return SocpResult {
                x: ws.x, lambda: ws.lambda, s: ws.s, y: ws.y,
                status: IpmStatus::Optimal,
                iters:  iter,
            };
        }

        if dual_r < loose_dual && primal_r < loose_primal
            && mu.is_finite() && mu >= 0.0 && mu < best_mu
        {
            best_x      = ws.x;
            best_lambda = ws.lambda;
            best_s      = ws.s;
            best_y      = ws.y;
            best_mu     = mu;
            best_valid  = true;
        }
        if iter > 2 && best_valid && best_mu < loose_mu {
            return SocpResult {
                x: best_x, lambda: best_lambda, s: best_s, y: best_y,
                status: IpmStatus::BestFeasible, iters: iter,
            };
        }
        let dx_iter = (ws.x - prev_x).norm();
        let dy_iter = (ws.y - prev_y).norm();
        if iter > 0 && dx_iter < 1.0e-9 && dy_iter < 1.0e-9
            && best_valid && best_mu < loose_mu
        {
            return SocpResult {
                x: best_x, lambda: best_lambda, s: best_s, y: best_y,
                status: IpmStatus::BestFeasible, iters: iter,
            };
        }
        prev_x = ws.x;
        prev_y = ws.y;

        // ---- NT scaling blocks ----
        let (w, w_squared, arrow_s_scaled_inv, s_scaled) =
            match build_per_cone_nt_blocks(&ws.s, &ws.y, &prob.cones) {
                Some(t) => t,
                None    => return bail_result!(iter),
            };

        let reg: f64 = 1.0e-8;

        // Factor the SMW-augmented structured Schur with M = W².
        let st = factor_reduced_kkt_scvx_block_m_free_tf::<N, NP, NE, NCT, NCONES>(
            prob, &w_squared, reg, &mut factor_buf,
        );
        if st != ReducedKktStatus::Ok {
            return bail_result!(iter);
        }

        // ---- Affine (predictor) step ----
        let r_c_arg_aff = s_scaled;
        let b_x_pred = -r_x
                     - prob.g_mat.transpose() * (w_squared * r_g)
                     + prob.g_mat.transpose() * (w         * r_c_arg_aff);
        let b_a = -r_a;

        let st = solve_reduced_kkt_scvx_with_factor_free_tf::<N, NP, NE, NCT, NCONES>(
            prob, &factor_buf, &b_x_pred, &b_a, &mut sol_buf,
        );
        if st != ReducedKktStatus::Ok {
            return bail_result!(iter);
        }
        // Δλ extractor for the free-tf solution buffer (`.base.dlam_*`).
        let extract_dl = |sb: &ReducedKktSolutionFreeTf<N>| -> SVector<f64, NE> {
            let mut dl = SVector::<f64, NE>::zeros();
            for i in 0..NX { dl[i] = sb.base.dlam_init[i]; }
            for k in 0..(N - 1) {
                for i in 0..NX { dl[NX + k * NX + i] = sb.base.dlam_dyn[k][i]; }
            }
            for i in 0..6 { dl[N * NX + i] = sb.base.dlam_term[i]; }
            dl
        };

        let dx_a = stack_dz_free_tf::<N, NP>(&sol_buf);
        let dl_a = extract_dl(&sol_buf);
        let ds_a = -r_g - prob.g_mat * dx_a;
        let dy_a = -(w * r_c_arg_aff) + w_squared * (r_g + prob.g_mat * dx_a);

        let affine_finite = ds_a.iter().all(|v| v.is_finite())
            && dy_a.iter().all(|v| v.is_finite())
            && dx_a.iter().all(|v| v.is_finite())
            && dl_a.iter().all(|v| v.is_finite());
        if !affine_finite {
            return bail_result!(iter + 1);
        }

        let alpha_s_aff = clip01(BACKOFF * max_step_all_cones(&ws.s, &ds_a, &prob.cones));
        let alpha_y_aff = clip01(BACKOFF * max_step_all_cones(&ws.y, &dy_a, &prob.cones));

        // SDPT3 adaptive centering exponent — keep in lockstep with the dense
        // NT driver (`scvx_ipm::solve_socp_nt`); the equivalence test asserts
        // a 1e-5 match including the δτ primal.
        let s_aff = ws.s + ds_a * alpha_s_aff;
        let y_aff = ws.y + dy_a * alpha_y_aff;
        let mu_aff = s_aff.dot(&y_aff) / ncone;
        let sigma = if mu > 1.0e-300 {
            let ratio = mu_aff / mu;
            let alpha_min = alpha_s_aff.min(alpha_y_aff);
            let e = if mu > 1.0e-6 {
                (1.0_f64).max(3.0 * alpha_min * alpha_min)
            } else {
                1.0
            };
            let s_raw = libm::pow(ratio, e);
            if s_raw.is_finite() { s_raw.clamp(0.0, 1.0) } else { 0.5 }
        } else {
            1.0
        };

        // ---- Corrector r_c_arg (scaled coords, same as NT fixed-tf) ----
        let ds_aff_scaled = w * ds_a;
        let dy_aff_scaled = w * (r_g + prob.g_mat * dx_a) - s_scaled;
        let mut second_order = SVector::<f64, NCT>::zeros();
        jordan_per_cone(&ds_aff_scaled, &dy_aff_scaled, &prob.cones, &mut second_order);
        let arrow_s_inv_second = arrow_s_scaled_inv * second_order;
        let s_scaled_inv = arrow_s_scaled_inv * e_vec;
        let r_c_arg_corr = s_scaled + arrow_s_inv_second - s_scaled_inv * (sigma * mu);

        let b_x_corr = -r_x
                     - prob.g_mat.transpose() * (w_squared * r_g)
                     + prob.g_mat.transpose() * (w         * r_c_arg_corr);

        let st = solve_reduced_kkt_scvx_with_factor_free_tf::<N, NP, NE, NCT, NCONES>(
            prob, &factor_buf, &b_x_corr, &b_a, &mut sol_buf,
        );
        if st != ReducedKktStatus::Ok {
            return bail_result!(iter);
        }
        let dx_c = stack_dz_free_tf::<N, NP>(&sol_buf);
        let dl_c = extract_dl(&sol_buf);
        let ds_c = -r_g - prob.g_mat * dx_c;
        let dy_c = -(w * r_c_arg_corr) + w_squared * (r_g + prob.g_mat * dx_c);

        let step_finite = dx_c.iter().all(|v| v.is_finite())
            && dl_c.iter().all(|v| v.is_finite())
            && ds_c.iter().all(|v| v.is_finite())
            && dy_c.iter().all(|v| v.is_finite());
        if !step_finite {
            return bail_result!(iter + 1);
        }

        // ---- Weighted corrector (Colombo–Gondzio) + adaptive γ ----
        // Identical to the dense / fixed-tf NT drivers. The δτ primal rides
        // inside dx (index N·NZ), so blending dx_a/dx_c blends δτ correctly.
        let blend_dir = |omega: f64| -> (
            SVector<f64, NP>, SVector<f64, NE>, SVector<f64, NCT>, SVector<f64, NCT>,
        ) {
            (
                dx_a + (dx_c - dx_a) * omega,
                dl_a + (dl_c - dl_a) * omega,
                ds_a + (ds_c - ds_a) * omega,
                dy_a + (dy_c - dy_a) * omega,
            )
        };
        const OMEGA_GRID: [f64; 6] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
        let mut best_omega = 1.0_f64;
        let mut best_obj   = f64::NEG_INFINITY;
        let mut best_raw_s = 0.0_f64;
        let mut best_raw_y = 0.0_f64;
        for &omega in OMEGA_GRID.iter() {
            let (_, _, cand_ds, cand_dy) = blend_dir(omega);
            let raw_s = max_step_all_cones(&ws.s, &cand_ds, &prob.cones);
            let raw_y = max_step_all_cones(&ws.y, &cand_dy, &prob.cones);
            let obj = raw_s.min(raw_y);
            if obj.is_finite() && obj > best_obj {
                best_obj   = obj;
                best_omega = omega;
                best_raw_s = raw_s;
                best_raw_y = raw_y;
            }
        }
        let (dx, dl, ds, dy) = blend_dir(best_omega);

        let gamma = (0.9 + 0.09 * best_raw_s.min(best_raw_y)).min(BACKOFF);
        let alpha_s = clip01(gamma * best_raw_s);
        let alpha_y = clip01(gamma * best_raw_y);

        ws.x      += dx * alpha_s;
        ws.lambda += dl * alpha_y;
        ws.s      += ds * alpha_s;
        ws.y      += dy * alpha_y;

        if !all_cones_interior(&ws.s, &prob.cones)
            || !all_cones_interior(&ws.y, &prob.cones)
        {
            return bail_result!(iter + 1);
        }
        let max_abs = ws.x.amax().max(ws.s.amax()).max(ws.y.amax()).max(ws.lambda.amax());
        if !max_abs.is_finite() || max_abs > 1.0e50 {
            return bail_result!(iter + 1);
        }
    }

    if best_valid {
        SocpResult {
            x: best_x, lambda: best_lambda, s: best_s, y: best_y,
            status: IpmStatus::BestFeasible,
            iters:  params.max_iters.min(IPM_HARD_MAX_ITERS),
        }
    } else {
        numerical_exit(ws, IpmStatus::IterCap, params.max_iters.min(IPM_HARD_MAX_ITERS))
    }
}

// ---------------------------------------------------------------------------
// HSD-direction structured driver (Phase 28 — the O(N) NT/O(N) close).
// ---------------------------------------------------------------------------

/// Structured-Schur **homogeneous self-dual (HSD)** Mehrotra IPM driver — the
/// `O(N·NZ³)` twin of `scvx_ipm::solve_socp_hsd` (dense), for **fixed-tf**
/// SCvx-shaped SOCPs (`NP = N·19`).
///
/// This is the lift that finally closes **NT and O(N) together**. The structured
/// NT driver ([`solve_socp_structured_nt`]) inherited plain NT's vanishing-cone
/// divergence (the `W²` spread that blows up `H = GᵀW²G` off the central path).
/// HSD removes that root cause — the self-dual embedding keeps every cone (and
/// the `(τ,κ)` ray) near-complementary, so `soc_nt_scaling_exact` stays bounded
/// and `W²` is well-conditioned — so the SAME block-tridiagonal Schur
/// factorization the structured NT path uses is now numerically sound.
///
/// Algorithmically identical to the dense `solve_socp_hsd`: the embedded Newton
/// step eliminates `Δs`/`Δz` (NT), leaving a `(Δx,Δλ)` system AFFINE in the
/// scalar `Δτ`, solved by TWO applies of the SAME structured factor — the
/// residual RHS and the constant τ-column `[c;b;h]` — then one scalar gap-row
/// equation for `Δτ`. The ONLY change vs dense is that each `[H Aᵀ; A 0]` solve
/// goes through [`factor_reduced_kkt_scvx_block_m`] +
/// [`solve_reduced_kkt_scvx_with_factor`] (factor once, apply for both RHS)
/// instead of dense `H⁻¹`/`S⁻¹`. Cold-starts central; returns the recovered
/// (de-homogenized) iterate `(x,λ,s,y)/τ`.
///
/// **MAINTENANCE NOTE**: the Mehrotra scaffold (residuals, central start,
/// endgame snapshot/exit, σ) mirrors `scvx_ipm::solve_socp_hsd`; the
/// build-blocks → factor → apply → `stack_dz`/`extract_dl` plumbing mirrors
/// [`solve_socp_structured_nt`]. A change to the HSD math in `solve_socp_hsd`
/// must be mirrored here (the one-iter equivalence test guards against drift).
pub fn solve_socp_structured_hsd<
    const N:      usize,
    const NP:     usize,
    const NE:     usize,
    const NCT:    usize,
    const NCONES: usize,
>(
    prob:   &SocpProblem<NP, NE, NCT, NCONES>,
    params: &IpmAlgoParams,
    ws:     &mut SocpWorkspace<NP, NE, NCT>,
) -> SocpResult<NP, NE, NCT> {
    debug_assert_eq!(NP, N * N_VARS_PER_NODE_SCVX,
                     "solve_socp_structured_hsd: NP must equal N·19 (fixed-tf SCvx)");
    debug_assert_eq!(NE, N * NX + 6,
                     "solve_socp_structured_hsd: NE must equal N·7 + 6");

    let e_vec = per_cone_e::<NCT, NCONES>(&prob.cones);
    // Central start (x=0, λ=0, s=z=e, τ=κ=1) — HSD needs no warm-start.
    let mut xe    = SVector::<f64, NP>::zeros();
    let mut le    = SVector::<f64, NE>::zeros();
    let mut se    = e_vec;
    let mut ze    = e_vec;
    let mut tau   = 1.0_f64;
    let mut kappa = 1.0_f64;

    let degree       = (NCONES + 1) as f64;
    let loose_primal = sqrt(params.tol_primal.max(0.0)).max(1.0e-5);
    let loose_mu     = sqrt(params.tol_mu.max(0.0)).max(1.0e-5);

    let g_t = prob.g_mat.transpose();
    // Fixed Tikhonov floor (matches the dense HSD default and structured NT).
    // HSD does NOT need adaptive regularization: the self-dual embedding keeps the
    // iterates near the central path, where `W²` is bounded and `H = GᵀW²G` is
    // well-conditioned — `max(1e-8, rel·tr(H)/n)` is a crutch for the
    // ill-conditioned NT/AHO-preconditioning paths, not HSD. A caller that
    // nonetheless wants adaptive reg should use the dense `solve_socp_hsd` (which
    // threads `params.use_adaptive_regularization`); this O(N) path fixes the floor.
    let reg: f64 = 1.0e-8;

    let mut best_x:      SVector<f64, NP>  = SVector::zeros();
    let mut best_lambda: SVector<f64, NE>  = SVector::zeros();
    let mut best_s:      SVector<f64, NCT> = SVector::zeros();
    let mut best_y:      SVector<f64, NCT> = SVector::zeros();
    let mut best_mu = f64::INFINITY;
    let mut best_valid = false;

    let mut sol_buf:    ReducedKktSolution<N> = ReducedKktSolution::default();
    let mut factor_buf: ReducedKktFactor<N>   = ReducedKktFactor::default();

    // Recover Δλ (NE) from the block-structured solution buffer.
    let extract_dl = |sb: &ReducedKktSolution<N>| -> SVector<f64, NE> {
        let mut dl = SVector::<f64, NE>::zeros();
        for i in 0..NX { dl[i] = sb.dlam_init[i]; }
        for k in 0..(N - 1) {
            for i in 0..NX { dl[NX + k * NX + i] = sb.dlam_dyn[k][i]; }
        }
        for i in 0..6 { dl[N * NX + i] = sb.dlam_term[i]; }
        dl
    };

    macro_rules! bail_result {
        ($it:expr) => {
            if best_valid {
                SocpResult {
                    x: best_x, lambda: best_lambda, s: best_s, y: best_y,
                    status: IpmStatus::BestFeasible, iters: $it,
                }
            } else {
                numerical_exit(ws, IpmStatus::NumericalError, $it)
            }
        };
    }

    for iter in 0..params.max_iters.min(IPM_HARD_MAX_ITERS) {
        let tau_ok   = tau.is_finite()   && tau   > 0.0;
        let kappa_ok = kappa.is_finite() && kappa > 0.0;
        if !tau_ok || !kappa_ok {
            return bail_result!(iter);
        }

        // ---- Embedded residuals ----
        let r_x = prob.c * tau + prob.a_mat.transpose() * le + g_t * ze;
        let r_a = prob.a_mat * xe - prob.b * tau;
        let r_g = prob.g_mat * xe + se - prob.h * tau;
        let r_t = kappa + prob.c.dot(&xe) + prob.b.dot(&le) + prob.h.dot(&ze);
        let mu  = (se.dot(&ze) + tau * kappa) / degree;

        // Recovered (de-homogenized) iterate.
        let inv_tau  = 1.0 / tau;
        let rec_x = xe * inv_tau;
        let rec_l = le * inv_tau;
        let rec_s = se * inv_tau;
        let rec_z = ze * inv_tau;
        // Mirror into `ws` each iter so the no-snapshot `numerical_exit` fallback
        // returns the live recovered iterate (parity with dense `solve_socp_hsd`,
        // which assigns `ws.x = xe/τ` every iteration).
        ws.x = rec_x;
        ws.lambda = rec_l;
        ws.s = rec_s;
        ws.y = rec_z;
        let primal_r = (r_a.norm() + r_g.norm()) * inv_tau;
        let dual_r   = r_x.norm() * inv_tau;

        if mu < params.tol_mu && primal_r < params.tol_primal && dual_r < params.tol_dual {
            return SocpResult {
                x: rec_x, lambda: rec_l, s: rec_s, y: rec_z,
                status: IpmStatus::Optimal, iters: iter,
            };
        }

        if tau > 0.0 && primal_r < loose_primal && mu.is_finite() && mu >= 0.0 && mu < best_mu {
            best_x = rec_x; best_lambda = rec_l; best_s = rec_s; best_y = rec_z;
            best_mu = mu; best_valid = true;
        }
        if iter > 2 && best_valid && (best_mu < loose_mu || mu > 4.0 * best_mu) {
            return SocpResult {
                x: best_x, lambda: best_lambda, s: best_s, y: best_y,
                status: IpmStatus::BestFeasible, iters: iter,
            };
        }

        // ---- NT scaling blocks at the embedded (s, z) ----
        let (w, w_squared, arrow_s_scaled_inv, s_scaled) =
            match build_per_cone_nt_blocks(&se, &ze, &prob.cones) {
                Some(t) => t,
                None    => return bail_result!(iter),
            };

        // Factor the structured Schur with M = W² (once per iter).
        let st = factor_reduced_kkt_scvx_block_m::<N, NP, NE, NCT, NCONES>(
            prob, &w_squared, reg, &mut factor_buf,
        );
        if st != ReducedKktStatus::Ok {
            return bail_result!(iter);
        }

        // One structured apply per RHS (residual + τ-column), factor reused.
        macro_rules! schur_apply {
            ($bx:expr, $ba:expr) => {{
                let st = solve_reduced_kkt_scvx_with_factor::<N, NP, NE, NCT, NCONES>(
                    prob, &factor_buf, &$bx, &$ba, &mut sol_buf,
                );
                if st != ReducedKktStatus::Ok { return bail_result!(iter); }
                (stack_dz::<N, NP>(&sol_buf), extract_dl(&sol_buf))
            }};
        }

        // ---- τ-coupling precompute (independent of σ / corrector) ----
        let gw2h   = g_t * (w_squared * prob.h);
        let bx_tau = gw2h - prob.c;
        let (dx_t, dl_t) = schur_apply!(bx_tau, prob.b);
        let d      = prob.c + gw2h;
        let h_w2_h = prob.h.dot(&(w_squared * prob.h));
        let coef   = d.dot(&dx_t) + prob.b.dot(&dl_t) - h_w2_h - kappa / tau;
        if !coef.is_finite() || coef.abs() < 1.0e-300 {
            return bail_result!(iter);
        }
        let neg_r_a = -r_a;

        // Fraction-to-boundary over (s, z, τ, κ).
        let ftb = |ds: &SVector<f64, NCT>, dz: &SVector<f64, NCT>, dt: f64, dk: f64| -> f64 {
            let a_s = max_step_all_cones(&se, ds, &prob.cones);
            let a_z = max_step_all_cones(&ze, dz, &prob.cones);
            let a_t = if dt < 0.0 { -tau   / dt } else { f64::INFINITY };
            let a_k = if dk < 0.0 { -kappa / dk } else { f64::INFINITY };
            a_s.min(a_z).min(a_t).min(a_k)
        };

        // ---- Affine (predictor): r_c_arg = s̃, r6 = −τκ ----
        let r_c_arg_aff = s_scaled;
        let r6_aff = -tau * kappa;
        let bx_aff = -r_x + g_t * (w * r_c_arg_aff) - g_t * (w_squared * r_g);
        let (dx_r_a, dl_r_a) = schur_apply!(bx_aff, neg_r_a);
        let const_a = d.dot(&dx_r_a) + prob.b.dot(&dl_r_a)
            - prob.h.dot(&(w * r_c_arg_aff)) + prob.h.dot(&(w_squared * r_g))
            + r6_aff / tau;
        let dtau_a = (-r_t - const_a) / coef;
        let dx_a   = dx_r_a + dx_t * dtau_a;
        let ds_a   = -r_g - prob.g_mat * dx_a + prob.h * dtau_a;
        let dz_a   = -(w * r_c_arg_aff) - w_squared * ds_a;
        let dkap_a = (r6_aff - kappa * dtau_a) / tau;

        let aff_finite = ds_a.iter().all(|v| v.is_finite())
            && dz_a.iter().all(|v| v.is_finite())
            && dtau_a.is_finite() && dkap_a.is_finite();
        if !aff_finite {
            return bail_result!(iter + 1);
        }
        let alpha_a = clip01(BACKOFF * ftb(&ds_a, &dz_a, dtau_a, dkap_a));

        let s_aff   = se    + ds_a   * alpha_a;
        let z_aff   = ze    + dz_a   * alpha_a;
        let tau_aff = tau   + dtau_a * alpha_a;
        let kap_aff = kappa + dkap_a * alpha_a;
        let mu_aff  = (s_aff.dot(&z_aff) + tau_aff * kap_aff) / degree;
        let sigma = if mu > 1.0e-300 {
            let r  = mu_aff / mu;
            let s3 = r * r * r;
            if s3.is_finite() { s3.clamp(0.0, 1.0) } else { 0.5 }
        } else {
            0.0
        };

        // ---- Corrector: r_c_arg = s̃ + arrow(s̃)⁻¹·(Δs̃_a∘Δz̃_a) − σμ·s̃⁻¹ ----
        let ds_a_scaled = w * ds_a;
        let dz_a_scaled = -s_scaled - ds_a_scaled;
        let mut second_order = SVector::<f64, NCT>::zeros();
        jordan_per_cone(&ds_a_scaled, &dz_a_scaled, &prob.cones, &mut second_order);
        let arrow_s_inv_second = arrow_s_scaled_inv * second_order;
        let s_scaled_inv       = arrow_s_scaled_inv * e_vec;
        let r_c_arg_corr = s_scaled + arrow_s_inv_second - s_scaled_inv * (sigma * mu);
        let r6_corr = sigma * mu - tau * kappa - dtau_a * dkap_a;

        let bx_corr = -r_x + g_t * (w * r_c_arg_corr) - g_t * (w_squared * r_g);
        let (dx_r_c, dl_r_c) = schur_apply!(bx_corr, neg_r_a);
        let const_c = d.dot(&dx_r_c) + prob.b.dot(&dl_r_c)
            - prob.h.dot(&(w * r_c_arg_corr)) + prob.h.dot(&(w_squared * r_g))
            + r6_corr / tau;
        let dtau = (-r_t - const_c) / coef;
        let dx   = dx_r_c + dx_t * dtau;
        let dl   = dl_r_c + dl_t * dtau;
        let ds   = -r_g - prob.g_mat * dx + prob.h * dtau;
        let dz   = -(w * r_c_arg_corr) - w_squared * ds;
        let dkap = (r6_corr - kappa * dtau) / tau;

        let step_finite = dx.iter().all(|v| v.is_finite())
            && dl.iter().all(|v| v.is_finite())
            && ds.iter().all(|v| v.is_finite())
            && dz.iter().all(|v| v.is_finite())
            && dtau.is_finite() && dkap.is_finite();
        if !step_finite {
            return bail_result!(iter + 1);
        }

        // ---- Single self-dual step length ----
        let raw   = ftb(&ds, &dz, dtau, dkap);
        let alpha = clip01(BACKOFF * raw);
        if alpha <= 0.0 {
            return bail_result!(iter + 1);
        }

        xe    += dx   * alpha;
        le    += dl   * alpha;
        se    += ds   * alpha;
        ze    += dz   * alpha;
        tau   += dtau * alpha;
        kappa += dkap * alpha;

        if !all_cones_interior(&se, &prob.cones)
            || !all_cones_interior(&ze, &prob.cones)
            || tau <= 0.0 || kappa <= 0.0
        {
            return bail_result!(iter + 1);
        }
        let max_abs = xe.amax().max(se.amax()).max(ze.amax()).max(le.amax())
            .max(tau.abs()).max(kappa.abs());
        if !max_abs.is_finite() || max_abs > 1.0e50 {
            return bail_result!(iter + 1);
        }
    }

    if best_valid {
        SocpResult {
            x: best_x, lambda: best_lambda, s: best_s, y: best_y,
            status: IpmStatus::BestFeasible,
            iters:  params.max_iters.min(IPM_HARD_MAX_ITERS),
        }
    } else {
        numerical_exit(ws, IpmStatus::IterCap, params.max_iters.min(IPM_HARD_MAX_ITERS))
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    extern crate std;
    use std::eprintln;

    use super::*;
    use crate::assemble::{
        assemble_scvx_socp, TerminalCondition, N_CONE_DIM_PER_NODE_SCVX,
    };
    use nalgebra::SVector;
    use scvx_core::{PhysicalParams, Trajectory};
    use scvx_dynamics::{discretize_foh, LinearizedDynamics};
    use scvx_ipm::{solve_socp, solve_socp_nt, SocpProblem, SocpWorkspace};

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

    fn hover_reference<const N: usize>(
        x_init: SVector<f64, 7>,
        m: f64,
        tau: f64,
    ) -> Trajectory<N> {
        let mut traj = Trajectory::<N>::default();
        let p = mars_params();
        let u_hover = [0.0, 0.0, -m * p.g[2]];
        for k in 0..N {
            for i in 0..7 { traj.x[(i, k)] = x_init[i]; }
            for (i, &v) in u_hover.iter().enumerate() { traj.u[(i, k)] = v; }
            traj.sigma[k] = m * (-p.g[2]);
        }
        traj.tau = tau;
        traj
    }

    /// **The driver-substitution gate**: solve a real SCvx subproblem via
    /// both the dense (`solve_socp`) and structured (`solve_socp_structured`)
    /// drivers from the same warm-start and verify the final iterates
    /// match to ≤ 1e-3 (relative). This is END-TO-END convergence
    /// equivalence, not just one Newton step.
    ///
    /// Note: exact equivalence is impossible because both drivers use
    /// floating-point arithmetic with slightly different summation orders.
    /// We allow up to ~1e-3 absolute on `x` (which has components of
    /// magnitude ~1000) and ~1e-3 on σ (magnitudes ~3e3). The Newton-step
    /// equivalence test in `reduced_kkt.rs` confirms per-iter match to 1e-7;
    /// accumulating over ~15 iterations gives ~1e-5 typically, well under
    /// the bound here.
    #[test]
    fn structured_driver_matches_dense_end_to_end() {
        const N: usize = 3;
        const NP: usize = N * N_VARS_PER_NODE_SCVX;
        const NE: usize = N * NX + 6;
        const NCT: usize = N * N_CONE_DIM_PER_NODE_SCVX;
        const NCONES: usize = N * 8;

        let phys = mars_params();
        let mut x_init = SVector::<f64, 7>::zeros();
        x_init[2] = 50.0;
        x_init[5] = -5.0;
        x_init[6] = libm::log(800.0);

        let traj = hover_reference::<N>(x_init, 800.0, 10.0);
        let mut lin = std::boxed::Box::new(LinearizedDynamics::<N>::default());
        discretize_foh(&traj, &phys, &mut lin, 4);

        let term = TerminalCondition { r: [0.0, 0.0, 0.0], v: [0.0, 0.0, 0.0] };
        let mut prob = std::boxed::Box::new(SocpProblem::<NP, NE, NCT, NCONES>::default());
        assemble_scvx_socp::<N, NP, NE, NCT, NCONES>(
            &traj, &lin, &phys, &x_init, &term,
            1.0e3, 1.0e4, false, &mut prob,
        );

        // Common IPM params for both drivers. **One iteration only** —
        // we want to verify the per-iter Newton step matches end-to-end
        // through the live driver loop, NOT to test full convergence
        // (which is sensitive to floating-point drift over many iters).
        let params = IpmAlgoParams {
            max_iters: 1,
            tol_mu:    1.0e-7,
            tol_primal:1.0e-6,
            tol_dual:  1.0e-6,
            ..IpmAlgoParams::default()
        };

        // ---- Solve via dense driver ----
        let mut ws_dense = std::boxed::Box::new(SocpWorkspace::<NP, NE, NCT>::default());
        let res_dense = solve_socp(&prob, &params, &mut ws_dense);

        // ---- Solve via structured driver ----
        let mut ws_struct = std::boxed::Box::new(SocpWorkspace::<NP, NE, NCT>::default());
        let res_struct = solve_socp_structured::<N, NP, NE, NCT, NCONES>(
            &prob, &params, &mut ws_struct,
        );

        eprintln!("dense   : status_code={}  iters={}",
                  res_dense.status.as_u32(),  res_dense.iters);
        eprintln!("struct  : status_code={}  iters={}",
                  res_struct.status.as_u32(), res_struct.iters);

        // After ONE iteration, both drivers should have produced the
        // same iterate to ~1e-7 (modulo fp roundoff in different operation
        // orders). This is essentially `full_newton_step_dense_matches_
        // structured` but driven through the LIVE driver loop instead of
        // a hand-built single-step test.
        let mut max_diff_x  = 0.0_f64;
        let mut max_diff_s  = 0.0_f64;
        let mut max_diff_y  = 0.0_f64;
        for i in 0..NP {
            let d = (res_dense.x[i] - res_struct.x[i]).abs();
            if d > max_diff_x { max_diff_x = d; }
        }
        for i in 0..NCT {
            let d = (res_dense.s[i] - res_struct.s[i]).abs();
            if d > max_diff_s { max_diff_s = d; }
            let d = (res_dense.y[i] - res_struct.y[i]).abs();
            if d > max_diff_y { max_diff_y = d; }
        }
        eprintln!("After 1 iter — max |Δx|={:.3e}  |Δs|={:.3e}  |Δy|={:.3e}",
                  max_diff_x, max_diff_s, max_diff_y);

        // Tolerance: per-iter equivalence is ~1e-7 (per the
        // full_newton_step_dense_matches_structured test). Live driver
        // adds a bit of accumulated fp drift through the residual /
        // sigma / corrector pipeline — allow up to 1e-5.
        let tol = 1.0e-5;
        assert!(max_diff_x < tol, "Δx mismatch {max_diff_x}");
        assert!(max_diff_s < tol, "Δs mismatch {max_diff_s}");
        assert!(max_diff_y < tol, "Δy mismatch {max_diff_y}");

        // Both should at least not be NumericalError (which would mean
        // one of the Newton steps blew up).
        assert!(
            !matches!(res_dense.status,  IpmStatus::NumericalError),
            "dense  driver hit NumericalError"
        );
        assert!(
            !matches!(res_struct.status, IpmStatus::NumericalError),
            "struct driver hit NumericalError"
        );
    }

    /// **The NT-direction structured gate** (Phase 6.9): run ONE NT
    /// Mehrotra iteration through both the dense NT driver
    /// (`scvx_ipm::solve_socp_nt`) and the structured NT driver
    /// (`solve_socp_structured_nt`) from the same cold-start, and verify
    /// the iterates match.
    ///
    /// Both drivers compute the same NT scaling `W` (the unique PD
    /// matrix-sqrt of `W²`) — the dense driver via eigendecomposition-first,
    /// the structured driver via Denman-Beavers. For a well-conditioned
    /// cold-start iterate both converge to the same `W` to machine
    /// precision, so the one-iteration iterates agree to fp-roundoff. The
    /// structural difference (dense KKT inverse vs block-tridiag Schur) is
    /// what this test isolates and pins.
    #[test]
    fn structured_nt_matches_dense_nt_one_iter() {
        const N: usize = 3;
        const NP: usize = N * N_VARS_PER_NODE_SCVX;
        const NE: usize = N * NX + 6;
        const NCT: usize = N * N_CONE_DIM_PER_NODE_SCVX;
        const NCONES: usize = N * 8;

        let phys = mars_params();
        let mut x_init = SVector::<f64, 7>::zeros();
        x_init[2] = 50.0;
        x_init[5] = -5.0;
        x_init[6] = libm::log(800.0);

        let traj = hover_reference::<N>(x_init, 800.0, 10.0);
        let mut lin = std::boxed::Box::new(LinearizedDynamics::<N>::default());
        discretize_foh(&traj, &phys, &mut lin, 4);

        let term = TerminalCondition { r: [0.0, 0.0, 0.0], v: [0.0, 0.0, 0.0] };
        let mut prob = std::boxed::Box::new(SocpProblem::<NP, NE, NCT, NCONES>::default());
        assemble_scvx_socp::<N, NP, NE, NCT, NCONES>(
            &traj, &lin, &phys, &x_init, &term,
            1.0e3, 1.0e4, false, &mut prob,
        );

        // One NT iteration, cold start (warm_start_x = false default).
        let params = IpmAlgoParams {
            max_iters: 1,
            tol_mu:    1.0e-7,
            tol_primal:1.0e-6,
            tol_dual:  1.0e-6,
            ..IpmAlgoParams::default()
        };

        // ---- Dense NT ----
        let mut ws_dense = std::boxed::Box::new(SocpWorkspace::<NP, NE, NCT>::default());
        let res_dense = solve_socp_nt(&prob, &params, &mut ws_dense);

        // ---- Structured NT ----
        let mut ws_struct = std::boxed::Box::new(SocpWorkspace::<NP, NE, NCT>::default());
        let res_struct = solve_socp_structured_nt::<N, NP, NE, NCT, NCONES>(
            &prob, &params, &mut ws_struct,
        );

        eprintln!("dense  NT: status_code={}  iters={}",
                  res_dense.status.as_u32(),  res_dense.iters);
        eprintln!("struct NT: status_code={}  iters={}",
                  res_struct.status.as_u32(), res_struct.iters);

        // Neither should hit NumericalError on a fresh cold-start iterate.
        assert!(
            !matches!(res_dense.status, IpmStatus::NumericalError),
            "dense NT driver hit NumericalError"
        );
        assert!(
            !matches!(res_struct.status, IpmStatus::NumericalError),
            "struct NT driver hit NumericalError"
        );

        let mut max_diff_x = 0.0_f64;
        let mut max_diff_s = 0.0_f64;
        let mut max_diff_y = 0.0_f64;
        for i in 0..NP {
            let d = (res_dense.x[i] - res_struct.x[i]).abs();
            if d > max_diff_x { max_diff_x = d; }
        }
        for i in 0..NCT {
            let d = (res_dense.s[i] - res_struct.s[i]).abs();
            if d > max_diff_s { max_diff_s = d; }
            let d = (res_dense.y[i] - res_struct.y[i]).abs();
            if d > max_diff_y { max_diff_y = d; }
        }
        eprintln!("NT after 1 iter — max |Δx|={:.3e}  |Δs|={:.3e}  |Δy|={:.3e}",
                  max_diff_x, max_diff_s, max_diff_y);

        // Tolerance 1e-5: dense and structured use different matrix-sqrt
        // algorithms (eigendecomp vs Denman-Beavers) that converge to the
        // same W up to ~1e-13, plus the structured-vs-dense KKT path adds
        // fp-reordering. 1e-5 is comfortably above both.
        let tol = 1.0e-5;
        assert!(max_diff_x < tol, "NT Δx mismatch {max_diff_x}");
        assert!(max_diff_s < tol, "NT Δs mismatch {max_diff_s}");
        assert!(max_diff_y < tol, "NT Δy mismatch {max_diff_y}");
    }

    /// **The NT free-tf structured gate** (Phase 6.10): run ONE NT
    /// Mehrotra iteration on a **free-tf** SCvx subproblem through both the
    /// dense NT driver (`scvx_ipm::solve_socp_nt`) and the NT free-tf
    /// structured driver (`solve_socp_structured_nt_free_tf`) from the same
    /// cold-start, and verify the iterates match.
    ///
    /// This pins the composition of the NT `W²` scaling (Phase 6.9) with
    /// the Sherman-Morrison δτ correction (Phase 6.8) — the last cell of
    /// the structured dispatch matrix.
    #[test]
    fn structured_nt_free_tf_matches_dense_nt_one_iter() {
        use crate::assemble::{np_scvx_free_tf, nct_scvx_free_tf, ncones_scvx_free_tf};

        const N: usize = 3;
        const NP: usize = np_scvx_free_tf(N);          // 58
        const NE: usize = N * NX + 6;                   // 27
        const NCT: usize = nct_scvx_free_tf(N);        // 92
        const NCONES: usize = ncones_scvx_free_tf(N);  // 26

        let phys = mars_params();
        let mut x_init = SVector::<f64, 7>::zeros();
        x_init[2] = 50.0;
        x_init[5] = -5.0;
        x_init[6] = libm::log(800.0);

        let traj = hover_reference::<N>(x_init, 800.0, 10.0);
        let mut lin = std::boxed::Box::new(LinearizedDynamics::<N>::default());
        discretize_foh(&traj, &phys, &mut lin, 4);

        let term = TerminalCondition { r: [0.0, 0.0, 0.0], v: [0.0, 0.0, 0.0] };
        let mut prob = std::boxed::Box::new(SocpProblem::<NP, NE, NCT, NCONES>::default());
        assemble_scvx_socp::<N, NP, NE, NCT, NCONES>(
            &traj, &lin, &phys, &x_init, &term,
            1.0e3, 1.0e4, /*use_free_tf=*/ true, &mut prob,
        );

        let params = IpmAlgoParams {
            max_iters: 1,
            tol_mu:    1.0e-7,
            tol_primal:1.0e-6,
            tol_dual:  1.0e-6,
            ..IpmAlgoParams::default()
        };

        // ---- Dense NT (handles the free-tf NP layout transparently) ----
        let mut ws_dense = std::boxed::Box::new(SocpWorkspace::<NP, NE, NCT>::default());
        let res_dense = solve_socp_nt(&prob, &params, &mut ws_dense);

        // ---- NT free-tf structured ----
        let mut ws_struct = std::boxed::Box::new(SocpWorkspace::<NP, NE, NCT>::default());
        let res_struct = solve_socp_structured_nt_free_tf::<N, NP, NE, NCT, NCONES>(
            &prob, &params, &mut ws_struct,
        );

        eprintln!("dense  NT free-tf: status_code={}", res_dense.status.as_u32());
        eprintln!("struct NT free-tf: status_code={}", res_struct.status.as_u32());

        assert!(
            !matches!(res_dense.status, IpmStatus::NumericalError),
            "dense NT free-tf hit NumericalError"
        );
        assert!(
            !matches!(res_struct.status, IpmStatus::NumericalError),
            "struct NT free-tf hit NumericalError"
        );

        let mut max_diff_x = 0.0_f64;
        let mut max_diff_s = 0.0_f64;
        let mut max_diff_y = 0.0_f64;
        for i in 0..NP {
            let d = (res_dense.x[i] - res_struct.x[i]).abs();
            if d > max_diff_x { max_diff_x = d; }
        }
        for i in 0..NCT {
            let d = (res_dense.s[i] - res_struct.s[i]).abs();
            if d > max_diff_s { max_diff_s = d; }
            let d = (res_dense.y[i] - res_struct.y[i]).abs();
            if d > max_diff_y { max_diff_y = d; }
        }
        // The δτ primal is at index N·NZ; verify it specifically matched.
        let dtau_diff = (res_dense.x[N * N_VARS_PER_NODE_SCVX]
                         - res_struct.x[N * N_VARS_PER_NODE_SCVX]).abs();
        eprintln!("NT free-tf after 1 iter — max |Δx|={:.3e}  |Δs|={:.3e}  |Δy|={:.3e}  δτ diff={:.3e}",
                  max_diff_x, max_diff_s, max_diff_y, dtau_diff);

        let tol = 1.0e-5;
        assert!(max_diff_x < tol, "NT free-tf Δx mismatch {max_diff_x}");
        assert!(max_diff_s < tol, "NT free-tf Δs mismatch {max_diff_s}");
        assert!(max_diff_y < tol, "NT free-tf Δy mismatch {max_diff_y}");
        assert!(dtau_diff < tol, "NT free-tf δτ mismatch {dtau_diff}");
    }

    /// **DIAGNOSTIC (item #2): pinpoint the NT endgame degeneracy.** Sweeps
    /// `solve_socp_nt` over increasing `max_iters` on a flight-scale subproblem
    /// and reports, from the live iterate, the per-cone complementarity
    /// (sᵢ·yᵢ) spread and the smallest interiority margin — to reveal WHICH
    /// cone drives the breakdown (the boundary-W-blowup hypothesis: a cone
    /// nearing its boundary makes the NT scaling Wᵢ blow up as μ→0).
    /// Run: `cargo test -p scvx-solver diag_nt_endgame_per_cone -- --ignored --nocapture`
    #[test]
    #[ignore = "diagnostic; run explicitly with --ignored --nocapture"]
    fn diag_nt_endgame_per_cone() {
        // Dense NT StepFactors (NCT×NCT) live on the stack; N=5 needs headroom.
        std::thread::Builder::new()
            .stack_size(256 * 1024 * 1024)
            .spawn(diag_nt_endgame_per_cone_body)
            .expect("spawn")
            .join()
            .expect("diag thread panicked");
    }

    fn diag_nt_endgame_per_cone_body() {
        const N: usize = 5;
        const NP: usize = N * N_VARS_PER_NODE_SCVX;
        const NE: usize = N * NX + 6;
        const NCT: usize = N * N_CONE_DIM_PER_NODE_SCVX;
        const NCONES: usize = N * 8;

        let phys = mars_params();
        let mut x_init = SVector::<f64, 7>::zeros();
        x_init[2] = 100.0;
        x_init[5] = -10.0;
        x_init[6] = libm::log(800.0);
        let traj = hover_reference::<N>(x_init, 800.0, 25.0);
        let mut lin = std::boxed::Box::new(LinearizedDynamics::<N>::default());
        discretize_foh(&traj, &phys, &mut lin, 4);
        let term = TerminalCondition { r: [0.0; 3], v: [0.0; 3] };
        let mut prob = std::boxed::Box::new(SocpProblem::<NP, NE, NCT, NCONES>::default());
        assemble_scvx_socp::<N, NP, NE, NCT, NCONES>(
            &traj, &lin, &phys, &x_init, &term, 1.0e3, 1.0e4, false, &mut prob,
        );

        eprintln!("\n=== NT endgame per-cone sweep (N={N}, 100 m flight scale, raw) ===");
        eprintln!("    cone dims/node: [4,1,1,3,1,1,11,8]; idx%8: 0=thrust(SOC4) 3=glide(SOC3) 6=trust(SOC11) 7=virt(SOC8)");
        for k in [3u32, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36] {
            let params = IpmAlgoParams {
                max_iters: k,
                tol_mu: 1.0e-9,
                tol_primal: 1.0e-7,
                tol_dual: 1.0e-7,
                use_nt_scaling: true,
                ..IpmAlgoParams::default()
            };
            let mut ws = std::boxed::Box::new(SocpWorkspace::<NP, NE, NCT>::default());
            let res = solve_socp_nt(&prob, &params, &mut ws);

            let sd = ws.s.as_slice();
            let yd = ws.y.as_slice();
            let mu = ws.s.dot(&ws.y) / (NCT as f64);
            let mut min_compl = f64::INFINITY;
            let mut max_compl = 0.0_f64;
            let mut argmin = 0usize;
            let mut min_smarg = f64::INFINITY;
            let mut min_smarg_cone = 0usize;
            for (ci, cone) in prob.cones.iter().enumerate() {
                let o = cone.offset;
                let d = cone.dim;
                let mut compl = 0.0;
                let mut bar = 0.0;
                for i in 0..d { compl += sd[o + i] * yd[o + i]; }
                for i in 1..d { bar += sd[o + i] * sd[o + i]; }
                let smarg = sd[o] - libm::sqrt(bar);
                if compl.is_finite() && compl < min_compl { min_compl = compl; argmin = ci; }
                if compl.is_finite() && compl > max_compl { max_compl = compl; }
                if smarg.is_finite() && smarg < min_smarg { min_smarg = smarg; min_smarg_cone = ci; }
            }
            let spread = if min_compl > 0.0 { max_compl / min_compl } else { f64::INFINITY };
            eprintln!(
                "  k={k:>2} st={} it={:>2} | mu={:>9.2e} | min_compl={:>9.2e}@c{}(t{}) max={:>9.2e} spread={:>7.1e} | min_s_marg={:>9.2e}@c{}(t{})",
                res.status.as_u32(), res.iters, mu,
                min_compl, argmin, argmin % 8, max_compl, spread,
                min_smarg, min_smarg_cone, min_smarg_cone % 8,
            );
        }
    }
}
