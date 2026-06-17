//! Generic Mehrotra predictor-corrector primal-dual IPM for SOCPs in
//! standard form
//! ```text
//!   min   cᵀ x
//!   s.t.  A x = b
//!         G x + s = h,   s ∈ K = SOC^{d_1} × … × SOC^{d_m}
//! ```
//!
//! Two directions, both LANDED here: AHO ([`solve_socp`], raw `arrow(s)` /
//! `arrow(y)` blocks) and NT ([`solve_socp_nt`], matrix-form Nesterov-Todd
//! scaling `M = W²` via [`build_step_factors_nt`], with the complementarity
//! residual `r_c` rederived in scaled coordinates and a weighted
//! Colombo-Gondzio corrector). The NT matrix-sqrt primitives live in
//! `crate::cone`. (NT converges on well-conditioned problems but stalls in
//! the endgame on the flight-scale SCvx subproblem — see HANDOFF "NT
//! endgame"; AHO is the production default.)
//!
//! Dense reduced-KKT solve via Schur complement on `(H, Aᵀ; A, 0)`. The
//! LTV-structured (block-tridiagonal) substitution for `H⁻¹` lives in
//! `scvx_solver::reduced_kkt` and is wired through `scvx_solver::structured_socp`.
//!
//! Const-generic over `(NP, NE, NCT, NCONES)`:
//! - `NP`     = primal dim
//! - `NE`     = equality constraints (≥ 1 supported)
//! - `NCT`    = total cone dim  (Σ d_i)
//! - `NCONES` = number of cones in the product
//!
//! Every buffer lives on the stack; no `alloc`, no panic on any solver path.

use libm::sqrt;
use nalgebra::{SMatrix, SVector};
use scvx_core::{IpmAlgoParams, IpmStatus};

use crate::cone::{
    soc_arrow_matrix, soc_in_interior, soc_jordan_product, soc_max_step,
    soc_nt_scaling_exact, soc_nt_w_and_inverse, soc_w_squared,
};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// One cone in the product `K`. Slice `[offset, offset+dim)` of `s` / `y`.
#[derive(Clone, Copy, Default)]
pub struct ConeDesc {
    pub offset: usize,
    pub dim:    usize,
}

pub struct SocpProblem<
    const NP: usize,
    const NE: usize,
    const NCT: usize,
    const NCONES: usize,
> {
    pub c:     SVector<f64, NP>,
    pub a_mat: SMatrix<f64, NE, NP>,
    pub b:     SVector<f64, NE>,
    pub g_mat: SMatrix<f64, NCT, NP>,
    pub h:     SVector<f64, NCT>,
    pub cones: [ConeDesc; NCONES],
}

impl<const NP: usize, const NE: usize, const NCT: usize, const NCONES: usize> Default
    for SocpProblem<NP, NE, NCT, NCONES>
{
    fn default() -> Self {
        Self {
            c:     SVector::zeros(),
            a_mat: SMatrix::zeros(),
            b:     SVector::zeros(),
            g_mat: SMatrix::zeros(),
            h:     SVector::zeros(),
            cones: [ConeDesc::default(); NCONES],
        }
    }
}

pub struct SocpWorkspace<const NP: usize, const NE: usize, const NCT: usize> {
    pub x:      SVector<f64, NP>,
    pub lambda: SVector<f64, NE>,
    pub s:      SVector<f64, NCT>,
    pub y:      SVector<f64, NCT>,
    /// Best-feasible iterate snapshot (the plan's BestFeasible fallback).
    best_x:      SVector<f64, NP>,
    best_lambda: SVector<f64, NE>,
    best_s:      SVector<f64, NCT>,
    best_y:      SVector<f64, NCT>,
    best_mu:     f64,
    best_valid:  bool,
}

impl<const NP: usize, const NE: usize, const NCT: usize> Default
    for SocpWorkspace<NP, NE, NCT>
{
    fn default() -> Self {
        Self {
            x:           SVector::zeros(),
            lambda:      SVector::zeros(),
            s:           SVector::zeros(),
            y:           SVector::zeros(),
            best_x:      SVector::zeros(),
            best_lambda: SVector::zeros(),
            best_s:      SVector::zeros(),
            best_y:      SVector::zeros(),
            best_mu:     f64::INFINITY,
            best_valid:  false,
        }
    }
}

pub struct SocpResult<const NP: usize, const NE: usize, const NCT: usize> {
    pub x:      SVector<f64, NP>,
    pub lambda: SVector<f64, NE>,
    pub s:      SVector<f64, NCT>,
    pub y:      SVector<f64, NCT>,
    pub status: IpmStatus,
    pub iters:  u32,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Initialize `v` per cone using `h` (or `h − G·x_ref`) as a warm-start hint.
///
/// **The boundary problem:** SCvx warm-start can put cone variables exactly
/// on the cone boundary (e.g., glide-slope at terminal `r_z = 0`, or thrust
/// magnitude when `‖u_ref‖ = σ_ref` exactly). AHO direction requires
/// strictly interior `s` and `y` — boundary values cause `arrow(s)` to
/// become singular and the IPM fails on iter 0.
///
/// **Three regimes:**
/// 1. `h` comfortably interior (`h_0 > ‖h_bar‖ + safety_margin`): use as-is.
///    This is the standard primal-feasible warm-start.
/// 2. `h` interior but near boundary: blend with identity to push into
///    strict interior. Loses some primal feasibility, gains conditioning.
/// 3. `h` on or outside the cone: identity fallback (no primal feasibility
///    at iter 0, but the IPM converges by closing the gap).
///
/// Crucially, **never** blend when `h[0]` is negative — that pulls the
/// identity-direction in the wrong sign and gives a non-interior point.
fn init_per_cone_warm_start<const NCT: usize, const NCONES: usize>(
    h:     &SVector<f64, NCT>,
    cones: &[ConeDesc; NCONES],
    v:     &mut SVector<f64, NCT>,
) {
    /// Safety margin (relative to cone magnitude) below which we either
    /// blend or fall back. Above this margin, use `h` directly.
    const SAFETY_MARGIN_RATIO: f64 = 0.05;
    /// Fraction of identity to mix in when `h` is interior-but-tight.
    const BLEND_FRACTION: f64 = 0.10;

    *v = SVector::zeros();
    let hd = h.as_slice();
    let vd = v.as_mut_slice();
    for cone in cones {
        let off = cone.offset;
        let d   = cone.dim;
        let h_slice = &hd[off..off + d];

        // Explicit finiteness gate on the WHOLE seed slice. A poisoned
        // warm-start `x` (e.g. NaN/Inf carried over from a prior diverged
        // solve) must route to the canonical interior point, not propagate.
        // The margin logic below already does this incidentally (a non-finite
        // tail poisons `margin`, and `margin > thr` is false ⇒ regime 3), but
        // checking up front makes iter-0 robustness explicit instead of
        // resting on f64::max/NaN-comparison subtleties. No-op for finite
        // seeds (every nominal path), so behavior is unchanged there.
        if !h_slice.iter().all(|x| x.is_finite()) {
            vd[off] = 1.0;
            for i in 1..d { vd[off + i] = 0.0; }
            continue;
        }

        let mut bar_mag_sq = 0.0;
        for &hb in &h_slice[1..d] {
            bar_mag_sq += hb * hb;
        }
        let bar_norm = sqrt(bar_mag_sq);
        let scale    = h_slice[0].abs().max(bar_norm).max(1.0);

        if !h_slice[0].is_finite() || !scale.is_finite() {
            vd[off] = 1.0;
            for i in 1..d { vd[off + i] = 0.0; }
            continue;
        }

        // Interior margin in physical units.
        let margin     = h_slice[0] - bar_norm;
        let safety_thr = SAFETY_MARGIN_RATIO * scale;

        if margin > safety_thr {
            // Regime 1: comfortably interior — use h directly (primal-feasible).
            vd[off..off + d].copy_from_slice(h_slice);
        } else if margin > 0.0 && h_slice[0] > 0.0 {
            // Regime 2: interior but tight. Blend with identity. Only safe
            // when h[0] > 0 (else the blend tugs the wrong way).
            let one_minus = 1.0 - BLEND_FRACTION;
            vd[off] = one_minus * h_slice[0] + BLEND_FRACTION * scale;
            for i in 1..d {
                vd[off + i] = one_minus * h_slice[i];
            }
        } else {
            // Regime 3: on/outside boundary, or h[0] ≤ 0. Identity fallback.
            vd[off] = scale;
            for i in 1..d { vd[off + i] = 0.0; }
        }
    }
}

/// Stacked Jordan-algebra identity `e = (1,0,…) ⊕ (1,0,…) ⊕ …`.
fn per_cone_e<const NCT: usize, const NCONES: usize>(
    cones: &[ConeDesc; NCONES],
) -> SVector<f64, NCT> {
    let mut e = SVector::zeros();
    for cone in cones {
        e[cone.offset] = 1.0;
    }
    e
}

/// Element-wise Jordan product per cone:
/// `out[cone] = u[cone] ∘ v[cone]` for every cone.
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

/// Block-diagonal arrow matrix: `arrow(z) = blkdiag( arrow(z_c) for c in cones )`.
fn build_block_arrow<const NCT: usize, const NCONES: usize>(
    z:     &SVector<f64, NCT>,
    cones: &[ConeDesc; NCONES],
) -> SMatrix<f64, NCT, NCT> {
    let mut m = SMatrix::<f64, NCT, NCT>::zeros();
    for cone in cones {
        let off = cone.offset;
        let d   = cone.dim;
        let z0  = z[off];
        m[(off, off)] = z0;
        for i in 1..d {
            m[(off,     off + i)] = z[off + i];
            m[(off + i, off    )] = z[off + i];
            m[(off + i, off + i)] = z0;
        }
    }
    m
}

/// Per-D matrix-sqrt of `W²` via symmetric eigendecomposition. This is the
/// **robust** path — uses `nalgebra::SymmetricEigen` (Jacobi rotations on
/// small matrices), which handles ill-conditioning that breaks generic-`D`
/// Denman-Beavers. Falls back to `soc_nt_w_and_inverse` (plain DB) if
/// eigendecomposition fails to converge.
///
/// The `SymmetricEigen` constraint `Const<D>: DimSub<U1>` is implemented
/// for fixed `D` in nalgebra but not generically — hence the macro that
/// emits one specialized function per `D` we actually use.
macro_rules! emit_nt_w_specialized {
    ($name:ident, $D:literal) => {
        fn $name(
            s: &SVector<f64, $D>,
            y: &SVector<f64, $D>,
        ) -> Option<(SMatrix<f64, $D, $D>, SMatrix<f64, $D, $D>)> {
            let w_sq = soc_w_squared::<$D>(s, y)?;
            // Try eigendecomp first (robust for ill-conditioned PD).
            if let Some(eig) =
                nalgebra::SymmetricEigen::try_new(w_sq, 1.0e-14, 200)
            {
                let mut sqrt_vals     = SVector::<f64, $D>::zeros();
                let mut inv_sqrt_vals = SVector::<f64, $D>::zeros();
                let mut all_pos = true;
                for i in 0..$D {
                    let lam = eig.eigenvalues[i];
                    if lam <= 0.0 || !lam.is_finite() {
                        all_pos = false;
                        break;
                    }
                    let r = libm::sqrt(lam);
                    sqrt_vals[i]     = r;
                    inv_sqrt_vals[i] = 1.0 / r;
                }
                if all_pos {
                    let u  = eig.eigenvectors;
                    let ut = u.transpose();
                    let w     = u * SMatrix::from_diagonal(&sqrt_vals)     * ut;
                    let w_inv = u * SMatrix::from_diagonal(&inv_sqrt_vals) * ut;
                    return Some((w, w_inv));
                }
            }
            // Fallback 1: Higham-scaled Denman-Beavers (per-D
            // specialization — uses `SMatrix::determinant`, which
            // requires `Const<D>: ToTypenum` and so can't go in the
            // generic `cone.rs` impl).
            //
            // Standard Higham scaled DB (Algorithm 6.15 in Higham,
            // "Functions of Matrices", 2008):
            //
            //   μ_k = (|det Z_k| / |det Y_k|)^(1/(2·D))
            //   Y_{k+1} = ½(μ_k · Y_k + μ_k⁻¹ · Z_k⁻¹)
            //   Z_{k+1} = ½(μ_k · Z_k + μ_k⁻¹ · Y_k⁻¹)
            //
            // The scaling keeps `det(μ·Y) · det(Z/μ) = det(Y)·det(Z)`
            // (invariant under one step) and drives both toward 1
            // simultaneously. Convergence is quadratic and dramatically
            // more robust than plain DB for ill-conditioned `W²`.
            {
                let mut y_mat = w_sq;
                let mut z_mat = SMatrix::<f64, $D, $D>::identity();
                const MAX_ITERS: usize = 40;
                const TOL: f64 = 1.0e-13;

                let mut converged = true;
                for _ in 0..MAX_ITERS {
                    let det_y = y_mat.determinant();
                    let det_z = z_mat.determinant();
                    let mu = if det_y.abs() > 1.0e-300 && det_z.abs() > 1.0e-300 {
                        let ratio = det_z.abs() / det_y.abs();
                        let m = libm::pow(ratio, 1.0 / (2.0 * ($D as f64)));
                        // Clamp to [0.01, 100] for numerical safety
                        // (extreme μ can introduce spurious large entries).
                        if m.is_finite() && m > 0.01 && m < 100.0 { m } else { 1.0 }
                    } else {
                        1.0
                    };
                    let inv_mu = 1.0 / mu;

                    let y_inv = match y_mat.try_inverse() {
                        Some(m) => m,
                        None    => { converged = false; break; }
                    };
                    let z_inv = match z_mat.try_inverse() {
                        Some(m) => m,
                        None    => { converged = false; break; }
                    };
                    let y_new = (y_mat * mu + z_inv * inv_mu) * 0.5;
                    let z_new = (z_mat * mu + y_inv * inv_mu) * 0.5;

                    let mut max_step = 0.0_f64;
                    for i in 0..$D {
                        for j in 0..$D {
                            let d = (y_new[(i, j)] - y_mat[(i, j)]).abs();
                            if d > max_step { max_step = d; }
                        }
                    }
                    y_mat = y_new;
                    z_mat = z_new;
                    if max_step < TOL { break; }
                }

                // Validate Y·Z ≈ I.
                if converged {
                    let prod = y_mat * z_mat;
                    let mut ok = true;
                    for i in 0..$D {
                        for j in 0..$D {
                            let want = if i == j { 1.0 } else { 0.0 };
                            if (prod[(i, j)] - want).abs() > 1.0e-9 {
                                ok = false;
                                break;
                            }
                        }
                        if !ok { break; }
                    }
                    if ok {
                        return Some((y_mat, z_mat));
                    }
                }
            }
            // Fallback 2: plain Denman-Beavers (less robust, last-resort
            // — Higham must have diverged or sanity-checked false).
            soc_nt_w_and_inverse::<$D>(s, y)
        }
    };
}
emit_nt_w_specialized!(nt_w_d1,  1);
emit_nt_w_specialized!(nt_w_d3,  3);
emit_nt_w_specialized!(nt_w_d4,  4);
emit_nt_w_specialized!(nt_w_d8,  8);
emit_nt_w_specialized!(nt_w_d11, 11);

/// Per-cone Nesterov-Todd block builder: write `W`, `W⁻¹`, and the scaled
/// iterate `s̃ = W·s` into the block-diagonal `w_out`, `w_inv_out`, and
/// `s_scaled_out` at offset `[cone.offset, cone.offset + cone.dim)`.
///
/// Dispatches per-D-specialized matrix-sqrt helpers (eigendecomp-first,
/// Denman-Beavers fallback) over the cone dimensions our SCvx subproblem
/// actually uses: `{1, 3, 4, 8, 11}`. Other dims would need explicit cases.
///
/// Returns `false` if the NT computation failed for any reason (non-interior
/// iterate, non-PD `arrow(y)`, eigendecomposition + Denman-Beavers both
/// failing) — caller falls back to AHO direction or aborts the iteration.
fn build_nt_block_for_cone<const NCT: usize>(
    s:            &[f64],
    y:            &[f64],
    cone:         &ConeDesc,
    w_out:        &mut SMatrix<f64, NCT, NCT>,
    w_inv_out:    &mut SMatrix<f64, NCT, NCT>,
    s_scaled_out: &mut SVector<f64, NCT>,
) -> bool {
    let off = cone.offset;
    let d   = cone.dim;
    let s_c = &s[off..off + d];
    let y_c = &y[off..off + d];

    /// Pull a slice into an SVector<D>, run the specialized NT-W computer,
    /// write blocks into the global buffers.
    macro_rules! handle {
        ($D:literal, $compute:ident) => {{
            let mut s_vec = SVector::<f64, $D>::zeros();
            let mut y_vec = SVector::<f64, $D>::zeros();
            for i in 0..$D {
                s_vec[i] = s_c[i];
                y_vec[i] = y_c[i];
            }
            // Exact closed-form NT scaling (normalized-point boost form) is the
            // primary path — it stays numerically bounded as the cone vanishes
            // (the SCvx virtual-control relaxation drives ν→0, so its SOC^8
            // cones ride onto their boundary at the optimum, where the
            // geometric-mean `arrow(s)^{−1/2}` overflows). Fall back to the
            // eigendecomp / Denman-Beavers matrix-sqrt of the geometric-mean
            // form only if the exact path declines (non-interior iterate).
            let (w_c, w_inv_c) = match soc_nt_scaling_exact::<$D>(&s_vec, &y_vec)
                .or_else(|| $compute(&s_vec, &y_vec))
            {
                Some(p) => p,
                None    => return false,
            };
            // s̃ = W·s (per cone)
            let s_scaled_c = w_c * s_vec;
            for i in 0..$D {
                s_scaled_out[off + i] = s_scaled_c[i];
                for j in 0..$D {
                    w_out    [(off + i, off + j)] = w_c    [(i, j)];
                    w_inv_out[(off + i, off + j)] = w_inv_c[(i, j)];
                }
            }
            true
        }};
    }
    match d {
        1  => handle!(1,  nt_w_d1),
        3  => handle!(3,  nt_w_d3),
        4  => handle!(4,  nt_w_d4),
        8  => handle!(8,  nt_w_d8),
        11 => handle!(11, nt_w_d11),
        _ => false,
    }
}

/// Per-cone `arrow(s̃)⁻¹` block builder. `s̃` is the scaled iterate from
/// `build_nt_block_for_cone`. Only the inverse is written into the global
/// block-diagonal matrix — the forward `arrow(s̃)` is constructed locally
/// in a `D×D` SMatrix and dropped, avoiding an `NCT×NCT` allocation.
///
/// Returns `false` if any per-cone arrow inversion fails (iterate on the
/// cone boundary).
fn build_arrow_scaled_inv_block<const NCT: usize>(
    s_scaled:      &SVector<f64, NCT>,
    cone:          &ConeDesc,
    arrow_inv_out: &mut SMatrix<f64, NCT, NCT>,
) -> bool {
    let off = cone.offset;
    let d   = cone.dim;
    macro_rules! handle {
        ($D:literal) => {{
            let mut sv = SVector::<f64, $D>::zeros();
            for i in 0..$D {
                sv[i] = s_scaled[off + i];
            }
            let inv = match soc_arrow_matrix(&sv).try_inverse() {
                Some(m) => m,
                None    => return false,
            };
            for i in 0..$D {
                for j in 0..$D {
                    arrow_inv_out[(off + i, off + j)] = inv[(i, j)];
                }
            }
            true
        }};
    }
    match d {
        1  => handle!(1),
        3  => handle!(3),
        4  => handle!(4),
        8  => handle!(8),
        11 => handle!(11),
        _ => false,
    }
}

/// Max scalar step along `(s + α·ds, y + α·dy)` keeping both in the closed
/// product cone. Per-cone bottleneck via [`soc_max_step`].
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

/// Clip step length `α` to `[0, 1]` with explicit NaN / ±∞ semantics
/// (same defense as `mehrotra::clip01` — see audit note). `f64::clamp`
/// would propagate NaN as a full step; we want NaN → 0 (reject step).
#[inline]
#[allow(clippy::manual_clamp)]
fn clip01(a: f64) -> f64 {
    if a.is_nan() {
        return 0.0;
    }
    if a > 1.0 {
        1.0
    } else if a < 0.0 {
        0.0
    } else {
        a
    }
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
// Newton-step solver — one Schur-complement reduction shared by predictor
// and corrector, since (H, A) don't change between affine and corrector.
// ---------------------------------------------------------------------------

/// Cached factors that depend only on `(s, y, G, A)` — built once per IPM
/// iteration, used for both affine and corrector solves.
struct StepFactors<const NP: usize, const NE: usize, const NCT: usize> {
    arrow_s_inv: SMatrix<f64, NCT, NCT>,
    arrow_y:     SMatrix<f64, NCT, NCT>,
    /// `M = arrow(s)⁻¹ · arrow(y)`  (block-diagonal, NOT symmetric in AHO)
    m_scale:     SMatrix<f64, NCT, NCT>,
    /// `H⁻¹` where `H = Gᵀ M G + reg·I`  (NP × NP; H is asymmetric for AHO).
    /// Field stores the *inverse* (used as `h_inv * b`), matching the NT
    /// `NtStepFactors::h_inv` convention.
    h_inv:       SMatrix<f64, NP, NP>,
    /// `S = A H⁻¹ Aᵀ`  (NE × NE)
    s_inv:       SMatrix<f64, NE, NE>,
}

/// Floor for the Tikhonov regularization on the reduced Hessian.
/// `H = GᵀMG` (or `Gᵀ·W²·G` for NT) is rank-deficient when some primal
/// variables are not touched by any cone constraint (e.g., velocity in
/// an LCvx-style problem without a trust-region cone). The equality
/// block `A` pins those directions down, but the Schur-via-`H`
/// factorization needs `H` itself invertible. `ε·I` makes `H` PD
/// without measurably biasing the solution.
const H_REG_FLOOR: f64 = 1.0e-8;

/// Relative scale for adaptive regularization: `reg = max(H_REG_FLOOR,
/// H_REG_RELATIVE · tr(H)/n)`. On small / unscaled problems the floor
/// dominates and behavior is identical to a fixed `1e-8`. On problems
/// with column-scaled SOCPs (per-variable preconditioning), `tr(H)/n`
/// can grow by orders of magnitude and the relative term takes over —
/// keeping the Tikhonov term meaningful relative to the scaled-Hessian
/// magnitudes. Without this, NT + preconditioning silently degenerates
/// (the `1e-8` floor is negligible against entries ~10^7 in `H'`).
const H_REG_RELATIVE: f64 = 1.0e-10;

/// Compute the regularization term to add to the reduced Hessian diagonal.
///
/// When `adaptive = false`, returns the fixed floor `H_REG_FLOOR` — same
/// as the legacy hardcoded value. Existing tests / oracle diffs depend on
/// this behavior, so it's the default.
///
/// When `adaptive = true`, returns `max(H_REG_FLOOR, rel_factor · tr(H)/n)`,
/// where `rel_factor` is the caller's `IpmAlgoParams::regularization` (default
/// `H_REG_RELATIVE = 1e-10`; a non-finite/negative value falls back to that
/// default). Use this on column-preconditioned problems where `H'` has
/// huge diagonal entries (the fixed floor becomes negligible). **Do NOT**
/// use this on unscaled problems whose IPM iterates approach the cone
/// boundary — `tr(H)` can grow without bound there, and the adaptive
/// term over-regularizes (effectively replacing the Newton step with a
/// gradient step).
#[inline]
fn regularization<const NP: usize>(
    h_mat:      &SMatrix<f64, NP, NP>,
    adaptive:   bool,
    rel_factor: f64,
) -> f64 {
    if !adaptive {
        return H_REG_FLOOR;
    }
    // `rel_factor` is the caller's `IpmAlgoParams::regularization`; fall back to
    // the documented `H_REG_RELATIVE` default if it is non-finite or negative.
    let rel = if rel_factor.is_finite() && rel_factor >= 0.0 {
        rel_factor
    } else {
        H_REG_RELATIVE
    };
    let mut trace = 0.0;
    for i in 0..NP {
        trace += h_mat[(i, i)];
    }
    let avg = trace / NP as f64;
    if avg.is_finite() && avg > 0.0 {
        (avg * rel).max(H_REG_FLOOR)
    } else {
        H_REG_FLOOR
    }
}

/// Build the Schur factors. Returns `None` if either `H` or `S` is singular —
/// caller flips to numerical-exit path.
///
/// **Currently uses AHO direction** (`M = arrow(s)^{−1}·arrow(y)`,
/// asymmetric). NT-direction integration would require rederiving the
/// complete Newton step in scaled coordinates (not just swapping the
/// scaling matrix — the complementarity residual `r_c` must also be
/// reformulated to be consistent with the NT scaling). The NT
/// primitives are implemented in [`crate::cone::soc_nt_scaling_matrix`]
/// for when that lift lands.
fn build_step_factors<
    const NP: usize,
    const NE: usize,
    const NCT: usize,
    const NCONES: usize,
>(
    prob:     &SocpProblem<NP, NE, NCT, NCONES>,
    s:        &SVector<f64, NCT>,
    y:        &SVector<f64, NCT>,
    adaptive_reg: bool,
    reg_rel:      f64,
) -> Option<StepFactors<NP, NE, NCT>> {
    let arrow_s = build_block_arrow(s, &prob.cones);
    let arrow_y = build_block_arrow(y, &prob.cones);
    let arrow_s_inv = arrow_s.try_inverse()?;
    let m_scale = arrow_s_inv * arrow_y;

    let mut h_mat = prob.g_mat.transpose() * m_scale * prob.g_mat;
    let reg = regularization::<NP>(&h_mat, adaptive_reg, reg_rel);
    for i in 0..NP {
        h_mat[(i, i)] += reg;
    }
    let h_inv = h_mat.try_inverse()?;
    let schur = prob.a_mat * h_inv * prob.a_mat.transpose();
    let s_inv = schur.try_inverse()?;
    Some(StepFactors {
        arrow_s_inv,
        arrow_y,
        m_scale,
        h_inv,
        s_inv,
    })
}

/// Given the cached factors and the four residuals `(r_x, r_a, r_g, r_c)`,
/// solve the reduced KKT system for `(Δx, Δλ, Δs, Δy)`.
///
/// Math (derived in source comments above):
/// ```text
///   Δs = -r_g - G Δx
///   Δy = arrow(s)⁻¹ ( arrow(y) (r_g + G Δx) - r_c )
///   [H  Aᵀ] [Δx]   [-r_x - Gᵀ arrow(s)⁻¹ arrow(y) r_g + Gᵀ arrow(s)⁻¹ r_c]
///   [A  0 ] [Δλ] = [-r_a                                                  ]
/// ```
/// Δx and Δλ via the Schur complement of H.
fn solve_newton_step<
    const NP: usize,
    const NE: usize,
    const NCT: usize,
    const NCONES: usize,
>(
    prob: &SocpProblem<NP, NE, NCT, NCONES>,
    fac:  &StepFactors<NP, NE, NCT>,
    r_x:  &SVector<f64, NP>,
    r_a:  &SVector<f64, NE>,
    r_g:  &SVector<f64, NCT>,
    r_c:  &SVector<f64, NCT>,
) -> (SVector<f64, NP>, SVector<f64, NE>, SVector<f64, NCT>, SVector<f64, NCT>) {
    // b_x = -r_x - Gᵀ M r_g + Gᵀ arrow(s)⁻¹ r_c
    let b_x = -r_x
            - prob.g_mat.transpose() * (fac.m_scale * r_g)
            + prob.g_mat.transpose() * (fac.arrow_s_inv * r_c);
    let b_a = -r_a;

    // Schur: Δλ = S⁻¹ ( A H⁻¹ b_x - b_a )
    let dl = fac.s_inv * (prob.a_mat * (fac.h_inv * b_x) - b_a);
    // Δx = H⁻¹ ( b_x - Aᵀ Δλ )
    let dx = fac.h_inv * (b_x - prob.a_mat.transpose() * dl);

    // Δs = -r_g - G Δx
    let ds = -r_g - prob.g_mat * dx;
    // Δy = arrow(s)⁻¹ ( arrow(y) (r_g + G Δx) - r_c )
    let dy = fac.arrow_s_inv * (fac.arrow_y * (r_g + prob.g_mat * dx) - r_c);

    (dx, dl, ds, dy)
}

// ---------------------------------------------------------------------------
// NT-direction Newton step (matrix-form Nesterov-Todd scaling).
// ---------------------------------------------------------------------------

/// NT-direction step factors — replaces AHO's `(arrow_s_inv, arrow_y,
/// m_scale)` with the symmetric matrix-form NT scaling.
///
/// The reduced Hessian becomes `H_NT = Gᵀ·W²·G` (symmetric PD), which
/// eliminates the AHO endgame degeneracy. Recovery of `Δs`/`Δy` uses `W`,
/// `W⁻¹`, and the per-cone `arrow(s̃)⁻¹`.
struct NtStepFactors<const NP: usize, const NE: usize, const NCT: usize> {
    /// Block-diagonal `W` (= NT scaling matrix, symmetric PD).
    w:                  SMatrix<f64, NCT, NCT>,
    /// Block-diagonal `W²` (the operator geometric mean of
    /// `arrow(s)⁻¹` and `arrow(y)`). Symmetric PD.
    w_squared:          SMatrix<f64, NCT, NCT>,
    /// Block-diagonal `arrow(s̃)⁻¹` where `s̃ = W·s` (= `ỹ` by NT property).
    arrow_s_scaled_inv: SMatrix<f64, NCT, NCT>,
    /// `s̃ = W·s` (= `ỹ` at iterate, by NT property).
    s_scaled:           SVector<f64, NCT>,
    /// `H⁻¹ = (Gᵀ·W²·G + reg·I)⁻¹`.
    h_inv:              SMatrix<f64, NP, NP>,
    /// Schur: `S⁻¹ = (A·H⁻¹·Aᵀ)⁻¹`.
    s_inv:              SMatrix<f64, NE, NE>,
}

/// Build all NT step factors at iterate `(s, y)`. Returns `None` if any
/// per-cone NT computation fails (non-interior iterate, Denman-Beavers
/// divergence) or any required inverse is singular.
fn build_step_factors_nt<
    const NP:     usize,
    const NE:     usize,
    const NCT:    usize,
    const NCONES: usize,
>(
    prob:     &SocpProblem<NP, NE, NCT, NCONES>,
    s:        &SVector<f64, NCT>,
    y:        &SVector<f64, NCT>,
    adaptive_reg: bool,
    reg_rel:      f64,
) -> Option<NtStepFactors<NP, NE, NCT>> {
    // Per-cone NT computation (W, W⁻¹, s̃).
    let mut w        = SMatrix::<f64, NCT, NCT>::zeros();
    let mut w_inv    = SMatrix::<f64, NCT, NCT>::zeros();
    let mut s_scaled = SVector::<f64, NCT>::zeros();
    let s_data = s.as_slice();
    let y_data = y.as_slice();
    for cone in &prob.cones {
        if !build_nt_block_for_cone::<NCT>(
            s_data, y_data, cone, &mut w, &mut w_inv, &mut s_scaled,
        ) {
            return None;
        }
    }
    let w_squared = w * w;

    // Per-cone arrow(s̃)⁻¹ — only the inverse is materialized into the
    // global block-diag matrix.
    let mut arrow_s_scaled_inv = SMatrix::<f64, NCT, NCT>::zeros();
    for cone in &prob.cones {
        if !build_arrow_scaled_inv_block::<NCT>(
            &s_scaled, cone, &mut arrow_s_scaled_inv,
        ) {
            return None;
        }
    }

    // H = Gᵀ·W²·G + reg·I. Symmetric PD by construction. When
    // `adaptive_reg = true` (typically because the SCvx outer loop
    // applied per-variable preconditioning), `reg` scales with `tr(H)/n`
    // so the Tikhonov term remains meaningful against `H'` entries ~10^7.
    let mut h_mat = prob.g_mat.transpose() * w_squared * prob.g_mat;
    let reg = regularization::<NP>(&h_mat, adaptive_reg, reg_rel);
    for i in 0..NP {
        h_mat[(i, i)] += reg;
    }
    let h_inv = h_mat.try_inverse()?;
    let schur = prob.a_mat * h_inv * prob.a_mat.transpose();
    let s_inv = schur.try_inverse()?;

    Some(NtStepFactors {
        w,
        w_squared,
        arrow_s_scaled_inv,
        s_scaled,
        h_inv,
        s_inv,
    })
}

/// NT-direction Newton-step solver. Same calling pattern as AHO's
/// `solve_newton_step`, but with NT-scaled formulas throughout.
///
/// Math (derived from the scaled-coords Newton system and converted back):
/// ```text
///   Δs = −r_g − G·Δx                                 (same as AHO)
///   Δỹ = −arrow(s̃)⁻¹·F̃_4 + W·(r_g + G·Δx)
///   Δy = W·Δỹ
///        = −W·arrow(s̃)⁻¹·F̃_4 + W²·(r_g + G·Δx)
///   b_x = −r_x − Gᵀ·W²·r_g + Gᵀ·W·arrow(s̃)⁻¹·F̃_4_arg
/// ```
///
/// `r_c_arg` is the scaled-coords `arrow(s̃)⁻¹·F̃_4` term (not the raw F̃_4),
/// because every NT use of F̃_4 multiplies by `arrow(s̃)⁻¹` first. Caller
/// computes this once and reuses for both Δx/Δλ solve and Δy recovery.
fn solve_newton_step_nt<
    const NP:     usize,
    const NE:     usize,
    const NCT:    usize,
    const NCONES: usize,
>(
    prob:    &SocpProblem<NP, NE, NCT, NCONES>,
    fac:     &NtStepFactors<NP, NE, NCT>,
    r_x:     &SVector<f64, NP>,
    r_a:     &SVector<f64, NE>,
    r_g:     &SVector<f64, NCT>,
    // `r_c_arg` = `arrow(s̃)⁻¹·F̃_4` already in scaled coords.
    r_c_arg: &SVector<f64, NCT>,
) -> (SVector<f64, NP>, SVector<f64, NE>, SVector<f64, NCT>, SVector<f64, NCT>) {
    // b_x = -r_x - Gᵀ·W²·r_g + Gᵀ·W·r_c_arg
    let b_x = -r_x
            - prob.g_mat.transpose() * (fac.w_squared * r_g)
            + prob.g_mat.transpose() * (fac.w        * r_c_arg);
    let b_a = -r_a;

    // Schur on H: Δλ = S⁻¹ ( A·H⁻¹·b_x − b_a )
    let dl = fac.s_inv * (prob.a_mat * (fac.h_inv * b_x) - b_a);
    // Δx = H⁻¹ ( b_x − Aᵀ·Δλ )
    let dx = fac.h_inv * (b_x - prob.a_mat.transpose() * dl);

    // Δs = -r_g - G·Δx     (same as AHO)
    let ds = -r_g - prob.g_mat * dx;
    // Δy = -W·r_c_arg + W²·(r_g + G·Δx)
    let dy = -fac.w * r_c_arg + fac.w_squared * (r_g + prob.g_mat * dx);

    // NOTE: iterative refinement of this reduced-KKT solve was implemented and
    // measured (see HANDOFF "NT endgame" notes). It did NOT help and slightly
    // hurt: a more-accurate NT direction broke down *earlier* on the flight
    // subproblem (inner iter 35→30), because the breakdown is degeneracy of the
    // Newton *linearization* as μ→0 — not linear-solve accuracy. By the same
    // logic, heavier accurate-solve remedies (Krylov/PSQMR) would not help
    // either. Reverted to the single Schur solve.
    (dx, dl, ds, dy)
}

// ---------------------------------------------------------------------------
// Solver
// ---------------------------------------------------------------------------

const BACKOFF: f64 = 0.99;

/// Hard upper bound on inner-IPM iterations, independent of the caller-supplied
/// [`IpmAlgoParams::max_iters`]. Every IPM loop runs `min(max_iters,
/// IPM_HARD_MAX_ITERS)` times, so the per-solve worst-case iteration count — and
/// therefore the WCET — is a compile-time constant regardless of what a Rust or
/// FFI caller passes (a caller cannot blow the flight time budget by requesting
/// a huge `max_iters`). Sized comfortably above every shipping configuration
/// (default 25; the SCvx outer loop and tests pass ≤ 50). Raise it — and
/// re-measure the WCET budget — only if a mission genuinely needs deeper inner
/// solves. The same cap is applied by the structured drivers in
/// `scvx_solver::structured_socp`.
pub const IPM_HARD_MAX_ITERS: u32 = 64;

/// Validate that the (`pub`, caller-constructed) cone descriptors fit within
/// `NCT`: every cone needs `dim ≥ 1` and `offset + dim ≤ NCT`, with no
/// `usize` overflow. A malformed descriptor (e.g. from a hand-built or
/// FFI-supplied `SocpProblem`) would otherwise produce an out-of-bounds
/// slice/index — a panic, violating the "no panic on any solver path"
/// guarantee. `NCONES` is tiny, so this one-time entry check is free.
fn cones_valid<const NCT: usize, const NCONES: usize>(
    cones: &[ConeDesc; NCONES],
) -> bool {
    for cone in cones {
        if cone.dim == 0 {
            return false;
        }
        match cone.offset.checked_add(cone.dim) {
            Some(end) if end <= NCT => {}
            _ => return false,
        }
    }
    true
}

pub fn solve_socp<
    const NP: usize,
    const NE: usize,
    const NCT: usize,
    const NCONES: usize,
>(
    prob:   &SocpProblem<NP, NE, NCT, NCONES>,
    params: &IpmAlgoParams,
    ws:     &mut SocpWorkspace<NP, NE, NCT>,
) -> SocpResult<NP, NE, NCT> {
    // ---- Cone-descriptor validation (panic-freedom on external input) ----
    if !cones_valid::<NCT, NCONES>(&prob.cones) {
        return SocpResult {
            x: SVector::zeros(), lambda: SVector::zeros(),
            s: SVector::zeros(), y: SVector::zeros(),
            status: IpmStatus::NumericalError,
            iters:  0,
        };
    }

    // ---- Initialization ----
    //
    // `x` and `λ` either reset to zero (cold start, default) or kept at
    // their caller-provided values (warm start; SCvx outer loop uses this).
    //
    // For `s` we want primal feasibility (`G·x + s = h`). With warm-start
    // `x = x_ref`, that means `s = h − G·x_ref` per cone — if that's
    // interior. Cold-start `s` falls back to `h` if `h` itself is interior,
    // else `(max(|h[0]|, 1), 0, …)`.
    if !params.warm_start_x {
        ws.x      = SVector::zeros();
        ws.lambda = SVector::zeros();
        init_per_cone_warm_start(&prob.h, &prob.cones, &mut ws.s);
        init_per_cone_warm_start(&prob.h, &prob.cones, &mut ws.y);
    } else {
        // Caller pre-seeded `ws.x` and `ws.lambda`. Compute `s` from the
        // primal-feasibility residual using the caller-provided `x`.
        let gx = prob.g_mat * ws.x;
        let s_target = prob.h - gx;
        init_per_cone_warm_start(&s_target, &prob.cones, &mut ws.s);
        // For `y`, fall back to the cold-start interior point (the dual
        // analogue requires the dual cost gradient, which we don't have).
        init_per_cone_warm_start(&prob.h, &prob.cones, &mut ws.y);
    }
    ws.best_mu    = f64::INFINITY;
    ws.best_valid = false;

    // Pre-build the per-cone e vector once (corrector RHS needs it).
    let e_vec = per_cone_e::<NCT, NCONES>(&prob.cones);

    let ncone = NCT as f64;
    // Loose tolerances for the best-feasible screen (sqrt of strict tols).
    let loose_dual   = libm::sqrt(params.tol_dual.max(0.0)).max(1.0e-5);
    let loose_primal = libm::sqrt(params.tol_primal.max(0.0)).max(1.0e-5);
    let loose_mu     = libm::sqrt(params.tol_mu.max(0.0)).max(1.0e-5);

    let mut prev_x = ws.x;
    let mut prev_y = ws.y;

    for iter in 0..params.max_iters.min(IPM_HARD_MAX_ITERS) {
        // ---- Residuals at current iterate ----
        let r_x = prob.c
                + prob.a_mat.transpose() * ws.lambda
                + prob.g_mat.transpose() * ws.y;             // dual stationarity
        let r_a = prob.a_mat * ws.x - prob.b;                // primal equality
        let r_g = prob.g_mat * ws.x + ws.s - prob.h;         // primal cone
        let mut r_c_aff = SVector::<f64, NCT>::zeros();      // = y ∘ s
        jordan_per_cone(&ws.y, &ws.s, &prob.cones, &mut r_c_aff);

        // ---- Termination check (strict tolerances) ----
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

        // ---- Best-feasible-iterate snapshot ----
        if dual_r < loose_dual
            && primal_r < loose_primal
            && mu.is_finite()
            && mu >= 0.0
            && mu < ws.best_mu
        {
            ws.best_x      = ws.x;
            ws.best_lambda = ws.lambda;
            ws.best_s      = ws.s;
            ws.best_y      = ws.y;
            ws.best_mu     = mu;
            ws.best_valid  = true;
        }

        // ---- Graceful early exit when AHO is about to degenerate ----
        //
        // AHO arrow(s)·arrow(μ) goes singular as both sit on the cone
        // boundary at the optimum. Once we have a best-feasible iterate at
        // `μ < loose_mu`, pushing further mostly produces NaN. Stop while
        // we still have a useful answer.
        if iter > 2 && ws.best_valid && ws.best_mu < loose_mu {
            return numerical_exit(ws, IpmStatus::BestFeasible, iter);
        }

        // No-progress fallback (slower-tightening case)
        let dx_iter = (ws.x - prev_x).norm();
        let dy_iter = (ws.y - prev_y).norm();
        if iter > 0 && dx_iter < 1.0e-9 && dy_iter < 1.0e-9 && ws.best_valid && ws.best_mu < loose_mu {
            return numerical_exit(ws, IpmStatus::BestFeasible, iter);
        }
        prev_x = ws.x;
        prev_y = ws.y;

        // ---- Factor the Newton matrix (reused for affine + corrector) ----
        let fac = match build_step_factors(prob, &ws.s, &ws.y, params.use_adaptive_regularization, params.regularization) {
            Some(f) => f,
            None    => return numerical_exit(ws, IpmStatus::NumericalError, iter),
        };

        // ---- Affine (predictor) step ----
        let (_dx_a, _dl_a, ds_a, dy_a) =
            solve_newton_step(prob, &fac, &r_x, &r_a, &r_g, &r_c_aff);

        // Same NaN guard as the corrector — if the predictor produces
        // garbage, the centering parameter σ will be NaN and poison the
        // corrector RHS.
        let affine_finite = ds_a.iter().all(|v| v.is_finite())
            && dy_a.iter().all(|v| v.is_finite());
        if !affine_finite {
            return numerical_exit(ws, IpmStatus::NumericalError, iter + 1);
        }

        let alpha_s_aff = clip01(BACKOFF * max_step_all_cones(&ws.s, &ds_a, &prob.cones));
        let alpha_y_aff = clip01(BACKOFF * max_step_all_cones(&ws.y, &dy_a, &prob.cones));

        // Mehrotra centering: estimate `μ_aff = (s+α·Δs_a)ᵀ(y+β·Δy_a)/N`
        // using the clipped affine step lengths, then σ = (μ_aff/μ)³.
        let s_aff = ws.s + ds_a * alpha_s_aff;
        let y_aff = ws.y + dy_a * alpha_y_aff;
        let mu_aff = s_aff.dot(&y_aff) / ncone;
        // `.powi(3)` is std-only — manual cube keeps us no_std-clean.
        let sigma_raw = if mu > 1.0e-300 {
            let r = mu_aff / mu;
            r * r * r
        } else {
            1.0
        };
        let sigma = if sigma_raw.is_finite() { sigma_raw.clamp(0.0, 1.0) } else { 0.5 };

        // ---- Corrector RHS: r_c = y∘s + Δy_aff ∘ Δs_aff − σμ e ----
        //
        // The second-order term is the Taylor remainder from expanding
        // `(s + Δs)∘(y + Δy)` around the current iterate; it is the *full*
        // affine direction product, NOT scaled by the affine step lengths.
        // (Mehrotra / Wright "Primal-Dual Interior-Point Methods" §10.5.)
        let mut second_order = SVector::<f64, NCT>::zeros();
        jordan_per_cone(&dy_a, &ds_a, &prob.cones, &mut second_order);
        let r_c = r_c_aff + second_order - e_vec * (sigma * mu);

        // ---- Corrector solve ----
        let (dx, dl, ds, dy) =
            solve_newton_step(prob, &fac, &r_x, &r_a, &r_g, &r_c);

        // ---- Red-team defense: reject Newton steps that contain non-finite
        // entries BEFORE they reach the iterate.
        //
        // Without this guard, a NaN direction passes through `soc_max_step`
        // unscathed (every comparison with NaN evaluates to false, so the
        // function returns `+∞`), `clip01(+∞) = 1.0` per the audit-fixed
        // semantics, and we apply a full Newton step in a NaN direction —
        // poisoning every iterate downstream. The post-step interior check
        // doesn't fire because by then `ws.x` is NaN but `ws.s, ws.y` may
        // still look "interior" against NaN tail norms.
        let step_finite = dx.iter().all(|v| v.is_finite())
            && dl.iter().all(|v| v.is_finite())
            && ds.iter().all(|v| v.is_finite())
            && dy.iter().all(|v| v.is_finite());
        if !step_finite {
            return numerical_exit(ws, IpmStatus::NumericalError, iter + 1);
        }

        // ---- Step lengths (corrector) ----
        let alpha_s = clip01(BACKOFF * max_step_all_cones(&ws.s, &ds, &prob.cones));
        let alpha_y = clip01(BACKOFF * max_step_all_cones(&ws.y, &dy, &prob.cones));

        // ---- Apply ----
        ws.x      += dx * alpha_s;
        ws.lambda += dl * alpha_y;
        ws.s      += ds * alpha_s;
        ws.y      += dy * alpha_y;

        // Defense-in-depth: confirm we didn't fall out of the interior.
        if !all_cones_interior(&ws.s, &prob.cones) || !all_cones_interior(&ws.y, &prob.cones) {
            return numerical_exit(ws, IpmStatus::NumericalError, iter + 1);
        }

        // Red-team defense: catch iterate explosion before f64 overflow
        // produces silent NaN downstream. When AHO is misbehaving (e.g.,
        // the Schur factorization is poorly conditioned and Δ-magnitudes
        // grow without bound), the iterate magnitudes can climb past 1e100
        // and ultimately overflow to ±∞. `inf − inf = NaN` in the cost
        // computation `c·x` poisons every consumer downstream. We bail at
        // 1e50 — comfortably above any sensible problem scale, far below
        // overflow.
        let max_abs = ws.x.amax()
            .max(ws.s.amax())
            .max(ws.y.amax())
            .max(ws.lambda.amax());
        if !max_abs.is_finite() || max_abs > 1.0e50 {
            return numerical_exit(ws, IpmStatus::NumericalError, iter + 1);
        }
    }

    // Hard iter cap. Same scrub-and-rebrand as numerical_exit so an
    // overflow-but-not-yet-NaN iterate doesn't leak garbage to the caller.
    if ws.best_valid {
        SocpResult {
            x: ws.best_x, lambda: ws.best_lambda,
            s: ws.best_s, y: ws.best_y,
            status: IpmStatus::BestFeasible,
            iters:  params.max_iters.min(IPM_HARD_MAX_ITERS),
        }
    } else {
        numerical_exit(ws, IpmStatus::IterCap, params.max_iters.min(IPM_HARD_MAX_ITERS))
    }
}

// ---------------------------------------------------------------------------
// NT-direction Mehrotra driver
// ---------------------------------------------------------------------------

/// Same contract as `solve_socp` but with the **Nesterov-Todd** direction
/// instead of AHO. The reduced Hessian `H = Gᵀ·W²·G` is symmetric PD, which
/// eliminates the AHO endgame degeneracy near the cone boundary. NT
/// scaling is the matrix-form operator geometric mean (Sturm 1999) —
/// exact when `arrow(s)`/`arrow(y)` commute, an excellent approximation
/// otherwise.
///
/// For convex SOCPs, this should converge tighter (smaller `μ`) and faster
/// than AHO on problems where the optimum lies on the cone boundary.
pub fn solve_socp_nt<
    const NP:     usize,
    const NE:     usize,
    const NCT:    usize,
    const NCONES: usize,
>(
    prob:   &SocpProblem<NP, NE, NCT, NCONES>,
    params: &IpmAlgoParams,
    ws:     &mut SocpWorkspace<NP, NE, NCT>,
) -> SocpResult<NP, NE, NCT> {
    // ---- Cone-descriptor validation (panic-freedom on external input) ----
    if !cones_valid::<NCT, NCONES>(&prob.cones) {
        return SocpResult {
            x: SVector::zeros(), lambda: SVector::zeros(),
            s: SVector::zeros(), y: SVector::zeros(),
            status: IpmStatus::NumericalError,
            iters:  0,
        };
    }

    // ---- Initialization (same as AHO solve_socp) ----
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
    ws.best_mu    = f64::INFINITY;
    ws.best_valid = false;

    let ncone = NCT as f64;
    let loose_dual   = libm::sqrt(params.tol_dual.max(0.0)).max(1.0e-5);
    let loose_primal = libm::sqrt(params.tol_primal.max(0.0)).max(1.0e-5);
    let loose_mu     = libm::sqrt(params.tol_mu.max(0.0)).max(1.0e-5);

    let mut prev_x = ws.x;
    let mut prev_y = ws.y;

    // Loop-invariant per-cone identity `e` (corrector needs `arrow(s̃)⁻¹·e`).
    // Hoisted out of the iteration loop (matches the AHO `solve_socp`).
    let e_vec = per_cone_e::<NCT, NCONES>(&prob.cones);

    for iter in 0..params.max_iters.min(IPM_HARD_MAX_ITERS) {
        // ---- Residuals at current iterate (in ORIGINAL coords) ----
        let r_x = prob.c
                + prob.a_mat.transpose() * ws.lambda
                + prob.g_mat.transpose() * ws.y;
        let r_a = prob.a_mat * ws.x - prob.b;
        let r_g = prob.g_mat * ws.x + ws.s - prob.h;

        // Complementarity in ORIGINAL coords (for termination checks).
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

        // Best-feasible snapshot — same logic as AHO.
        if dual_r < loose_dual
            && primal_r < loose_primal
            && mu.is_finite()
            && mu >= 0.0
            && mu < ws.best_mu
        {
            ws.best_x      = ws.x;
            ws.best_lambda = ws.lambda;
            ws.best_s      = ws.s;
            ws.best_y      = ws.y;
            ws.best_mu     = mu;
            ws.best_valid  = true;
        }

        // No-progress / endgame guards — NT shouldn't normally degenerate
        // at the boundary, but keep the defenses defensively.
        if iter > 2 && ws.best_valid && ws.best_mu < loose_mu {
            return numerical_exit(ws, IpmStatus::BestFeasible, iter);
        }
        let dx_iter = (ws.x - prev_x).norm();
        let dy_iter = (ws.y - prev_y).norm();
        if iter > 0 && dx_iter < 1.0e-9 && dy_iter < 1.0e-9
            && ws.best_valid && ws.best_mu < loose_mu
        {
            return numerical_exit(ws, IpmStatus::BestFeasible, iter);
        }
        prev_x = ws.x;
        prev_y = ws.y;

        // ---- NT factors ----
        let fac = match build_step_factors_nt(prob, &ws.s, &ws.y, params.use_adaptive_regularization, params.regularization) {
            Some(f) => f,
            None    => return numerical_exit(ws, IpmStatus::NumericalError, iter),
        };

        // ---- Affine step ----
        // In NT, F̃_4_aff = arrow(s̃)·s̃ ⇒ arrow(s̃)⁻¹·F̃_4_aff = s̃.
        let r_c_arg_aff = fac.s_scaled;

        // `dx_a`/`dl_a` feed the scaled-coords corrector AND the weighted-
        // corrector blend below — all four affine components are retained.
        let (dx_a, dl_a, ds_a, dy_a) =
            solve_newton_step_nt(prob, &fac, &r_x, &r_a, &r_g, &r_c_arg_aff);

        let affine_finite = ds_a.iter().all(|v| v.is_finite())
            && dy_a.iter().all(|v| v.is_finite())
            && dx_a.iter().all(|v| v.is_finite())
            && dl_a.iter().all(|v| v.is_finite());
        if !affine_finite {
            return numerical_exit(ws, IpmStatus::NumericalError, iter + 1);
        }

        let alpha_s_aff = clip01(BACKOFF * max_step_all_cones(&ws.s, &ds_a, &prob.cones));
        let alpha_y_aff = clip01(BACKOFF * max_step_all_cones(&ws.y, &dy_a, &prob.cones));

        // ---- Mehrotra centering σ with the SDPT3 **adaptive exponent** ----
        //
        // SDPT3 (Toh-Todd-Tütüncü, NT direction): σ = min(1, (μ_aff/μ)^e) with
        //   e = max(1, 3·min(α_p,α_d)²)   for μ > 1e-6,
        //   e = 1                          for μ ≤ 1e-6.
        // The fixed `e = 3` we used before over-reduces σ (drives toward the
        // affine/predictor end) even when the achievable affine step is small —
        // which is destabilizing on the ill-conditioned flight subproblem. The
        // adaptive exponent backs off to gentler centering (e→1) precisely
        // when steps are short. (Research: SDPT3 v4.0 implementation paper.)
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

        // ---- Corrector r_c_arg ----
        //
        //   F̃_4_corr           = s̃² + (Δs̃_aff ∘ Δỹ_aff) − σμ·e
        //   arrow(s̃)⁻¹·F̃_4    = s̃ + arrow(s̃)⁻¹·(Δs̃_aff ∘ Δỹ_aff) − σμ·s̃⁻¹
        //
        // Compute Δs̃_aff, Δỹ_aff in scaled coords from the affine step:
        //   Δs̃_aff = W·Δs_aff, Δỹ_aff = W⁻¹·Δy_aff
        // — then the Jordan product per cone gives the second-order term.
        let ds_aff_scaled = fac.w * ds_a;
        // For Δỹ_aff = W⁻¹·Δy_aff, we need W⁻¹. But we don't have it stored.
        // Use the relation Δỹ_aff = arrow(s̃)⁻¹·(arrow(s̃)·F̃_3 + arrow(s̃)·G̃·Δx_aff − F̃_4_aff)
        //                       = F̃_3 + G̃·Δx_aff − arrow(s̃)⁻¹·F̃_4_aff
        //                       = W·(r_g + G·Δx_aff) − s̃
        let dy_aff_scaled =
            fac.w * (r_g + prob.g_mat * dx_a) - fac.s_scaled;

        // Per-cone Jordan product Δs̃_aff ∘ Δỹ_aff.
        let mut second_order = SVector::<f64, NCT>::zeros();
        jordan_per_cone(&ds_aff_scaled, &dy_aff_scaled, &prob.cones, &mut second_order);

        // arrow(s̃)⁻¹·second_order
        let arrow_s_inv_second = fac.arrow_s_scaled_inv * second_order;

        // arrow(s̃)⁻¹·e = s̃⁻¹ — but rather than computing the Jordan inverse
        // per cone, we use arrow_s_scaled_inv multiplied by the per-cone
        // identity e (which is `per_cone_e` from earlier).
        let s_scaled_inv = fac.arrow_s_scaled_inv * e_vec;

        // Centering target `σμ·e` (global). NOTE (NT/O(N) frontier, Phase 24):
        // a per-cone target `σ·μ_c·e_c` (wide-neighborhood-flavored) was
        // implemented behind a flag and measured on the flight subproblem — it
        // is a byte-identical NO-OP at every altitude, because the
        // Colombo-Gondzio blend already drives ω→0 (rejecting the corrector) on
        // this ill-conditioned subproblem, so any corrector-centering change is
        // moot. The divergence is in the affine NT step (the `H = GᵀW²G`
        // ill-conditioning), not the centering. Reverted; recorded so it isn't
        // re-treaded. See HANDOFF "Phase 25".
        let r_c_arg_corr = fac.s_scaled + arrow_s_inv_second - s_scaled_inv * (sigma * mu);

        // ---- Corrector solve ----
        let (dx_c, dl_c, ds_c, dy_c) =
            solve_newton_step_nt(prob, &fac, &r_x, &r_a, &r_g, &r_c_arg_corr);

        let step_finite = dx_c.iter().all(|v| v.is_finite())
            && dl_c.iter().all(|v| v.is_finite())
            && ds_c.iter().all(|v| v.is_finite())
            && dy_c.iter().all(|v| v.is_finite());
        if !step_finite {
            return numerical_exit(ws, IpmStatus::NumericalError, iter + 1);
        }

        // ---- Weighted corrector (Colombo–Gondzio) ----
        //
        // The full Mehrotra corrector direction `Δ_c` can be far larger than
        // the affine predictor `Δ_a` and point the wrong way — a PROVEN
        // failure mode (Cartis 2009; Colombo–Gondzio 2008) that diverges in
        // exact AND finite arithmetic regardless of step rule. The fix is to
        // blend: `Δ(ω) = Δ_a + ω·(Δ_corr − Δ_a)`, ω ∈ [0,1], choosing ω each
        // iteration to MAXIMIZE the achievable fraction-to-boundary step.
        // ω = 1 recovers Mehrotra (good when the corrector helps); ω → 0
        // falls back to the safe affine predictor (when the corrector is
        // destructive). This is the single most-cited safeguard against the
        // endgame μ→NaN divergence we observed on the flight subproblem.
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
        // 6-point line-search on ω ∈ [0,1]; objective = min raw fraction-to-
        // boundary step over (s, y). Default to the full corrector (ω = 1).
        const OMEGA_GRID: [f64; 6] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
        let mut best_omega   = 1.0_f64;
        let mut best_obj     = f64::NEG_INFINITY;
        let mut best_raw_s   = 0.0_f64;
        let mut best_raw_y   = 0.0_f64;
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

        // ---- Adaptive fraction-to-boundary (SDPT3): γ = 0.9 + 0.09·min(αp,αd) ----
        //
        // A fixed γ = 0.99 lets iterates crowd the cone boundary when the raw
        // step is small/uncertain (which precedes the endgame breakdown).
        // SDPT3's adaptive rule is conservative (0.9) when the achievable step
        // is small and approaches 0.99 as it nears a full step — keeping the
        // iterate strictly interior exactly when conditioning is worst.
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
            return numerical_exit(ws, IpmStatus::NumericalError, iter + 1);
        }
        let max_abs = ws.x.amax().max(ws.s.amax()).max(ws.y.amax()).max(ws.lambda.amax());
        if !max_abs.is_finite() || max_abs > 1.0e50 {
            return numerical_exit(ws, IpmStatus::NumericalError, iter + 1);
        }
    }

    if ws.best_valid {
        SocpResult {
            x: ws.best_x, lambda: ws.best_lambda,
            s: ws.best_s, y: ws.best_y,
            status: IpmStatus::BestFeasible,
            iters:  params.max_iters.min(IPM_HARD_MAX_ITERS),
        }
    } else {
        numerical_exit(ws, IpmStatus::IterCap, params.max_iters.min(IPM_HARD_MAX_ITERS))
    }
}

/// Common exit path for numerical breakdown / no-progress termini.
///
/// **Caller defense:** if `best_valid` is false (no clean snapshot was
/// captured) AND the live iterate contains non-finite entries, scrub them
/// to zero before returning. This prevents NaN/inf from poisoning consumers
/// downstream — typically the outer SCvx loop's cost/residual computation.
/// We always force the status to `NumericalError` in that case so callers
/// know the result is not a true solution, just a guaranteed-finite stand-in.
fn numerical_exit<const NP: usize, const NE: usize, const NCT: usize>(
    ws:     &SocpWorkspace<NP, NE, NCT>,
    status_if_no_best: IpmStatus,
    iter:   u32,
) -> SocpResult<NP, NE, NCT> {
    if ws.best_valid {
        SocpResult {
            x:      ws.best_x,
            lambda: ws.best_lambda,
            s:      ws.best_s,
            y:      ws.best_y,
            status: IpmStatus::BestFeasible,
            iters:  iter,
        }
    } else {
        // Scrub non-finite entries so downstream consumers never see NaN/inf.
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
        let status = if had_dirty {
            IpmStatus::NumericalError
        } else {
            status_if_no_best
        };
        SocpResult { x, lambda, s, y, status, iters: iter }
    }
}

// ---------------------------------------------------------------------------
// HSD-direction Mehrotra driver (homogeneous self-dual embedding).
// ---------------------------------------------------------------------------

/// Apply the cached NT Schur factor to ONE right-hand side `(bx, ba)`, solving
/// ```text
///   [ H   Aᵀ ] [Δx]   [ bx ]
///   [ A   0  ] [Δλ] = [ ba ]
/// ```
/// via the Schur complement `S = A·H⁻¹·Aᵀ` (both `H⁻¹` and `S⁻¹` precomputed in
/// `fac`). HSD solves this core system TWICE per Newton iteration — once for the
/// residual RHS and once for the constant τ-coupling column `[c; b; h]` — so the
/// expensive factor is built once (`build_step_factors_nt`) and applied many
/// times, the same factor/apply economy the structured drivers use.
#[inline]
fn nt_schur_apply<
    const NP: usize, const NE: usize, const NCT: usize, const NCONES: usize,
>(
    prob: &SocpProblem<NP, NE, NCT, NCONES>,
    fac:  &NtStepFactors<NP, NE, NCT>,
    bx:   &SVector<f64, NP>,
    ba:   &SVector<f64, NE>,
) -> (SVector<f64, NP>, SVector<f64, NE>) {
    let dl = fac.s_inv * (prob.a_mat * (fac.h_inv * bx) - ba);
    let dx = fac.h_inv * (bx - prob.a_mat.transpose() * dl);
    (dx, dl)
}

/// Mehrotra predictor-corrector IPM on the **homogeneous self-dual (HSD)
/// embedding** of the SOCP — the NT/O(N) frontier path (HANDOFF "Phase 26").
///
/// ## Why HSD (the crack)
///
/// The plain primal-dual NT driver ([`solve_socp_nt`]) DIVERGES on the flight-
/// scale SCvx subproblem: its virtual-control SOC^8 cones VANISH at the optimum
/// (the relaxation drives ν→0), the iterates drift OFF the central path (per-
/// cone complementarity spans ~10^11), and `H = Gᵀ·W²·G` becomes catastrophically
/// ill-conditioned. Phases 15/22/25 measured the three incremental NT levers
/// (exact per-cone scaling, per-cone/wide-neighborhood centering, iterative
/// refinement) as no-ops: symmetric-NT scaling intrinsically produces an
/// ill-conditioned affine step when cones vanish off-centre.
///
/// HSD is the production-solver fix (ECOS, Clarabel). It embeds (P)/(D) into a
/// single SELF-DUAL system with two extra scalars — `τ` (homogenizing) and `κ`
/// (its complementary slack) — and path-follows the EMBEDDED central path, where
/// every cone AND the `(τ,κ)` ray share ONE `μ`. That near-complementarity
/// (`s∘z ≈ μe` per cone) is exactly the regime where
/// [`crate::cone::soc_nt_scaling_exact`] is provably bounded (`s̄≈z̄ ⇒ w̄≈e ⇒
/// W≈η·I`) even as `det→0`, so the `W²`-spread that destroys the plain NT step
/// never forms. The homogenizing `τ` also removes the need for a feasible
/// warm-start: HSD cold-starts from the strictly-feasible central point.
///
/// ## Embedding (this crate's standard form `min cᵀx s.t. Ax=b, Gx+s=h, s∈K`)
///
/// Find `(x, λ, z, s, τ, κ)`, `s,z ∈ K`, `τ,κ ≥ 0`, solving the skew-symmetric
/// self-dual system as `μ → 0` (`z` is the cone dual the other drivers call `y`):
/// ```text
///   Aᵀλ + Gᵀz + cτ        = 0      (R_x  dual feasibility, homogenized)
///   Ax            − bτ    = 0      (R_λ  primal-equality feasibility)
///   Gx + s        − hτ    = 0      (R_g  primal-cone feasibility)
///   cᵀx + bᵀλ + hᵀz + κ   = 0      (R_t  the self-dual gap row)
///   s ∘ z = μ e ,   τ κ = μ
/// ```
/// The 4×4 operator is skew-symmetric (`Mᵀ = −M`), so the embedding is self-dual.
/// At `μ=0`: `τ>0 ⇒ (x,λ,s,z)/τ` solves (P)/(D); `τ=0, κ>0 ⇒` a primal/dual
/// infeasibility certificate. Central start: `x=0, λ=0, s=z=e, τ=κ=1` (`μ₀=1`).
///
/// ## Newton step (NT-scaled; τ via two-RHS + scalar-Δτ reduction)
///
/// Reuses [`build_step_factors_nt`] for the `H=Gᵀ·W²·G`, `S=A·H⁻¹·Aᵀ`
/// factorization at the embedded `(s,z)`. The `τ`-coupling makes the system
/// non-separable; eliminating `Δs` (cone-feasibility row) and `Δz` (NT
/// complementarity, giving the same `Δz = −W·r_c_arg − W²·Δs` relation the plain
/// NT driver uses) leaves a `(Δx,Δλ)` system that is AFFINE in the scalar `Δτ`:
/// ```text
///   [H Aᵀ][Δx]   [ bx_res + (GᵀW²h − c)·Δτ ]
///   [A 0 ][Δλ] = [ ba_res + b·Δτ           ]
/// ```
/// Solve the core system for the residual RHS `(bx_res, ba_res)` and the constant
/// τ-column `(GᵀW²h−c, b)` → `(Δx_r,Δλ_r)`, `(Δx_t,Δλ_t)`. Substituting
/// `Δx = Δx_r + Δτ·Δx_t` into the gap row gives ONE scalar equation for `Δτ`
/// (denominator `coef`), then back-substitute `Δx,Δλ,Δs,Δz,Δκ`. Derivation inline.
///
/// ## Contract
///
/// Same return shape as [`solve_socp`]/[`solve_socp_nt`]: the recovered
/// (de-homogenized) `(x,λ,s,y) = (x,λ,s,z)/τ`, a status, and the iteration count.
/// HSD cold-starts centrally (ignores `warm_start_x` / the seeded `ws.x`); `ws`
/// is reused only for the recovered iterate + best-feasible snapshot. Every
/// buffer is on the stack; no alloc, no panic on any path; the loop is bounded by
/// `min(max_iters, IPM_HARD_MAX_ITERS)`.
pub fn solve_socp_hsd<
    const NP:     usize,
    const NE:     usize,
    const NCT:    usize,
    const NCONES: usize,
>(
    prob:   &SocpProblem<NP, NE, NCT, NCONES>,
    params: &IpmAlgoParams,
    ws:     &mut SocpWorkspace<NP, NE, NCT>,
) -> SocpResult<NP, NE, NCT> {
    // ---- Cone-descriptor validation (panic-freedom on external input) ----
    if !cones_valid::<NCT, NCONES>(&prob.cones) {
        return SocpResult {
            x: SVector::zeros(), lambda: SVector::zeros(),
            s: SVector::zeros(), y: SVector::zeros(),
            status: IpmStatus::NumericalError,
            iters:  0,
        };
    }

    // ---- Embedded variables (local; recovered into `ws` each iteration) ----
    // Strictly-feasible central start: x=0, λ=0, s=z=e (per-cone identity),
    // τ=κ=1. No warm-start needed — that is HSD's structural advantage.
    let e_vec = per_cone_e::<NCT, NCONES>(&prob.cones);
    let mut xe    = SVector::<f64, NP>::zeros();
    let mut le    = SVector::<f64, NE>::zeros();
    let mut se    = e_vec;
    let mut ze    = e_vec;
    let mut tau   = 1.0_f64;
    let mut kappa = 1.0_f64;

    ws.best_mu    = f64::INFINITY;
    ws.best_valid = false;

    // Barrier degree of the embedded cone K × ℝ₊ = (#cones) + 1 (the (τ,κ) ray).
    // The SOC barrier degree is 1 PER cone (not its dim): on the central path
    // `s_c ∘ z_c = μ e_c` gives `s_c·z_c = μ`, so total `s·z = μ·NCONES`.
    let degree       = (NCONES + 1) as f64;
    let loose_primal = libm::sqrt(params.tol_primal.max(0.0)).max(1.0e-5);
    let loose_mu     = libm::sqrt(params.tol_mu.max(0.0)).max(1.0e-5);

    let g_t = prob.g_mat.transpose();

    for iter in 0..params.max_iters.min(IPM_HARD_MAX_ITERS) {
        // `τ`/`κ` must stay finite and strictly positive (the embedded ℝ₊ ray).
        let tau_ok   = tau.is_finite()   && tau   > 0.0;
        let kappa_ok = kappa.is_finite() && kappa > 0.0;
        if !tau_ok || !kappa_ok {
            return numerical_exit(ws, IpmStatus::NumericalError, iter);
        }

        // ---- Embedded residuals (drive each to 0) ----
        let r_x = prob.c * tau
                + prob.a_mat.transpose() * le
                + g_t * ze;                                  // R_x  (NP)
        let r_a = prob.a_mat * xe - prob.b * tau;            // R_λ  (NE)
        let r_g = prob.g_mat * xe + se - prob.h * tau;       // R_g  (NCT)
        let r_t = kappa
                + prob.c.dot(&xe)
                + prob.b.dot(&le)
                + prob.h.dot(&ze);                           // R_t  (scalar)

        let mu = (se.dot(&ze) + tau * kappa) / degree;

        // ---- Recover the current iterate into `ws` (live fallback) ----
        let inv_tau = 1.0 / tau;
        ws.x      = xe * inv_tau;
        ws.lambda = le * inv_tau;
        ws.s      = se * inv_tau;
        ws.y      = ze * inv_tau;

        // De-homogenized residuals (the original-problem residuals are the
        // embedded residuals divided by τ).
        let primal_r = (r_a.norm() + r_g.norm()) * inv_tau;
        let dual_r   = r_x.norm() * inv_tau;

        // ---- Termination (strict) ----
        if mu < params.tol_mu && primal_r < params.tol_primal && dual_r < params.tol_dual {
            return SocpResult {
                x: ws.x, lambda: ws.lambda, s: ws.s, y: ws.y,
                status: IpmStatus::Optimal,
                iters:  iter,
            };
        }

        // ---- Best-feasible snapshot (recovered space) ----
        // Key on primal feasibility + the embedded gap `μ` (the HSD
        // suboptimality measure). The dual residual is deliberately NOT gated
        // here: with a large preconditioned cost vector (`virt_weight ~ 1e5`)
        // the recovered dual residual's absolute floor sits above the loose
        // tolerance even at the optimum, yet small `μ` + primal feasibility +
        // `τ > 0` already certify near-optimality in the self-dual embedding.
        if tau > 0.0
            && primal_r < loose_primal
            && mu.is_finite()
            && mu >= 0.0
            && mu < ws.best_mu
        {
            ws.best_x      = ws.x;
            ws.best_lambda = ws.lambda;
            ws.best_s      = ws.s;
            ws.best_y      = ws.y;
            ws.best_mu     = mu;
            ws.best_valid  = true;
        }

        // ---- Graceful endgame exit (the robustness fix) ----
        // Once a good-enough snapshot exists, STOP. Driving `μ → 0` further
        // pushes the embedded `(τ,κ) → 0` region where the NT scaling
        // re-conditions badly and `τ` can collapse (the recovered `x/τ` then
        // blows up — the measured alt=100 breakdown at iter ~30). Exit when the
        // snapshot gap is already tight (`< loose_mu`), OR when the LIVE gap has
        // regressed past 4× the best (overshoot past the optimum detected) —
        // returning the captured near-optimal iterate. Mirrors the AHO/NT
        // endgame guards; without it HSD reaches the optimum (~iter 20) then
        // diverges by iter ~30.
        if iter > 2 && ws.best_valid && (ws.best_mu < loose_mu || mu > 4.0 * ws.best_mu) {
            return numerical_exit(ws, IpmStatus::BestFeasible, iter);
        }

        // ---- NT factor at the EMBEDDED cone iterate (s, z) ----
        // `build_step_factors_nt` routes per cone through `soc_nt_scaling_exact`
        // (bounded on the central path), giving `W`, `W²`, `arrow(s̃)⁻¹`, `s̃=W·s`,
        // `H⁻¹`, `S⁻¹`. This is the expensive factor — reused for both RHS below.
        let fac = match build_step_factors_nt(
            prob, &se, &ze, params.use_adaptive_regularization, params.regularization,
        ) {
            Some(f) => f,
            None    => return numerical_exit(ws, IpmStatus::NumericalError, iter),
        };
        let w  = fac.w;
        let w2 = fac.w_squared;

        // ---- τ-coupling precompute (independent of σ / corrector) ----
        // τ-column of the core system: bx_tau = GᵀW²h − c, ba_tau = b.
        // d = c + GᵀW²h ; coef = the Δτ-equation denominator (gap row).
        let gw2h   = g_t * (w2 * prob.h);
        let bx_tau = gw2h - prob.c;
        let (dx_t, dl_t) = nt_schur_apply(prob, &fac, &bx_tau, &prob.b);
        let d      = prob.c + gw2h;
        let h_w2_h = prob.h.dot(&(w2 * prob.h));
        let coef   = d.dot(&dx_t) + prob.b.dot(&dl_t) - h_w2_h - kappa / tau;
        if !coef.is_finite() || coef.abs() < 1.0e-300 {
            return numerical_exit(ws, IpmStatus::NumericalError, iter);
        }

        // ---- HSD Newton direction for a given (r_c_arg, r6) ----
        // `r_c_arg` = scaled-coords complementarity argument `arrow(s̃)⁻¹·F̃₄`;
        // `r6` = the scalar (τ,κ) complementarity RHS `σμ − τκ [− Δτ_a·Δκ_a]`.
        let hsd_dir = |r_c_arg: &SVector<f64, NCT>, r6: f64| {
            // Residual RHS of the core system:
            //   bx_res = −R_x + Gᵀ·W·r_c_arg − Gᵀ·W²·R_g ; ba_res = −R_λ.
            let bx_res = -r_x + g_t * (w * r_c_arg) - g_t * (w2 * r_g);
            let ba_res = -r_a;
            let (dx_r, dl_r) = nt_schur_apply(prob, &fac, &bx_res, &ba_res);
            // Gap-row scalar equation:  const + coef·Δτ = −R_t.
            let const_lhs = d.dot(&dx_r) + prob.b.dot(&dl_r)
                - prob.h.dot(&(w * r_c_arg)) + prob.h.dot(&(w2 * r_g))
                + r6 / tau;
            let dtau = (-r_t - const_lhs) / coef;
            let dx   = dx_r + dx_t * dtau;
            let dl   = dl_r + dl_t * dtau;
            // Δs = −R_g − G·Δx + h·Δτ ;  Δz = −W·r_c_arg − W²·Δs ;
            // Δκ = (r6 − κ·Δτ)/τ.
            let ds   = -r_g - prob.g_mat * dx + prob.h * dtau;
            let dz   = -(w * r_c_arg) - w2 * ds;
            let dkap = (r6 - kappa * dtau) / tau;
            (dx, dl, ds, dz, dtau, dkap)
        };

        // Fraction-to-boundary of a full direction over (s, z, τ, κ).
        let ftb = |ds: &SVector<f64, NCT>, dz: &SVector<f64, NCT>, dt: f64, dk: f64| -> f64 {
            let a_s = max_step_all_cones(&se, ds, &prob.cones);
            let a_z = max_step_all_cones(&ze, dz, &prob.cones);
            let a_t = if dt < 0.0 { -tau   / dt } else { f64::INFINITY };
            let a_k = if dk < 0.0 { -kappa / dk } else { f64::INFINITY };
            a_s.min(a_z).min(a_t).min(a_k)
        };

        // ---- Affine (predictor): σ = 0, r_c_arg = s̃, r6 = −τκ ----
        // Only the affine (s,z,τ,κ) directions are needed (for σ and the
        // corrector's second-order term); the affine x/λ are recomputed by the
        // corrector, so they are bound to `_`.
        let (_dx_a, _dl_a, ds_a, dz_a, dtau_a, dkap_a) = hsd_dir(&fac.s_scaled, -tau * kappa);
        let aff_finite = ds_a.iter().all(|v| v.is_finite())
            && dz_a.iter().all(|v| v.is_finite())
            && dtau_a.is_finite() && dkap_a.is_finite();
        if !aff_finite {
            return numerical_exit(ws, IpmStatus::NumericalError, iter + 1);
        }
        let alpha_a = clip01(BACKOFF * ftb(&ds_a, &dz_a, dtau_a, dkap_a));

        // Mehrotra centering σ = (μ_aff/μ)³ (the gap the affine step would reach).
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

        // ---- Corrector r_c_arg = s̃ + arrow(s̃)⁻¹·(Δs̃_a∘Δz̃_a) − σμ·s̃⁻¹ ----
        // Scaled affine directions: Δs̃_a = W·Δs_a, Δz̃_a = −s̃ − Δs̃_a (the
        // W⁻¹-free identity the plain NT corrector uses).
        let ds_a_scaled = w * ds_a;
        let dz_a_scaled = -fac.s_scaled - ds_a_scaled;
        let mut second_order = SVector::<f64, NCT>::zeros();
        jordan_per_cone(&ds_a_scaled, &dz_a_scaled, &prob.cones, &mut second_order);
        let arrow_s_inv_second = fac.arrow_s_scaled_inv * second_order;
        let s_scaled_inv       = fac.arrow_s_scaled_inv * e_vec;   // arrow(s̃)⁻¹·e = s̃⁻¹
        let r_c_arg_corr = fac.s_scaled + arrow_s_inv_second - s_scaled_inv * (sigma * mu);
        // Scalar (τ,κ) corrector RHS: σμ − τκ − Δτ_a·Δκ_a.
        let r6_corr = sigma * mu - tau * kappa - dtau_a * dkap_a;

        let (dx, dl, ds, dz, dtau, dkap) = hsd_dir(&r_c_arg_corr, r6_corr);
        let step_finite = dx.iter().all(|v| v.is_finite())
            && dl.iter().all(|v| v.is_finite())
            && ds.iter().all(|v| v.is_finite())
            && dz.iter().all(|v| v.is_finite())
            && dtau.is_finite() && dkap.is_finite();
        if !step_finite {
            return numerical_exit(ws, IpmStatus::NumericalError, iter + 1);
        }

        // ---- Single self-dual step length (one α for the whole direction) ----
        let raw   = ftb(&ds, &dz, dtau, dkap);
        let alpha = clip01(BACKOFF * raw);
        // `clip01` returns a finite value in `[0, 1]`, so `<= 0` is exactly the
        // "no interior step available" case (NaN already mapped to 0 there).
        if alpha <= 0.0 {
            // No interior progress possible — return the best snapshot if any.
            return numerical_exit(ws, IpmStatus::BestFeasible, iter + 1);
        }

        // ---- Apply ----
        xe    += dx   * alpha;
        le    += dl   * alpha;
        se    += ds   * alpha;
        ze    += dz   * alpha;
        tau   += dtau * alpha;
        kappa += dkap * alpha;

        // `τ`/`κ` are finite here (the corrector direction passed `step_finite`
        // and `alpha` is finite), so `<= 0` catches a non-positive overshoot;
        // any residual non-finite is caught by the `max_abs` gate just below.
        if !all_cones_interior(&se, &prob.cones)
            || !all_cones_interior(&ze, &prob.cones)
            || tau <= 0.0 || kappa <= 0.0
        {
            return numerical_exit(ws, IpmStatus::NumericalError, iter + 1);
        }
        let max_abs = xe.amax().max(se.amax()).max(ze.amax()).max(le.amax())
            .max(tau.abs()).max(kappa.abs());
        if !max_abs.is_finite() || max_abs > 1.0e50 {
            return numerical_exit(ws, IpmStatus::NumericalError, iter + 1);
        }
    }

    // Hard iter cap. Prefer the best snapshot; else the live recovered iterate
    // (already in `ws.x/...`) scrubbed for NaN-safety by `numerical_exit`.
    if ws.best_valid {
        SocpResult {
            x: ws.best_x, lambda: ws.best_lambda,
            s: ws.best_s, y: ws.best_y,
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
    use libm::sqrt;

    use super::*;

    /// **Regression test**: a 3-var, 1-cone, 1-eq SOCP with hand-computable
    /// optimum, mirroring `mehrotra::tests::toy_socp_recovers_closed_form`
    /// in spirit but via the generic standard-form IPM.
    ///
    /// Problem:
    ///   min  c·x  =  x_1
    ///   s.t. a·x  =  x_2 + x_3 = 1
    ///        (x_1, x_2, x_3) ∈ SOC^3      (x_1 ≥ ‖(x_2, x_3)‖)
    ///
    /// Standard form: `s = h − Gx` with `G = −I`, `h = 0`, so `s ≡ x`.
    /// Cost vector `c = (1, 0, 0)` makes the problem bounded: `min x_1`
    /// subject to the cone lower-bounding `x_1` by `√(x_2² + x_3²)`.
    /// Closed-form optimum: `x_2 = x_3 = 0.5`, `x_1 = √0.5 ≈ 0.7071`.
    #[test]
    fn regression_toy_socp() {
        const NP: usize     = 3;
        const NE: usize     = 1;
        const NCT: usize    = 3;
        const NCONES: usize = 1;

        let prob = SocpProblem::<NP, NE, NCT, NCONES> {
            c:     SVector::<f64, 3>::from_column_slice(&[1.0, 0.0, 0.0]),
            a_mat: SMatrix::<f64, 1, 3>::from_row_slice(&[0.0, 1.0, 1.0]),
            b:     SVector::<f64, 1>::from_element(1.0),
            g_mat: -SMatrix::<f64, 3, 3>::identity(),
            h:     SVector::<f64, 3>::zeros(),
            cones: [ConeDesc { offset: 0, dim: 3 }],
        };

        let mut ws = SocpWorkspace::<NP, NE, NCT>::default();
        let params = IpmAlgoParams::default();
        let result = solve_socp(&prob, &params, &mut ws);

        eprintln!("regression toy: status as_u32 = {}, iters = {}",
                  result.status.as_u32(), result.iters);
        eprintln!("  x = ({:.6}, {:.6}, {:.6})",
                  result.x[0], result.x[1], result.x[2]);

        // Optimum satisfies: x_2 = x_3 = 0.5, x_1 = √0.5 = 0.7071…
        //
        // AHO degenerates as both `s` and the dual `y` sit on the SOC
        // boundary at the optimum: `arrow(s)·arrow(y)` becomes singular.
        // The IPM exits at `BestFeasible` once `μ < √(tol_mu) ≈ 1e-4`,
        // giving roughly 5-6 digits of precision — fine for the inner
        // solve of an SCvx outer loop (the outer iteration re-linearizes
        // and corrects), and matches the original `solve_toy_socp` regime.
        let expected_x = [sqrt(0.5), 0.5, 0.5];
        for (i, &expected) in expected_x.iter().enumerate() {
            assert!(
                (result.x[i] - expected).abs() < 1.0e-3,
                "x[{i}] = {} vs {} ({:?})",
                result.x[i], expected, result.status.as_u32()
            );
        }
        // Equality constraint to loose-primal tolerance.
        assert!((result.x[1] + result.x[2] - 1.0).abs() < 1.0e-3,
                "equality: x₂ + x₃ = {}", result.x[1] + result.x[2]);
        // Cone constraint (with margin since IPM stops at finite μ).
        let cone_slack = result.x[0] - sqrt(result.x[1].powi(2) + result.x[2].powi(2));
        assert!(cone_slack > -1.0e-3,
                "cone violated: x₁ − ‖(x₂,x₃)‖ = {}", cone_slack);
    }

    /// **New 2-cone test**: mixed SOC + 1D-cone problem with hand-pinned
    /// optimum.
    ///
    /// Problem:
    ///   variables x = (x_1, x_2, x_3) ∈ ℝ³
    ///   min x_3
    ///   s.t. x_1 = 1,  x_2 = 1
    ///        (x_3, x_1, x_2) ∈ SOC^3        (cone 1)
    ///        x_3 - 2 ≥ 0                    (cone 2, = SOC^1 = ℝ_+)
    ///
    /// Cone 1 alone would give x_3 ≥ √2; cone 2 forces x_3 ≥ 2. Optimum:
    /// x = (1, 1, 2), x_3 = 2. Cone 2 binding; cone 1 strictly interior.
    /// At optimum: λ = (0, 0), y = (0, 0, 0, 1).
    #[test]
    fn two_cone_mixed_socp() {
        const NP: usize     = 3;
        const NE: usize     = 2;
        const NCT: usize    = 4;  // 3 (cone 1) + 1 (cone 2)
        const NCONES: usize = 2;

        let prob = SocpProblem::<NP, NE, NCT, NCONES> {
            c:     SVector::<f64, 3>::from_column_slice(&[0.0, 0.0, 1.0]),
            a_mat: SMatrix::<f64, 2, 3>::from_row_slice(&[
                1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
            ]),
            b:     SVector::<f64, 2>::from_column_slice(&[1.0, 1.0]),
            // G: cone 1 rows expose (x_3, x_1, x_2); cone 2 row exposes (x_3 − 2)
            // Encoding `s = h − Gx`:  s_2[0] = (−2) − (−1·x_3) = x_3 − 2.
            g_mat: SMatrix::<f64, 4, 3>::from_row_slice(&[
                 0.0,  0.0, -1.0,   // s_1[0] = x_3
                -1.0,  0.0,  0.0,   // s_1[1] = x_1
                 0.0, -1.0,  0.0,   // s_1[2] = x_2
                 0.0,  0.0, -1.0,   // s_2[0] = x_3 − 2
            ]),
            h:     SVector::<f64, 4>::from_column_slice(&[0.0, 0.0, 0.0, -2.0]),
            cones: [
                ConeDesc { offset: 0, dim: 3 },
                ConeDesc { offset: 3, dim: 1 },
            ],
        };

        let mut ws = SocpWorkspace::<NP, NE, NCT>::default();
        let params = IpmAlgoParams::default();
        let result = solve_socp(&prob, &params, &mut ws);

        eprintln!("2-cone: status = {}, iters = {}, x = ({:.6}, {:.6}, {:.6})",
                  result.status.as_u32(), result.iters,
                  result.x[0], result.x[1], result.x[2]);
        eprintln!("  s = ({:.4e}, {:.4e}, {:.4e}, {:.4e})",
                  result.s[0], result.s[1], result.s[2], result.s[3]);
        eprintln!("  y = ({:.4e}, {:.4e}, {:.4e}, {:.4e})",
                  result.y[0], result.y[1], result.y[2], result.y[3]);

        // x = (1, 1, 2) within 1e-4
        let expected = [1.0, 1.0, 2.0];
        for (i, &want) in expected.iter().enumerate() {
            assert!(
                (result.x[i] - want).abs() < 1.0e-4,
                "x[{i}] = {} vs {}", result.x[i], want
            );
        }
        // Cone 2 binding ⇒ s[3] small, y[3] ≈ 1
        assert!(result.s[3] < 1.0e-3,  "s[3] = {} (should bind to 0)", result.s[3]);
        assert!((result.y[3] - 1.0).abs() < 1.0e-3,
                "y[3] = {} (should be ≈ 1)", result.y[3]);
        // Cone 1 strictly interior ⇒ y_1 ≈ 0
        for i in 0..3 {
            assert!(result.y[i].abs() < 1.0e-3,
                    "y_1[{i}] = {} (cone 1 should be interior with y≈0)", result.y[i]);
        }
    }

    /// **NT-direction regression**: the same 2-cone problem from the AHO test
    /// must be solvable by `solve_socp_nt` to at least the same precision.
    /// Expected `x = (1, 1, 2)`, `s[3] → 0` (cone 2 binding), `y[3] → 1`.
    #[test]
    fn nt_solves_two_cone_problem() {
        const NP: usize     = 3;
        const NE: usize     = 2;
        const NCT: usize    = 4;
        const NCONES: usize = 2;

        let prob = SocpProblem::<NP, NE, NCT, NCONES> {
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

        let mut ws = SocpWorkspace::<NP, NE, NCT>::default();
        let params = IpmAlgoParams::default();
        let result = solve_socp_nt(&prob, &params, &mut ws);

        eprintln!("NT 2-cone: status = {}, iters = {}, x = ({:.6}, {:.6}, {:.6})",
                  result.status.as_u32(), result.iters,
                  result.x[0], result.x[1], result.x[2]);

        assert!(matches!(
            result.status,
            IpmStatus::Optimal | IpmStatus::BestFeasible
        ), "NT failed: status = {}", result.status.as_u32());

        // x = (1, 1, 2) within 1e-4
        let expected = [1.0, 1.0, 2.0];
        for (i, &want) in expected.iter().enumerate() {
            assert!(
                (result.x[i] - want).abs() < 1.0e-4,
                "x[{i}] = {} vs {}", result.x[i], want
            );
        }
        assert!(result.s[3] < 1.0e-3, "s[3] = {}", result.s[3]);
        assert!((result.y[3] - 1.0).abs() < 1.0e-3, "y[3] = {}", result.y[3]);
    }

    /// **NT-direction toy SOCP** regression — the same `min x_1 s.t. x_2+x_3=1,
    /// x ∈ SOC^3` problem from the AHO regression. NT should reach the same
    /// `x ≈ (√0.5, 0.5, 0.5)`.
    #[test]
    fn nt_toy_socp_recovers_closed_form() {
        const NP: usize     = 3;
        const NE: usize     = 1;
        const NCT: usize    = 3;
        const NCONES: usize = 1;

        let prob = SocpProblem::<NP, NE, NCT, NCONES> {
            c:     SVector::<f64, 3>::from_column_slice(&[1.0, 0.0, 0.0]),
            a_mat: SMatrix::<f64, 1, 3>::from_row_slice(&[0.0, 1.0, 1.0]),
            b:     SVector::<f64, 1>::from_element(1.0),
            g_mat: -SMatrix::<f64, 3, 3>::identity(),
            h:     SVector::<f64, 3>::zeros(),
            cones: [ConeDesc { offset: 0, dim: 3 }],
        };
        let mut ws = SocpWorkspace::<NP, NE, NCT>::default();
        let params = IpmAlgoParams::default();
        let result = solve_socp_nt(&prob, &params, &mut ws);

        eprintln!("NT toy SOCP: status = {}, iters = {}, x = ({:.6}, {:.6}, {:.6})",
                  result.status.as_u32(), result.iters,
                  result.x[0], result.x[1], result.x[2]);

        let expected_x = [sqrt(0.5), 0.5, 0.5];
        for (i, &expected) in expected_x.iter().enumerate() {
            assert!(
                (result.x[i] - expected).abs() < 1.0e-3,
                "x[{i}] = {} vs {} (status {})",
                result.x[i], expected, result.status.as_u32()
            );
        }
        assert!((result.x[1] + result.x[2] - 1.0).abs() < 1.0e-3);
    }

    /// Iter-cap defense: zero `max_iters` returns immediately with `IterCap`,
    /// no crash, no panic.
    #[test]
    fn zero_iters_returns_clean() {
        const NP: usize = 3;
        const NE: usize = 1;
        const NCT: usize = 3;
        const NCONES: usize = 1;

        let prob = SocpProblem::<NP, NE, NCT, NCONES> {
            c:     SVector::zeros(),
            a_mat: SMatrix::<f64, 1, 3>::from_row_slice(&[1.0, 1.0, 1.0]),
            b:     SVector::<f64, 1>::from_element(1.0),
            g_mat: -SMatrix::<f64, 3, 3>::identity(),
            h:     SVector::zeros(),
            cones: [ConeDesc { offset: 0, dim: 3 }],
        };
        let mut ws = SocpWorkspace::<NP, NE, NCT>::default();
        let params = IpmAlgoParams { max_iters: 0, ..IpmAlgoParams::default() };
        let result = solve_socp(&prob, &params, &mut ws);
        assert_eq!(result.iters, 0);
        // No best-feasible iterate seen ⇒ IterCap, not BestFeasible.
        assert!(result.status == IpmStatus::IterCap);
    }

    /// **HSD-direction regression** — the same 2-cone problem
    /// (`two_cone_mixed_socp` / `nt_solves_two_cone_problem`) must be solved by
    /// the homogeneous-self-dual driver `solve_socp_hsd` to the same precision.
    /// HSD cold-starts from the central point (ignores the warm-start), so this
    /// also exercises the de-homogenization (`recover = embedded/τ`). Expected
    /// `x = (1, 1, 2)`, `s[3] → 0` (cone 2 binding), `y[3] → 1`.
    #[test]
    fn hsd_solves_two_cone_problem() {
        const NP: usize     = 3;
        const NE: usize     = 2;
        const NCT: usize    = 4;
        const NCONES: usize = 2;

        let prob = SocpProblem::<NP, NE, NCT, NCONES> {
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

        let mut ws = SocpWorkspace::<NP, NE, NCT>::default();
        let params = IpmAlgoParams::default();
        let result = solve_socp_hsd(&prob, &params, &mut ws);

        eprintln!("HSD 2-cone: status = {}, iters = {}, x = ({:.6}, {:.6}, {:.6})",
                  result.status.as_u32(), result.iters,
                  result.x[0], result.x[1], result.x[2]);

        assert!(matches!(
            result.status,
            IpmStatus::Optimal | IpmStatus::BestFeasible
        ), "HSD failed: status = {}", result.status.as_u32());

        let expected = [1.0, 1.0, 2.0];
        for (i, &want) in expected.iter().enumerate() {
            assert!(
                (result.x[i] - want).abs() < 1.0e-4,
                "x[{i}] = {} vs {} (status {})", result.x[i], want, result.status.as_u32()
            );
        }
        assert!(result.s[3] < 1.0e-3, "s[3] = {}", result.s[3]);
        assert!((result.y[3] - 1.0).abs() < 1.0e-3, "y[3] = {}", result.y[3]);
    }

    /// **HSD toy SOCP** — `min x_1 s.t. x_2+x_3=1, x ∈ SOC^3`. The optimum
    /// `x = (√0.5, 0.5, 0.5)` sits EXACTLY on the cone boundary (`det(x)=0`) —
    /// the vanishing-cone scenario in miniature, where AHO degenerates and the
    /// plain NT scaling overflows. HSD must still recover it (the central-path
    /// embedding keeps the NT scaling bounded).
    #[test]
    fn hsd_toy_socp_recovers_closed_form() {
        const NP: usize     = 3;
        const NE: usize     = 1;
        const NCT: usize    = 3;
        const NCONES: usize = 1;

        let prob = SocpProblem::<NP, NE, NCT, NCONES> {
            c:     SVector::<f64, 3>::from_column_slice(&[1.0, 0.0, 0.0]),
            a_mat: SMatrix::<f64, 1, 3>::from_row_slice(&[0.0, 1.0, 1.0]),
            b:     SVector::<f64, 1>::from_element(1.0),
            g_mat: -SMatrix::<f64, 3, 3>::identity(),
            h:     SVector::<f64, 3>::zeros(),
            cones: [ConeDesc { offset: 0, dim: 3 }],
        };
        let mut ws = SocpWorkspace::<NP, NE, NCT>::default();
        let params = IpmAlgoParams::default();
        let result = solve_socp_hsd(&prob, &params, &mut ws);

        eprintln!("HSD toy SOCP: status = {}, iters = {}, x = ({:.6}, {:.6}, {:.6})",
                  result.status.as_u32(), result.iters,
                  result.x[0], result.x[1], result.x[2]);

        let expected_x = [sqrt(0.5), 0.5, 0.5];
        for (i, &expected) in expected_x.iter().enumerate() {
            assert!(
                (result.x[i] - expected).abs() < 1.0e-3,
                "x[{i}] = {} vs {} (status {})",
                result.x[i], expected, result.status.as_u32()
            );
        }
        assert!((result.x[1] + result.x[2] - 1.0).abs() < 1.0e-3);
    }
}
