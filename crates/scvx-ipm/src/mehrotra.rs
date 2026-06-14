//! Mehrotra predictor-corrector primal-dual IPM driver.
//!
//! Direction: **AHO** (Alizadeh-Haeberly-Overton 1998) — linearizes the
//! complementarity equation directly via `arrow(x)·Δμ + arrow(μ)·Δx`. This
//! is asymmetric in `(Δx, Δμ)` and converges more slowly per iteration than
//! Nesterov-Todd, but the implementation has no sign-convention landmines
//! and is unambiguously correct. NT scaling is a P1b follow-up (see
//! [`crate::cone::soc_nt_scaling_matrix`]).
//!
//! Scope: toy SOCP in standard form
//! ```text
//!   min  cᵀ x
//!   s.t. A x = b
//!        x ∈ K     (single SOC of dim 3 in this dim-locked variant)
//! ```
//! Generalization to multi-cone, multi-equality SOCP rides on Riccati KKT
//! in a later phase — this driver exists to validate the cone primitives
//! and the IPM loop structure on a problem with a closed-form optimum.

use libm::sqrt;
use nalgebra::{SMatrix, SVector};

use scvx_core::{IpmAlgoParams, IpmStatus};

use crate::cone::{soc_arrow_matrix, soc_in_interior, soc_max_step};

/// Result of a toy IPM solve.
#[derive(Clone, Copy)]
pub struct ToyIpmResult {
    pub x:      SVector<f64, 3>,
    pub lambda: f64,
    pub mu:     SVector<f64, 3>,
    pub status: IpmStatus,
    pub iters:  u32,
}

/// Solve the toy SOCP  `min cᵀ x   s.t.  A x = b,  x ∈ SOC³`.
///
/// `A` is the 1×3 equality matrix, `b` the scalar RHS. Returns the optimal
/// primal `x`, dual scalar `λ`, cone dual `μ`, IPM status, and iteration
/// count. Hard cap from `params.max_iters`; on cap-hit reports
/// `IpmStatus::IterCap`.
pub fn solve_toy_socp(
    c:      &SVector<f64, 3>,
    a_row:  &SMatrix<f64, 1, 3>,
    b:      f64,
    params: &IpmAlgoParams,
) -> ToyIpmResult {
    // ---- Initialization: strictly interior point ----
    //
    // `(κ, 0, 0)` lies strictly inside SOC³ for any κ > 0. We pick κ large
    // enough that primal/dual residuals start small relative to the cone
    // interior margin. A simple constant works because the problem is small
    // and well-scaled; a real IPM would do a Mehrotra-style starting-point
    // heuristic (solve two least-squares problems) — overkill here.
    let mut x = SVector::<f64, 3>::zeros();
    x[0] = 1.0;
    let mut lambda: f64 = 0.0;
    let mut mu_dual = SVector::<f64, 3>::zeros();
    mu_dual[0] = 1.0;

    let dim_cone = 3usize;

    // ---- Best-feasible-iterate tracking ----
    //
    // AHO direction's Achilles heel: at the SOCP optimum both `x` and `μ`
    // sit on the cone boundary, so `arrow(x)` and `arrow(μ)` go singular
    // and the Newton system blows up if we iterate past convergence.
    // We track the smallest-`μ_avg` iterate that is "feasible enough"
    // (residuals below the loose tolerance) and return it on any
    // non-Optimal terminus — exactly the plan's "BestFeasible fallback".
    let mut best = BestIterate {
        x,
        lambda,
        mu: mu_dual,
        mavg: f64::INFINITY,
        valid: false,
    };

    // Looser screen for "feasible enough" — used only to gate best-iterate
    // updates and the no-progress early-exit. Strict-tolerance convergence
    // still requires the user-supplied tolerances.
    let loose_primal = sqrt(params.tol_primal).max(1.0e-5);
    let loose_dual   = sqrt(params.tol_dual).max(1.0e-5);
    let loose_mu     = sqrt(params.tol_mu).max(1.0e-5);

    let mut prev_x  = x;
    let mut prev_mu = mu_dual;

    for iter in 0..params.max_iters {
        // ---- Residuals ----
        //
        // Stationarity:    r_d = c − Aᵀλ − μ
        // Primal feas.:    r_p = A·x − b
        // Complementarity: r_c is built per predictor/corrector below.
        let aty: SVector<f64, 3> = a_row.transpose() * lambda;
        let r_d: SVector<f64, 3> = c - aty - mu_dual;
        let r_p_vec: SVector<f64, 1> = a_row * x - SVector::<f64, 1>::from_element(b);
        let r_p: f64 = r_p_vec[0];

        let dual_meas = x.dot(&mu_dual) / dim_cone as f64; // μ_avg

        // ---- Best-feasible-iterate update ----
        if r_d.norm() < loose_dual
            && r_p.abs()  < loose_primal
            && dual_meas.is_finite()
            && dual_meas >= 0.0
            && dual_meas < best.mavg
        {
            best = BestIterate {
                x,
                lambda,
                mu: mu_dual,
                mavg: dual_meas,
                valid: true,
            };
        }

        // ---- Strict convergence test (Optimal) ----
        if r_d.norm() < params.tol_dual
            && r_p.abs() < params.tol_primal
            && dual_meas.abs() < params.tol_mu
        {
            return ToyIpmResult {
                x,
                lambda,
                mu: mu_dual,
                status: IpmStatus::Optimal,
                iters: iter,
            };
        }

        // ---- No-progress early exit ----
        //
        // If the iterates have stopped moving and we already have a "good
        // enough" best, declare success at the best iterate. This catches
        // the AHO-stalled-at-boundary case where the optimum was found but
        // tight tolerance was never reached due to arrow-matrix singularity.
        let dx_iter  = (x  - prev_x ).norm();
        let dmu_iter = (mu_dual - prev_mu).norm();
        if iter > 0
            && dx_iter < 1.0e-9
            && dmu_iter < 1.0e-9
            && best.valid
            && best.mavg < loose_mu
        {
            return ToyIpmResult {
                x: best.x,
                lambda: best.lambda,
                mu: best.mu,
                status: IpmStatus::BestFeasible,
                iters: iter,
            };
        }
        prev_x  = x;
        prev_mu = mu_dual;

        // ---- Linearized-KKT matrix factors ----
        let arr_x  = soc_arrow_matrix(&x);
        let arr_mu = soc_arrow_matrix(&mu_dual);

        // arr_x must be invertible — true when x ∈ int(K) (we maintain that).
        let arr_x_inv = match arr_x.try_inverse() {
            Some(m) => m,
            None => {
                return numerical_exit(&best, x, lambda, mu_dual, iter);
            }
        };

        // H = arrow(x)⁻¹·arrow(μ).  3×3, asymmetric in general.
        let h: SMatrix<f64, 3, 3> = arr_x_inv * arr_mu;

        // Schur-complement scalar  S = A H⁻¹ Aᵀ.  Compute H⁻¹ once.
        let h_inv = match h.try_inverse() {
            Some(m) => m,
            None => {
                return numerical_exit(&best, x, lambda, mu_dual, iter);
            }
        };

        let aha: SMatrix<f64, 1, 1> = a_row * h_inv * a_row.transpose();
        let schur: f64 = aha[(0, 0)];
        if !schur.is_finite() || schur.abs() < 1e-18 {
            return numerical_exit(&best, x, lambda, mu_dual, iter);
        }

        // ============================================================
        // Predictor (affine) step  —  target zero centering, σ = 0
        // ============================================================
        //
        // r_c_aff = -x ∘ μ      ( = -arrow(x)·μ )
        let r_c_aff: SVector<f64, 3> = -(arr_x * mu_dual);

        // q_aff = -r_d + arrow(x)⁻¹ · r_c_aff
        let q_aff: SVector<f64, 3> = -r_d + arr_x_inv * r_c_aff;

        // Δλ_aff = (1/S)·( -r_p - A H⁻¹ q_aff )
        let a_hinv_q: SMatrix<f64, 1, 1> = a_row * h_inv * q_aff;
        let dlam_aff: f64 = (-r_p - a_hinv_q[(0, 0)]) / schur;

        // Δx_aff = H⁻¹ ( q_aff + Aᵀ Δλ_aff )
        let dx_aff: SVector<f64, 3> = h_inv * (q_aff + a_row.transpose() * dlam_aff);

        // Δμ_aff = arrow(x)⁻¹ ( r_c_aff − arrow(μ) Δx_aff )
        let dmu_aff: SVector<f64, 3> = arr_x_inv * (r_c_aff - arr_mu * dx_aff);

        // Affine step lengths
        let alpha_p_aff = clip01(0.99 * soc_max_step(x.as_slice(),  dx_aff.as_slice()));
        let alpha_d_aff = clip01(0.99 * soc_max_step(mu_dual.as_slice(), dmu_aff.as_slice()));

        // Affine duality measure  μ_aff = (x + α_p Δx)ᵀ(μ + α_d Δμ) / D
        let x_aff_dot_mu_aff = (x + alpha_p_aff * dx_aff).dot(&(mu_dual + alpha_d_aff * dmu_aff));
        let mu_aff = x_aff_dot_mu_aff / dim_cone as f64;

        // Centering parameter: classic Mehrotra cubic.
        let sigma = if dual_meas > 0.0 {
            let r = mu_aff / dual_meas;
            r * r * r
        } else {
            0.0
        };

        // ============================================================
        // Corrector step
        // ============================================================
        //
        // r_c_corr = σ μ_avg e − x ∘ μ − Δx_aff ∘ Δμ_aff
        let mut r_c_corr = -(arr_x * mu_dual);
        r_c_corr[0] += sigma * dual_meas;

        // Subtract Δx_aff ∘ Δμ_aff  (using arrow form: arrow(Δx_aff) · Δμ_aff)
        let arr_dx_aff = soc_arrow_matrix(&dx_aff);
        r_c_corr -= arr_dx_aff * dmu_aff;

        let q_corr: SVector<f64, 3> = -r_d + arr_x_inv * r_c_corr;

        let a_hinv_q_corr: SMatrix<f64, 1, 1> = a_row * h_inv * q_corr;
        let dlam: f64 = (-r_p - a_hinv_q_corr[(0, 0)]) / schur;

        let dx: SVector<f64, 3> = h_inv * (q_corr + a_row.transpose() * dlam);
        let dmu: SVector<f64, 3> = arr_x_inv * (r_c_corr - arr_mu * dx);

        // Step lengths for the actual update
        let alpha_p = clip01(0.99 * soc_max_step(x.as_slice(),       dx.as_slice()));
        let alpha_d = clip01(0.99 * soc_max_step(mu_dual.as_slice(), dmu.as_slice()));

        // Update
        x       += alpha_p * dx;
        lambda  += alpha_d * dlam;
        mu_dual += alpha_d * dmu;

        // Guard: ensure we stay in interior (numerical drift can push us
        // onto / over the boundary). If it happens, fall back to
        // best-feasible if we have one.
        if !soc_in_interior(x.as_slice()) || !soc_in_interior(mu_dual.as_slice()) {
            return numerical_exit(&best, x, lambda, mu_dual, iter + 1);
        }
    }

    // Hard iter cap. Prefer best-feasible-iterate when available.
    if best.valid {
        ToyIpmResult {
            x: best.x,
            lambda: best.lambda,
            mu: best.mu,
            status: IpmStatus::BestFeasible,
            iters: params.max_iters,
        }
    } else {
        ToyIpmResult {
            x,
            lambda,
            mu: mu_dual,
            status: IpmStatus::IterCap,
            iters: params.max_iters,
        }
    }
}

/// Snapshot of the "best feasible" iterate found so far. Bundled so the
/// numerical-exit / iter-cap return paths take a single reference instead of
/// six positional arguments (clippy `too_many_arguments`) — and so future
/// fields can be added without churning every call site.
#[derive(Clone, Copy)]
struct BestIterate {
    x:      SVector<f64, 3>,
    lambda: f64,
    mu:     SVector<f64, 3>,
    /// Duality measure `x·μ / D` at the captured iterate. `+∞` until the
    /// first feasible iterate is seen.
    mavg:   f64,
    /// `true` once at least one iterate has passed the loose-feasibility
    /// screen. We never return `BestFeasible` unless this is set.
    valid:  bool,
}

/// Common exit path on numerical breakdown: return best-feasible if we
/// have one, else flag the failure with the current iterate.
#[inline]
fn numerical_exit(
    best:    &BestIterate,
    x:       SVector<f64, 3>,
    lambda:  f64,
    mu_dual: SVector<f64, 3>,
    iter:    u32,
) -> ToyIpmResult {
    if best.valid {
        ToyIpmResult {
            x:      best.x,
            lambda: best.lambda,
            mu:     best.mu,
            status: IpmStatus::BestFeasible,
            iters:  iter,
        }
    } else {
        ToyIpmResult {
            x,
            lambda,
            mu:     mu_dual,
            status: IpmStatus::NumericalError,
            iters:  iter,
        }
    }
}

/// Clip step length `α` to `[0, 1]` with explicit NaN / ±∞ semantics.
///
/// - `NaN`        ⇒ `0.0`  (reject the step; never propagate poison).
/// - `+∞`         ⇒ `1.0`  (no cone binding ⇒ take full Newton).
/// - `−∞` or `< 0` ⇒ `0.0`.
/// - `> 1`        ⇒ `1.0`.
/// - finite `[0,1]` ⇒ `a`.
///
/// Note: `f64::clamp` would *propagate* NaN here, which would taint the
/// iterate. Clippy's `manual_clamp` lint suggests `clamp` — that suggestion
/// is wrong for step lengths and is explicitly suppressed.
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

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    extern crate std;
    use std::eprintln;

    use super::*;

    /// Debug helper: dump iteration trajectory of the toy solve so we can
    /// see where conditioning goes wrong. Std-only (test context).
    #[allow(dead_code)]
    fn dbg_trajectory(
        c: &SVector<f64, 3>,
        a_row: &SMatrix<f64, 1, 3>,
        b: f64,
        max_iters: u32,
    ) {
        use crate::cone::{soc_arrow_matrix, soc_det};
        let mut x = SVector::<f64, 3>::zeros();
        x[0] = 1.0;
        let mut lambda: f64 = 0.0;
        let mut mu = SVector::<f64, 3>::zeros();
        mu[0] = 1.0;
        eprintln!("iter |    x[0]      x[1]      x[2]   | det(x)    det(μ) | μ_avg   | r_p    | ‖r_d‖");
        for k in 0..max_iters {
            let r_d  = c - a_row.transpose() * lambda - mu;
            let r_p  = (a_row * x)[0] - b;
            let mavg = x.dot(&mu) / 3.0;
            eprintln!(
                "{k:3}  | {:8.4}  {:8.4}  {:8.4} | {:.2e}  {:.2e} | {:.2e} | {:.2e} | {:.2e}",
                x[0], x[1], x[2],
                soc_det(x.as_slice()),
                soc_det(mu.as_slice()),
                mavg, r_p, r_d.norm()
            );
            // One step of the same code as solve_toy_socp (kept terse)
            let arr_x  = soc_arrow_matrix(&x);
            let arr_mu = soc_arrow_matrix(&mu);
            let Some(arr_x_inv) = arr_x.try_inverse() else {
                eprintln!("  ! arr_x singular at iter {k}"); return;
            };
            let h = arr_x_inv * arr_mu;
            let Some(h_inv) = h.try_inverse() else {
                eprintln!("  ! H singular at iter {k}"); return;
            };
            let schur = (a_row * h_inv * a_row.transpose())[(0, 0)];
            let r_c_aff = -(arr_x * mu);
            let q_aff = -r_d + arr_x_inv * r_c_aff;
            let dlam_aff = (-r_p - (a_row * h_inv * q_aff)[(0,0)]) / schur;
            let dx_aff  = h_inv * (q_aff + a_row.transpose() * dlam_aff);
            let dmu_aff = arr_x_inv * (r_c_aff - arr_mu * dx_aff);
            let ap = clip01(0.99 * soc_max_step(x.as_slice(),  dx_aff.as_slice()));
            let ad = clip01(0.99 * soc_max_step(mu.as_slice(), dmu_aff.as_slice()));
            let mu_aff = (x + ap*dx_aff).dot(&(mu + ad*dmu_aff)) / 3.0;
            let sigma = if mavg > 0.0 { let r = mu_aff/mavg; r*r*r } else { 0.0 };
            let mut r_c_corr = -(arr_x * mu);
            r_c_corr[0] += sigma * mavg;
            r_c_corr -= soc_arrow_matrix(&dx_aff) * dmu_aff;
            let q_corr = -r_d + arr_x_inv * r_c_corr;
            let dlam = (-r_p - (a_row * h_inv * q_corr)[(0,0)]) / schur;
            let dx = h_inv * (q_corr + a_row.transpose() * dlam);
            let dmu = arr_x_inv * (r_c_corr - arr_mu * dx);
            let ap = clip01(0.99 * soc_max_step(x.as_slice(),  dx.as_slice()));
            let ad = clip01(0.99 * soc_max_step(mu.as_slice(), dmu.as_slice()));
            x      += ap * dx;
            lambda += ad * dlam;
            mu     += ad * dmu;
        }
    }

    /// Canonical toy: `min t s.t. x₁ + x₂ = 1, (t, x₁, x₂) ∈ SOC³`.
    ///
    /// Closed-form optimum: `x* = (1/√2, 1/2, 1/2)`, `λ* = 1/√2`,
    /// `μ* = (1, −1/√2, −1/√2)`, cost `= 1/√2 ≈ 0.7071`.
    #[test]
    fn toy_socp_recovers_closed_form() {
        use core::f64::consts::SQRT_2;

        let c     = SVector::<f64, 3>::from_column_slice(&[1.0, 0.0, 0.0]);
        let a_row = SMatrix::<f64, 1, 3>::from_row_slice(&[0.0, 1.0, 1.0]);
        let b     = 1.0;

        // Uncomment for trajectory dump:
        // dbg_trajectory(&c, &a_row, b, 25);

        let params = IpmAlgoParams::default();
        let res    = solve_toy_socp(&c, &a_row, b, &params);

        assert!(
            res.status == IpmStatus::Optimal || res.status == IpmStatus::BestFeasible,
            "status code = {} at iter {}", res.status.as_u32(), res.iters
        );

        let inv_sqrt2 = 1.0 / SQRT_2;
        let cost = c.dot(&res.x);

        // Precision: AHO direction stalls at ~1e-5 because at the SOCP
        // optimum both x and μ sit on the cone boundary, so arrow(x) and
        // arrow(μ) go singular. Best-feasible-iterate captures the last
        // well-conditioned iterate (typically ~iter 3) which is accurate
        // to ~1e-5. Tighter convergence requires NT scaling (P1b).
        assert!((cost - inv_sqrt2).abs() < 5.0e-5,
                "cost {} vs 1/√2 = {}, status code {}", cost, inv_sqrt2, res.status.as_u32());
        assert!((res.x[0] - inv_sqrt2).abs() < 5.0e-4, "x₀ = {}", res.x[0]);
        assert!((res.x[1] - 0.5).abs()       < 5.0e-4, "x₁ = {}", res.x[1]);
        assert!((res.x[2] - 0.5).abs()       < 5.0e-4, "x₂ = {}", res.x[2]);
        assert!((res.lambda - inv_sqrt2).abs() < 5.0e-4, "λ = {}", res.lambda);
    }

    /// A second toy with a tilted feasible set:
    /// `min 2 x₁ + x₂ s.t. (t, x₁, x₂) ∈ SOC³, t = 1`.
    /// At optimum, x lies on cone boundary in the direction that minimizes
    /// 2 x₁ + x₂ subject to x₁² + x₂² = 1.
    /// Solving: x₁ = −2/√5, x₂ = −1/√5; cost = (2)(−2/√5) + (−1/√5) = −5/√5 = −√5.
    #[test]
    fn toy_socp_with_fixed_t() {
        let c     = SVector::<f64, 3>::from_column_slice(&[0.0, 2.0, 1.0]);
        let a_row = SMatrix::<f64, 1, 3>::from_row_slice(&[1.0, 0.0, 0.0]);
        let b     = 1.0;

        let params = IpmAlgoParams::default();
        let res    = solve_toy_socp(&c, &a_row, b, &params);

        assert!(
            res.status == IpmStatus::Optimal || res.status == IpmStatus::BestFeasible,
            "status code = {} at iter {}", res.status.as_u32(), res.iters
        );

        let cost     = c.dot(&res.x);
        let expected = -(5.0_f64).sqrt();
        assert!((cost - expected).abs() < 1e-5,
                "cost {} vs −√5 = {}", cost, expected);
    }

    #[test]
    fn clip01_nan_returns_zero() {
        // Defense against NaN propagation: a NaN step length MUST translate
        // to "reject the step" (α = 0), never to "full Newton" (α = 1).
        // Regression for the safety-audit finding that
        // `!a.is_finite()` lumped NaN with +∞ and returned 1.0.
        assert_eq!(clip01(f64::NAN), 0.0);
        assert_eq!(clip01(f64::NEG_INFINITY), 0.0);
        assert_eq!(clip01(f64::INFINITY),     1.0); // +∞ = "no cone binding"
        assert_eq!(clip01(-1.0),              0.0);
        assert_eq!(clip01(0.5),               0.5);
        assert_eq!(clip01(1.5),               1.0);
    }

    #[test]
    fn iter_count_bounded() {
        // Mehrotra on a well-scaled SOCP should finish in well under the cap.
        let c     = SVector::<f64, 3>::from_column_slice(&[1.0, 0.0, 0.0]);
        let a_row = SMatrix::<f64, 1, 3>::from_row_slice(&[0.0, 1.0, 1.0]);
        let b     = 1.0;

        let params = IpmAlgoParams::default();
        let res    = solve_toy_socp(&c, &a_row, b, &params);

        assert!(res.iters < params.max_iters,
                "Mehrotra should not hit cap on toy; took {} of {}",
                res.iters, params.max_iters);
        // Empirically AHO direction does this in ~15-20 iters; assert < 30.
        assert!(res.iters < 30, "took {} iters", res.iters);
    }
}
