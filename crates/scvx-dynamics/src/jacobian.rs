use libm::{exp, sqrt};
use nalgebra::{SMatrix, SVector};
use scvx_core::PhysicalParams;

use crate::continuous::{THRUST_SMOOTH_EPS, VELOCITY_SMOOTH_EPS};

/// Analytic `∂f/∂x` evaluated at `(x, u)` for the 3-DoF dynamics.
///
/// The structure (zero blocks shown as `·`):
/// ```text
///       ∂/∂r   ∂/∂v   ∂/∂z
///   ṙ : ·      I₃     ·
///   v̇ : ·      A_vv   a_vz
///   ż : ·      ·      a_zz
/// ```
/// Derivation:
///
/// - `A_vv = ∂v̇/∂v = −α·D·(‖v‖_ε·I + v·vᵀ/‖v‖_ε)`
/// - `a_vz = ∂v̇/∂z = g − v̇   (since ∂α/∂z = −α)`
/// - `a_zz = ∂ż/∂z = −ż`
pub fn df_dx(
    x: &SVector<f64, 7>,
    u: &SVector<f64, 3>,
    p: &PhysicalParams,
) -> SMatrix<f64, 7, 7> {
    let v       = SVector::<f64, 3>::from_column_slice(&[x[3], x[4], x[5]]);
    let z       = x[6];
    let alpha   = exp(-z);
    let v_norm2 = v[0]*v[0] + v[1]*v[1] + v[2]*v[2];
    let v_ne    = sqrt(v_norm2 + VELOCITY_SMOOTH_EPS);
    let u_ne    = sqrt(u[0]*u[0] + u[1]*u[1] + u[2]*u[2] + THRUST_SMOOTH_EPS);
    let d_coef  = 0.5 * p.rho * p.cd_a;
    let drag    = d_coef * v_ne;

    // For the ∂/∂z column we need v̇ and ż themselves.
    let mut vdot = SVector::<f64, 3>::zeros();
    for i in 0..3 {
        vdot[i] = alpha * (u[i] - drag * v[i]) + p.g[i];
    }
    let zdot = -alpha * u_ne / (p.isp * p.g0);

    let mut a = SMatrix::<f64, 7, 7>::zeros();

    // Row 0..3: ∂ṙ/∂x = [·, I₃, ·]
    a[(0, 3)] = 1.0;
    a[(1, 4)] = 1.0;
    a[(2, 5)] = 1.0;

    // Row 3..6: ∂v̇/∂x
    //   ∂v̇_i/∂v_j = −α·D·(v_ne·δ_ij + v_i v_j / v_ne)
    for i in 0..3 {
        for j in 0..3 {
            let kron = if i == j { 1.0 } else { 0.0 };
            a[(3 + i, 3 + j)] = -alpha * d_coef * (v_ne * kron + v[i] * v[j] / v_ne);
        }
        // ∂v̇_i/∂z = g_i - v̇_i
        a[(3 + i, 6)] = p.g[i] - vdot[i];
    }

    // Row 6: ∂ż/∂z = -ż.  All other entries 0.
    a[(6, 6)] = -zdot;

    a
}

/// Analytic `∂f/∂u` evaluated at `(x, u)`.
///
/// Structure:
/// ```text
///       ∂/∂u
///   ṙ : ·                       (3×3 zero)
///   v̇ : α · I₃
///   ż : −α·u / ( ‖u‖_ε · Isp·g₀ )
/// ```
pub fn df_du(
    x: &SVector<f64, 7>,
    u: &SVector<f64, 3>,
    p: &PhysicalParams,
) -> SMatrix<f64, 7, 3> {
    let z     = x[6];
    let alpha = exp(-z);
    let u_ne  = sqrt(u[0]*u[0] + u[1]*u[1] + u[2]*u[2] + THRUST_SMOOTH_EPS);

    let mut b = SMatrix::<f64, 7, 3>::zeros();

    // ∂v̇/∂u = α·I₃
    b[(3, 0)] = alpha;
    b[(4, 1)] = alpha;
    b[(5, 2)] = alpha;

    // ∂ż/∂u_j = −α · u_j / (‖u‖_ε · Isp · g₀)
    let factor = -alpha / (u_ne * p.isp * p.g0);
    b[(6, 0)] = factor * u[0];
    b[(6, 1)] = factor * u[1];
    b[(6, 2)] = factor * u[2];

    b
}

// ===========================================================================
// Tests — central-difference validation
// ===========================================================================

#[cfg(test)]
mod tests {
    extern crate std;
    use std::eprintln;

    use super::*;
    use crate::continuous::f_continuous;

    fn test_params() -> PhysicalParams {
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
            rho:              0.020,
            cd_a:             1.0,
            tau_lo:           5.0,
            tau_hi:          50.0,
        }
    }

    /// Central-difference Jacobian: `(f(x+he) - f(x-he)) / (2h)`.
    /// `h` is per-component, scaled so we hit f64 ULPs ≈ ε^(1/3) · |x_j|
    /// (the optimal step for central differences).
    fn fd_df_dx(
        x: &SVector<f64, 7>,
        u: &SVector<f64, 3>,
        p: &PhysicalParams,
    ) -> SMatrix<f64, 7, 7> {
        let mut j = SMatrix::<f64, 7, 7>::zeros();
        for col in 0..7 {
            let scale = x[col].abs().max(1.0);
            let h = scale * 1.0e-6;
            let mut xp = *x;
            let mut xm = *x;
            xp[col] += h;
            xm[col] -= h;
            let fp = f_continuous(&xp, u, p);
            let fm = f_continuous(&xm, u, p);
            for row in 0..7 {
                j[(row, col)] = (fp[row] - fm[row]) / (2.0 * h);
            }
        }
        j
    }

    fn fd_df_du(
        x: &SVector<f64, 7>,
        u: &SVector<f64, 3>,
        p: &PhysicalParams,
    ) -> SMatrix<f64, 7, 3> {
        let mut j = SMatrix::<f64, 7, 3>::zeros();
        for col in 0..3 {
            let scale = u[col].abs().max(1.0);
            let h = scale * 1.0e-6;
            let mut up = *u;
            let mut um = *u;
            up[col] += h;
            um[col] -= h;
            let fp = f_continuous(x, &up, p);
            let fm = f_continuous(x, &um, p);
            for row in 0..7 {
                j[(row, col)] = (fp[row] - fm[row]) / (2.0 * h);
            }
        }
        j
    }

    /// Deterministic LCG so the random-points test is reproducible
    /// (Rust 1.94 stable, no external rand dep). Knuth's MMIX constants.
    struct Lcg(u64);
    impl Lcg {
        fn new(seed: u64) -> Self { Lcg(seed) }
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0
        }
        /// Uniform `[lo, hi)`.
        fn range(&mut self, lo: f64, hi: f64) -> f64 {
            let raw = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
            lo + raw * (hi - lo)
        }
    }

    /// Compare two matrices element-wise.  `rel_tol` is the relative tolerance;
    /// `abs_tol` kicks in when the analytic entry is near zero so we don't
    /// divide-by-tiny.
    fn matrices_close<const R: usize, const C: usize>(
        a: &SMatrix<f64, R, C>,
        b: &SMatrix<f64, R, C>,
        rel_tol: f64,
        abs_tol: f64,
    ) -> Option<(usize, usize, f64, f64)> {
        for i in 0..R {
            for j in 0..C {
                let diff = (a[(i, j)] - b[(i, j)]).abs();
                let scale = a[(i, j)].abs().max(b[(i, j)].abs()).max(1.0);
                if diff > abs_tol && diff / scale > rel_tol {
                    return Some((i, j, a[(i, j)], b[(i, j)]));
                }
            }
        }
        None
    }

    #[test]
    fn analytic_df_dx_matches_central_difference_at_origin() {
        let p = test_params();
        let mut x = SVector::<f64, 7>::zeros();
        x[6] = (500.0_f64).ln(); // m = 500
        let u = SVector::<f64, 3>::from_column_slice(&[0.0, 0.0, 2000.0]);

        let an = df_dx(&x, &u, &p);
        let fd = fd_df_dx(&x, &u, &p);
        if let Some((i, j, av, fv)) = matrices_close(&an, &fd, 1.0e-5, 1.0e-7) {
            panic!("df_dx mismatch at ({i},{j}): analytic = {av}, FD = {fv}");
        }
    }

    #[test]
    fn analytic_df_du_matches_central_difference_at_origin() {
        let p = test_params();
        let mut x = SVector::<f64, 7>::zeros();
        x[6] = (500.0_f64).ln();
        let u = SVector::<f64, 3>::from_column_slice(&[0.0, 0.0, 2000.0]);

        let an = df_du(&x, &u, &p);
        let fd = fd_df_du(&x, &u, &p);
        if let Some((i, j, av, fv)) = matrices_close(&an, &fd, 1.0e-5, 1.0e-7) {
            panic!("df_du mismatch at ({i},{j}): analytic = {av}, FD = {fv}");
        }
    }

    #[test]
    fn random_points_df_dx_matches_fd() {
        let p = test_params();
        let mut rng = Lcg::new(0x00C0_FFEE_C0DE_5CBC);
        let n_trials = 200;
        let mut max_rel_err = 0.0_f64;
        for trial in 0..n_trials {
            let mut x = SVector::<f64, 7>::zeros();
            x[0] = rng.range(-1000.0, 1000.0);
            x[1] = rng.range(-1000.0, 1000.0);
            x[2] = rng.range(    0.0, 2000.0); // altitude > 0
            x[3] = rng.range( -100.0,  100.0);
            x[4] = rng.range( -100.0,  100.0);
            x[5] = rng.range(  -50.0,   10.0); // mostly descending
            x[6] = rng.range( (200.0_f64).ln(), (1000.0_f64).ln()); // m in [200, 1000]
            let u = SVector::<f64, 3>::from_column_slice(&[
                rng.range(-3000.0, 3000.0),
                rng.range(-3000.0, 3000.0),
                rng.range( 1000.0, 6000.0),
            ]);

            let an = df_dx(&x, &u, &p);
            let fd = fd_df_dx(&x, &u, &p);
            for i in 0..7 {
                for j in 0..7 {
                    let diff = (an[(i, j)] - fd[(i, j)]).abs();
                    let scale = an[(i, j)].abs().max(fd[(i, j)].abs()).max(1.0);
                    let rel = diff / scale;
                    if rel > max_rel_err { max_rel_err = rel; }
                    if rel > 1.0e-4 && diff > 1.0e-6 {
                        eprintln!(
                            "trial {trial} ({i},{j}): an={}, fd={}, rel={}",
                            an[(i,j)], fd[(i,j)], rel
                        );
                        panic!("df_dx Jacobian mismatch");
                    }
                }
            }
        }
        eprintln!("df_dx max relative error over {n_trials} trials: {:.2e}", max_rel_err);
    }

    #[test]
    fn random_points_df_du_matches_fd() {
        let p = test_params();
        let mut rng = Lcg::new(0xDEAD_BEEF_CAFE);
        let n_trials = 200;
        let mut max_rel_err = 0.0_f64;
        for trial in 0..n_trials {
            let mut x = SVector::<f64, 7>::zeros();
            x[6] = rng.range((200.0_f64).ln(), (1000.0_f64).ln());
            x[3] = rng.range( -50.0,  50.0);
            x[4] = rng.range( -50.0,  50.0);
            x[5] = rng.range( -50.0,  50.0);
            let u = SVector::<f64, 3>::from_column_slice(&[
                rng.range(-3000.0, 3000.0),
                rng.range(-3000.0, 3000.0),
                rng.range( 1000.0, 6000.0),
            ]);

            let an = df_du(&x, &u, &p);
            let fd = fd_df_du(&x, &u, &p);
            for i in 0..7 {
                for j in 0..3 {
                    let diff = (an[(i, j)] - fd[(i, j)]).abs();
                    let scale = an[(i, j)].abs().max(fd[(i, j)].abs()).max(1.0);
                    let rel = diff / scale;
                    if rel > max_rel_err { max_rel_err = rel; }
                    if rel > 1.0e-4 && diff > 1.0e-6 {
                        eprintln!(
                            "trial {trial} ({i},{j}): an={}, fd={}, rel={}",
                            an[(i,j)], fd[(i,j)], rel
                        );
                        panic!("df_du Jacobian mismatch");
                    }
                }
            }
        }
        eprintln!("df_du max relative error over {n_trials} trials: {:.2e}", max_rel_err);
    }

    #[test]
    fn jacobian_smooth_near_zero_velocity() {
        // The drag-smoothing trick: at v=0, ‖v‖·v/‖v‖_ε terms degenerate to 0
        // rather than blowing up. Verify by perturbing v slightly and showing
        // the Jacobian doesn't change wildly.
        let p = test_params();
        let mut x = SVector::<f64, 7>::zeros();
        x[6] = (500.0_f64).ln();
        let u = SVector::<f64, 3>::from_column_slice(&[0.0, 0.0, 2000.0]);

        let j0 = df_dx(&x, &u, &p);
        x[3] = 1.0e-4;
        let j1 = df_dx(&x, &u, &p);
        x[3] = 1.0e-3;
        let j2 = df_dx(&x, &u, &p);

        // Jacobian entries should change smoothly across these perturbations
        // (small ‖v‖ ⇒ all v̇/∂v entries dominated by `−α·D·√ε`).
        for i in 0..7 {
            for j in 0..7 {
                assert!(
                    (j0[(i, j)] - j2[(i, j)]).abs() < 1.0e-2,
                    "non-smooth at ({i},{j}): {} vs {}",
                    j0[(i, j)], j2[(i, j)]
                );
                let _ = j1; // (unused in assertion; documents the intermediate)
            }
        }
    }
}
