//! Nonlinear forward integration of the 3-DoF dynamics.
//!
//! Unlike [`crate::discretize::discretize_foh`], this module does NOT
//! propagate sensitivities — it just produces the state trajectory you'd
//! get by FOH-integrating the *nonlinear* `f_continuous` from `x_init`
//! along a given control schedule. Used by the SCvx outer loop's true
//! LM ρ-ratio computation to measure linearization quality (the gap
//! between linearized and nonlinear propagation gives the actual cost).
//!
//! FOH on control, RK4 sub-steps, time-dilated by `τ` — same conventions
//! as `discretize_foh` so that "linearized at reference" and "nonlinear"
//! coincide at the reference (the discretization's `c_k` was built that
//! way; this function is the nonlinear counterpart we compare against).
//!
//! No allocation, no panic, const-generic on `N`. Safe in no_std.

use nalgebra::{SMatrix, SVector};
use scvx_core::PhysicalParams;

use crate::continuous::f_continuous;

/// Propagate the nonlinear 3-DoF dynamics forward from `x_init` along the
/// control schedule `u_schedule`. FOH between adjacent controls, RK4 sub-
/// steps within each node-to-node interval.
///
/// Output convention: `x_out[:, 0] = x_init` exactly; `x_out[:, k+1]` is the
/// nonlinear propagation of `x_out[:, k]` under FOH(`u_k`, `u_{k+1}`) over
/// one normalized interval `[k·dt, (k+1)·dt]` scaled by `τ`, where
/// `dt = 1/(N-1)`.
///
/// This is the "ground truth" trajectory the linearization should
/// approximate. The gap between this and the linearized-predicted
/// trajectory is the *defect* the SCvx outer loop's ρ ratio measures.
pub fn nonlinear_propagate<const N: usize>(
    x_init:       &SVector<f64, 7>,
    u_schedule:   &SMatrix<f64, 3, N>,
    tau:          f64,
    params:       &PhysicalParams,
    rk4_substeps: u32,
    x_out:        &mut SMatrix<f64, 7, N>,
) {
    if N == 0 {
        return;
    }

    // Seed x_out[:, 0] = x_init.
    for i in 0..7 {
        x_out[(i, 0)] = x_init[i];
    }
    if N < 2 || rk4_substeps == 0 {
        return;
    }

    let dt_norm = 1.0 / ((N - 1) as f64);
    let h_sub   = dt_norm / (rk4_substeps as f64);

    for k in 0..(N - 1) {
        // Pull endpoint controls for this FOH interval.
        let u_k = SVector::<f64, 3>::from_column_slice(&[
            u_schedule[(0, k)],
            u_schedule[(1, k)],
            u_schedule[(2, k)],
        ]);
        let u_kp1 = SVector::<f64, 3>::from_column_slice(&[
            u_schedule[(0, k + 1)],
            u_schedule[(1, k + 1)],
            u_schedule[(2, k + 1)],
        ]);

        // Pull current state x_k.
        let mut x = SVector::<f64, 7>::from_column_slice(&[
            x_out[(0, k)], x_out[(1, k)], x_out[(2, k)],
            x_out[(3, k)], x_out[(4, k)], x_out[(5, k)],
            x_out[(6, k)],
        ]);

        // RK4 sub-step the interval.
        let mut s = 0.0_f64;
        for _ in 0..rk4_substeps {
            let lambda_at = |t: f64| t / dt_norm;
            let f_at = |t: f64, x_in: &SVector<f64, 7>| -> SVector<f64, 7> {
                let lam = lambda_at(t);
                let u   = u_k * (1.0 - lam) + u_kp1 * lam;
                f_continuous(x_in, &u, params) * tau
            };
            let k1 = f_at(s, &x);
            let x2 = x + k1 * (0.5 * h_sub);
            let k2 = f_at(s + 0.5 * h_sub, &x2);
            let x3 = x + k2 * (0.5 * h_sub);
            let k3 = f_at(s + 0.5 * h_sub, &x3);
            let x4 = x + k3 * h_sub;
            let k4 = f_at(s + h_sub, &x4);
            x += (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (h_sub / 6.0);
            s += h_sub;
        }

        // Write x_out[:, k+1].
        for i in 0..7 {
            x_out[(i, k + 1)] = x[i];
        }
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
    use crate::discretize::{discretize_foh, LinearizedDynamics};
    use scvx_core::Trajectory;

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
            rho:              0.020,
            cd_a:             1.0,
            tau_lo:           5.0,
            tau_hi:          50.0,
        }
    }

    /// Sanity: `nonlinear_propagate` writes `x_init` at index 0 and produces
    /// a finite trajectory.
    #[test]
    fn nonlinear_propagate_seeds_x_init_and_stays_finite() {
        const N: usize = 5;
        let p = mars_params();
        let mut x_init = SVector::<f64, 7>::zeros();
        x_init[2] = 500.0;
        x_init[5] = -20.0;
        x_init[6] = (800.0_f64).ln();

        let mut u = SMatrix::<f64, 3, N>::zeros();
        for k in 0..N {
            u[(2, k)] = 800.0 * 3.7114; // hover thrust
        }

        let mut x_out = SMatrix::<f64, 7, N>::zeros();
        nonlinear_propagate::<N>(&x_init, &u, 20.0, &p, 4, &mut x_out);

        // x_out[:, 0] must equal x_init exactly.
        for i in 0..7 {
            assert_eq!(x_out[(i, 0)], x_init[i], "x_out[{i},0]");
        }
        // Every other column must be finite.
        for k in 1..N {
            for i in 0..7 {
                assert!(x_out[(i, k)].is_finite(),
                        "x_out[{i},{k}] = {}", x_out[(i, k)]);
            }
        }
    }

    /// Closed-form check: free-fall under constant gravity (no drag, no
    /// thrust) is exactly kinematic. After `t` seconds from `(r_z, v_z)`:
    ///   r_z(t) = r_z₀ + v_z₀·t + ½·g_z·t²
    ///   v_z(t) = v_z₀ + g_z·t
    /// `nonlinear_propagate` should reproduce this to RK4 accuracy.
    #[test]
    fn free_fall_matches_kinematic_closed_form() {
        const N: usize = 4;
        // No drag, no thrust — pure kinematic ballistic.
        let mut p = mars_params();
        p.rho  = 0.0;
        p.cd_a = 0.0;
        let g_z = p.g[2];

        let mut x_init = SVector::<f64, 7>::zeros();
        let r_z0 = 100.0;
        let v_z0 = -10.0;
        x_init[2] = r_z0;
        x_init[5] = v_z0;
        x_init[6] = (700.0_f64).ln();

        let u_zero = SMatrix::<f64, 3, N>::zeros();
        let tau = 20.0;
        let dt = tau / ((N - 1) as f64);

        let mut x_out = SMatrix::<f64, 7, N>::zeros();
        nonlinear_propagate::<N>(&x_init, &u_zero, tau, &p, 8, &mut x_out);

        for k in 1..N {
            let t = (k as f64) * dt;
            let r_z_expected = r_z0 + v_z0 * t + 0.5 * g_z * t * t;
            let v_z_expected = v_z0 + g_z * t;
            assert!((x_out[(2, k)] - r_z_expected).abs() < 1.0e-6,
                    "k={k} r_z: got {} expected {}", x_out[(2, k)], r_z_expected);
            assert!((x_out[(5, k)] - v_z_expected).abs() < 1.0e-6,
                    "k={k} v_z: got {} expected {}", x_out[(5, k)], v_z_expected);
            // r_x, r_y, v_x, v_y should stay zero
            assert!(x_out[(0, k)].abs() < 1.0e-10);
            assert!(x_out[(1, k)].abs() < 1.0e-10);
            assert!(x_out[(3, k)].abs() < 1.0e-10);
            assert!(x_out[(4, k)].abs() < 1.0e-10);
        }
        eprintln!("free-fall kinematics match closed form to ≤ 1e-6 over N={N} nodes ✓");
    }

    /// Single-transition consistency: with constant reference (x̄_k =
    /// x_init ∀k, ū_k = u_hover ∀k), `nonlinear_propagate` over ONE
    /// transition (N=2) must match the linearization's per-transition
    /// prediction — that's the relationship that makes the SCvx `c_k`
    /// term consistent.
    #[test]
    fn single_transition_matches_discretize_foh_nominal() {
        const N: usize = 2;
        let p = mars_params();

        let mut x0 = SVector::<f64, 7>::zeros();
        x0[2] = 500.0;
        x0[5] = -10.0;
        x0[6] = (700.0_f64).ln();

        let mut traj = Trajectory::<N>::default();
        for k in 0..N {
            for i in 0..7 {
                traj.x[(i, k)] = x0[i];
            }
            traj.u[(2, k)] = 700.0 * 3.7114;
        }
        traj.tau = 15.0;

        let mut lin = LinearizedDynamics::<N>::default();
        discretize_foh(&traj, &p, &mut lin, 4);

        let mut x_nonlin = SMatrix::<f64, 7, N>::zeros();
        nonlinear_propagate::<N>(&x0, &traj.u, traj.tau, &p, 4, &mut x_nonlin);

        // For a SINGLE transition, the nonlinear propagation starts at
        // x̄_0 (= x_init) and the linearization is anchored at the same.
        // They must agree exactly.
        let x_lin = lin.a[0] * x0
                  + lin.b_minus[0] * SVector::<f64, 3>::from_column_slice(&[
                        traj.u[(0,0)], traj.u[(1,0)], traj.u[(2,0)]])
                  + lin.b_plus[0]  * SVector::<f64, 3>::from_column_slice(&[
                        traj.u[(0,1)], traj.u[(1,1)], traj.u[(2,1)]])
                  + lin.s[0]       * traj.tau
                  + lin.c[0];

        for i in 0..7 {
            let nl = x_nonlin[(i, 1)];
            let diff = (x_lin[i] - nl).abs();
            assert!(diff < 1.0e-9,
                    "i={i} lin={} nl={} diff={:.2e}", x_lin[i], nl, diff);
        }
        eprintln!("single transition nonlinear vs linearized: agree to ≤ 1e-9 ✓");
    }
}
