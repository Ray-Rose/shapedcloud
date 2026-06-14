use nalgebra::{SMatrix, SVector};
use scvx_core::{PhysicalParams, Trajectory};

use crate::continuous::f_continuous;
use crate::jacobian::{df_du, df_dx};

/// Time-discretized LTV approximation about a reference trajectory.
///
/// FOH on control (`u(t) = (1−λ)·u_k + λ·u_{k+1}`), RK4 sub-steps within each
/// `[t_k, t_{k+1}]`, augmented-state propagation of state ⊕ STM ⊕ FOH-
/// sensitivity ⊕ τ-sensitivity in one ODE column.
///
/// After [`discretize_foh`] returns, the discrete linearization satisfies
/// ```text
///   x_{k+1} ≈ A_k · x_k + B⁻_k · u_k + B⁺_k · u_{k+1} + s_k · τ + c_k
/// ```
/// in absolute (not delta) variables — the form used by the SOCP layer.
/// `c_k` is the constant offset chosen so the identity holds *exactly* at
/// the reference `(x̄_k, ū_k, ū_{k+1}, τ̄)`.
pub struct LinearizedDynamics<const N: usize> {
    pub a:       [SMatrix<f64, 7, 7>; N],
    pub b_minus: [SMatrix<f64, 7, 3>; N],
    pub b_plus:  [SMatrix<f64, 7, 3>; N],
    pub c:       [SVector<f64, 7>; N],
    pub s:       [SVector<f64, 7>; N],
}

impl<const N: usize> Default for LinearizedDynamics<N> {
    fn default() -> Self {
        Self {
            a:       [SMatrix::zeros(); N],
            b_minus: [SMatrix::zeros(); N],
            b_plus:  [SMatrix::zeros(); N],
            c:       [SVector::zeros(); N],
            s:       [SVector::zeros(); N],
        }
    }
}

/// Augmented integration state. Carries the trajectory plus all four
/// linearization sensitivities so a single RK4 pass produces every output
/// block. Plain old data, `Copy` for cheap RK4 stage construction.
#[derive(Clone, Copy)]
struct AugmentedState {
    x:       SVector<f64, 7>,
    phi:     SMatrix<f64, 7, 7>, // ∂x(t)/∂x_k
    b_minus: SMatrix<f64, 7, 3>, // ∂x(t)/∂u_k
    b_plus:  SMatrix<f64, 7, 3>, // ∂x(t)/∂u_{k+1}
    s:       SVector<f64, 7>,    // ∂x(t)/∂τ
}

impl AugmentedState {
    /// Initial condition at `t = t_k`: state = `x_k`, STM = `I`, all other
    /// sensitivities zero (the propagators start fresh at each node).
    fn initial(x_k: SVector<f64, 7>) -> Self {
        Self {
            x:       x_k,
            phi:     SMatrix::identity(),
            b_minus: SMatrix::zeros(),
            b_plus:  SMatrix::zeros(),
            s:       SVector::zeros(),
        }
    }
}

/// Per-interval constants needed by every RK4 stage evaluation. Bundling
/// reduces signature noise (and shuts up clippy `too_many_arguments`,
/// which is right to flag this in safety-relevant code paths).
struct StepContext<'a> {
    dt_norm: f64,
    tau:     f64,
    u_k:     &'a SVector<f64, 3>,
    u_kp1:   &'a SVector<f64, 3>,
    params:  &'a PhysicalParams,
}

/// Compute `dY/ds` at normalized sub-time `s_local ∈ [0, dt_norm]`.
///
/// The augmented ODE:
/// ```text
///   ẋ       = τ · f(x, u(λ))
///   Φ̇       = τ · A · Φ
///   Ḃ⁻      = τ · ( A · B⁻ + (1−λ) · B )
///   Ḃ⁺      = τ · ( A · B⁺ +     λ  · B )
///   ṡ       = τ · A · s + f(x, u(λ))     ← τ-sensitivity ODE
/// ```
/// where `A = ∂f/∂x`, `B = ∂f/∂u`, `λ = s_local / dt_norm`.
fn augmented_ode(
    y:       &AugmentedState,
    s_local: f64,
    ctx:     &StepContext<'_>,
) -> AugmentedState {
    let lambda = s_local / ctx.dt_norm;
    let u_t    = ctx.u_k * (1.0 - lambda) + ctx.u_kp1 * lambda;

    let f_val = f_continuous(&y.x, &u_t, ctx.params);
    let a_mat = df_dx(&y.x, &u_t, ctx.params);
    let b_mat = df_du(&y.x, &u_t, ctx.params);

    AugmentedState {
        x:       f_val * ctx.tau,
        phi:     a_mat * y.phi * ctx.tau,
        b_minus: (a_mat * y.b_minus + b_mat * (1.0 - lambda)) * ctx.tau,
        b_plus:  (a_mat * y.b_plus  + b_mat *        lambda ) * ctx.tau,
        s:       a_mat * y.s * ctx.tau + f_val,
    }
}

/// Classical fourth-order Runge-Kutta step over `[s, s+h]`.
fn rk4_step(
    y:   &AugmentedState,
    s:   f64,
    h:   f64,
    ctx: &StepContext<'_>,
) -> AugmentedState {
    let k1 = augmented_ode(y, s, ctx);

    let y2 = AugmentedState {
        x:       y.x       + k1.x       * (0.5 * h),
        phi:     y.phi     + k1.phi     * (0.5 * h),
        b_minus: y.b_minus + k1.b_minus * (0.5 * h),
        b_plus:  y.b_plus  + k1.b_plus  * (0.5 * h),
        s:       y.s       + k1.s       * (0.5 * h),
    };
    let k2 = augmented_ode(&y2, s + 0.5 * h, ctx);

    let y3 = AugmentedState {
        x:       y.x       + k2.x       * (0.5 * h),
        phi:     y.phi     + k2.phi     * (0.5 * h),
        b_minus: y.b_minus + k2.b_minus * (0.5 * h),
        b_plus:  y.b_plus  + k2.b_plus  * (0.5 * h),
        s:       y.s       + k2.s       * (0.5 * h),
    };
    let k3 = augmented_ode(&y3, s + 0.5 * h, ctx);

    let y4 = AugmentedState {
        x:       y.x       + k3.x       * h,
        phi:     y.phi     + k3.phi     * h,
        b_minus: y.b_minus + k3.b_minus * h,
        b_plus:  y.b_plus  + k3.b_plus  * h,
        s:       y.s       + k3.s       * h,
    };
    let k4 = augmented_ode(&y4, s + h, ctx);

    let one_sixth_h = h / 6.0;
    AugmentedState {
        x:       y.x       + (k1.x       + k2.x       * 2.0 + k3.x       * 2.0 + k4.x      ) * one_sixth_h,
        phi:     y.phi     + (k1.phi     + k2.phi     * 2.0 + k3.phi     * 2.0 + k4.phi    ) * one_sixth_h,
        b_minus: y.b_minus + (k1.b_minus + k2.b_minus * 2.0 + k3.b_minus * 2.0 + k4.b_minus) * one_sixth_h,
        b_plus:  y.b_plus  + (k1.b_plus  + k2.b_plus  * 2.0 + k3.b_plus  * 2.0 + k4.b_plus ) * one_sixth_h,
        s:       y.s       + (k1.s       + k2.s       * 2.0 + k3.s       * 2.0 + k4.s      ) * one_sixth_h,
    }
}

/// Extract column `j` of an SMatrix as an owned SVector.
/// Helper because nalgebra's column-view API is awkward with const generics.
#[inline]
fn col<const R: usize, const C: usize>(
    m: &SMatrix<f64, R, C>,
    j: usize,
) -> SVector<f64, R> {
    let mut out = SVector::<f64, R>::zeros();
    for i in 0..R {
        out[i] = m[(i, j)];
    }
    out
}

/// Discretize the dynamics about the reference trajectory, filling `lin`
/// in place. Uses FOH on control + RK4 with `rk4_substeps` sub-steps per
/// node-to-node interval. Recommend `rk4_substeps = 4` for a good
/// accuracy/cost tradeoff; gates pass at 2.
///
/// Indices `0..N-1` of `lin` are filled; index `N-1` is left untouched
/// (there is no "next" interval after the final node).
pub fn discretize_foh<const N: usize>(
    reference:     &Trajectory<N>,
    params:        &PhysicalParams,
    lin:           &mut LinearizedDynamics<N>,
    rk4_substeps:  u32,
) {
    if N < 2 || rk4_substeps == 0 {
        return;
    }
    let dt_norm = 1.0 / ((N - 1) as f64);
    let tau     = reference.tau;
    let h_sub   = dt_norm / (rk4_substeps as f64);

    for k in 0..(N - 1) {
        let x_k_ref = col(&reference.x, k);
        let u_k     = col(&reference.u, k);
        let u_kp1   = col(&reference.u, k + 1);

        let ctx = StepContext { dt_norm, tau, u_k: &u_k, u_kp1: &u_kp1, params };
        let mut y = AugmentedState::initial(x_k_ref);
        let mut s = 0.0_f64;
        for _ in 0..rk4_substeps {
            y = rk4_step(&y, s, h_sub, &ctx);
            s += h_sub;
        }

        lin.a[k]       = y.phi;
        lin.b_minus[k] = y.b_minus;
        lin.b_plus[k]  = y.b_plus;
        lin.s[k]       = y.s;

        // c_k chosen so the identity x_{k+1} = A·x + B⁻·u_k + B⁺·u_{k+1} + s·τ + c
        // holds exactly at the reference.
        let x_kp1_nominal = y.x;
        lin.c[k] = x_kp1_nominal
            - lin.a[k]       * x_k_ref
            - lin.b_minus[k] * u_k
            - lin.b_plus[k]  * u_kp1
            - lin.s[k]       * tau;
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

    /// Build a constant-control reference trajectory: `x_k = x_0` (we
    /// don't propagate the reference for testing — we treat each node as
    /// independent, since the linearization happens around each node's
    /// own state). `u_k = u_0` for all k. `τ = tau`.
    fn const_ref<const N: usize>(
        x0: SVector<f64, 7>,
        u0: SVector<f64, 3>,
        tau: f64,
    ) -> Trajectory<N> {
        let mut traj = Trajectory::<N>::default();
        for k in 0..N {
            for i in 0..7 { traj.x[(i, k)] = x0[i]; }
            for i in 0..3 { traj.u[(i, k)] = u0[i]; }
        }
        traj.tau = tau;
        traj
    }

    /// Run the full nonlinear FOH+RK4 propagation from (x_k, u_k, u_{k+1})
    /// over one interval — no STM, no sensitivities. Used as the
    /// roundtrip oracle.
    fn nonlinear_step<const N: usize>(
        x_k:    &SVector<f64, 7>,
        u_k:    &SVector<f64, 3>,
        u_kp1:  &SVector<f64, 3>,
        tau:    f64,
        rk4_substeps: u32,
        params: &PhysicalParams,
    ) -> SVector<f64, 7> {
        let dt_norm = 1.0 / ((N - 1) as f64);
        let h_sub   = dt_norm / (rk4_substeps as f64);
        let mut x = *x_k;
        let mut s = 0.0;
        for _ in 0..rk4_substeps {
            let f = |t: f64, x_in: &SVector<f64, 7>| {
                let lam = t / dt_norm;
                let u   = u_k * (1.0 - lam) + u_kp1 * lam;
                f_continuous(x_in, &u, params) * tau
            };
            let k1 = f(s,             &x);
            let k2 = f(s + 0.5 * h_sub, &(x + k1 * (0.5 * h_sub)));
            let k3 = f(s + 0.5 * h_sub, &(x + k2 * (0.5 * h_sub)));
            let k4 = f(s + h_sub,       &(x + k3 * h_sub));
            x += (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (h_sub / 6.0);
            s += h_sub;
        }
        x
    }

    #[test]
    fn linearization_identity_at_reference() {
        // Gate from plan: at the reference, the discrete linearization
        // x_{k+1} = A x + B⁻ u_k + B⁺ u_{k+1} + s τ + c
        // must recover the nominal propagation exactly (within numerical
        // noise from f64 + RK4 stage evaluation).
        const N: usize = 10;
        let p   = test_params();
        let mut x0 = SVector::<f64, 7>::zeros();
        x0[2] = 1500.0;            // altitude
        x0[5] = -50.0;             // descent rate
        x0[6] = (800.0_f64).ln();  // m = 800 kg
        let u0 = SVector::<f64, 3>::from_column_slice(&[100.0, 50.0, 3500.0]);
        let tau = 25.0;
        let traj = const_ref::<N>(x0, u0, tau);

        let mut lin = LinearizedDynamics::<N>::default();
        discretize_foh(&traj, &p, &mut lin, 4);

        for k in 0..(N - 1) {
            let x_k    = col(&traj.x, k);
            let u_k    = col(&traj.u, k);
            let u_kp1  = col(&traj.u, k + 1);
            let x_lin  = lin.a[k] * x_k
                       + lin.b_minus[k] * u_k
                       + lin.b_plus[k]  * u_kp1
                       + lin.s[k]       * tau
                       + lin.c[k];
            let x_nl = nonlinear_step::<N>(&x_k, &u_k, &u_kp1, tau, 4, &p);
            for i in 0..7 {
                let diff = (x_lin[i] - x_nl[i]).abs();
                let scale = x_lin[i].abs().max(x_nl[i].abs()).max(1.0);
                assert!(
                    diff < 1.0e-9 || diff / scale < 1.0e-12,
                    "k={k}, i={i}: lin = {}, nl = {}, diff = {:.2e}",
                    x_lin[i], x_nl[i], diff
                );
            }
        }
    }

    #[test]
    fn free_fall_stm_matches_kinematics() {
        // Zero thrust, zero initial velocity, drag~0 at v=0. Then
        // f ≈ (0, g, ~0), A_cont ≈ [[0, I, 0], [0, 0, 0], [0, 0, 0]]
        // — almost nilpotent. The STM should be close to
        //   [[I, τ·dt·I, 0], [0, I, 0], [0, 0, 1]]
        // up to (a) RK4 truncation (exact for linear ODEs of degree ≤ 4) and
        // (b) the THRUST_SMOOTH_EPS bias: ‖u‖_ε ≈ 0.01 N when u=0 yields
        // ż ≈ −α·0.01/(Isp·g₀) ≈ −4.5e-9, which propagates through the STM
        // giving O(1e-8) residual on the kinematic block. Tolerance set to
        // 1e-7 to clear the smoothing floor with margin.
        const N: usize = 11;
        let p   = test_params();
        let mut x0 = SVector::<f64, 7>::zeros();
        x0[6] = (500.0_f64).ln();
        let u0 = SVector::<f64, 3>::zeros();
        let tau = 1.0;
        let traj = const_ref::<N>(x0, u0, tau);

        let mut lin = LinearizedDynamics::<N>::default();
        discretize_foh(&traj, &p, &mut lin, 2);

        let dt_norm = 1.0 / ((N - 1) as f64);
        let a0 = lin.a[0];

        // ∂r/∂v block ≈ τ · dt_norm · I (kinematic coupling).
        // Why not machine-precision: at sub-step time t > 0, gravity has
        // already pulled v_z to ≈ g·t, drag couples ‖v‖ back into v̇, and
        // the STM picks up O(α·D·‖v‖·dt) ≈ 1e-8 perturbations.
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { tau * dt_norm } else { 0.0 };
                assert!(
                    (a0[(i, 3 + j)] - expected).abs() < 1.0e-7,
                    "STM[r{i}, v{j}] = {} vs {expected}", a0[(i, 3 + j)]
                );
            }
        }
        // Position diagonal: machine-clean (no drag dependency on r).
        for i in 0..3 {
            assert!((a0[(i, i)] - 1.0).abs() < 1.0e-12,
                    "position diag a[{i},{i}] = {}", a0[(i, i)]);
        }
        // Velocity diagonal: deviates by exactly ~α·D·(v_ne + v²/v_ne)·dt_sub.
        // For v_z (driven negative by gravity during the step), the
        // deviation is roughly 2× that of v_x, v_y. Observed: ~7e-7 in the
        // worst case — well within 1e-6.
        for i in 0..3 {
            assert!((a0[(3 + i, 3 + i)] - 1.0).abs() < 1.0e-6,
                    "velocity diag a[{},{}] = {}", 3+i, 3+i, a0[(3 + i, 3 + i)]);
        }
        // Mass diagonal: see plan comment on THRUST_SMOOTH_EPS bias.
        assert!((a0[(6, 6)] - 1.0).abs() < 1.0e-6,
                "mass diag a[6,6] = {}", a0[(6, 6)]);
    }

    #[test]
    fn linearization_is_first_order_accurate_under_state_perturbation() {
        // Perturb x_k by ε, compare linear-predicted vs nonlinear-propagated
        // x_{k+1}. Difference should be O(ε²): halving ε quarters the error.
        const N: usize = 5;
        let p   = test_params();
        let mut x0 = SVector::<f64, 7>::zeros();
        x0[2] = 1000.0;
        x0[5] = -30.0;
        x0[6] = (700.0_f64).ln();
        let u0 = SVector::<f64, 3>::from_column_slice(&[50.0, 30.0, 3000.0]);
        let tau = 20.0;
        let traj = const_ref::<N>(x0, u0, tau);

        let mut lin = LinearizedDynamics::<N>::default();
        discretize_foh(&traj, &p, &mut lin, 4);

        let k = 1;
        let x_k_ref = col(&traj.x, k);
        let u_k     = col(&traj.u, k);
        let u_kp1   = col(&traj.u, k + 1);

        // Perturb in v_z direction
        let mut dx = SVector::<f64, 7>::zeros();
        dx[5] = 1.0;

        let mut prev_err = f64::INFINITY;
        let mut prev_eps = 0.0;
        for &eps in &[1.0e-2, 1.0e-3, 1.0e-4] {
            let x_k_pert = x_k_ref + dx * eps;
            let x_kp1_nl = nonlinear_step::<N>(&x_k_pert, &u_k, &u_kp1, tau, 4, &p);
            let x_kp1_lin = lin.a[k] * x_k_pert
                          + lin.b_minus[k] * u_k
                          + lin.b_plus[k]  * u_kp1
                          + lin.s[k]       * tau
                          + lin.c[k];
            let err = (x_kp1_nl - x_kp1_lin).norm();

            if prev_eps > 0.0 {
                // Halving ε should drop err by ~factor (eps/prev_eps)²
                let ratio_eps = (eps / prev_eps).powi(2);
                let ratio_err = err / prev_err;
                eprintln!("ε={eps:.0e}: err = {err:.3e}, ratio = {:.2} (expected ~{:.2})",
                          ratio_err, ratio_eps);
                // Tolerate up to 3× the expected ratio (round-off floor).
                assert!(
                    ratio_err < ratio_eps * 3.0 || err < 1.0e-10,
                    "first-order accuracy violated at ε={eps}: ratio {} vs expected {}",
                    ratio_err, ratio_eps
                );
            }
            prev_err = err;
            prev_eps = eps;
        }
    }

    #[test]
    fn b_minus_plus_split_sums_to_total_control_jacobian() {
        // For a single FOH interval at constant control, B⁻ + B⁺ should equal
        // the "lumped" ∂x_{k+1}/∂(u_k = u_{k+1}) Jacobian — perturbing both
        // endpoints together is equivalent to perturbing a constant control.
        const N: usize = 5;
        let p   = test_params();
        let mut x0 = SVector::<f64, 7>::zeros();
        x0[2] = 1000.0;
        x0[6] = (600.0_f64).ln();
        let u0 = SVector::<f64, 3>::from_column_slice(&[0.0, 0.0, 2500.0]);
        let tau = 15.0;
        let traj = const_ref::<N>(x0, u0, tau);

        let mut lin = LinearizedDynamics::<N>::default();
        discretize_foh(&traj, &p, &mut lin, 4);

        let k = 0;
        let total = lin.b_minus[k] + lin.b_plus[k];

        // FD-check: perturb BOTH u_k and u_{k+1} together by ε.
        let eps = 1.0e-5;
        let x_k_ref = col(&traj.x, k);
        let u_k     = col(&traj.u, k);
        let u_kp1   = col(&traj.u, k + 1);
        for j in 0..3 {
            let mut du = SVector::<f64, 3>::zeros();
            du[j] = eps;
            let xp = nonlinear_step::<N>(&x_k_ref, &(u_k + du), &(u_kp1 + du), tau, 4, &p);
            let xm = nonlinear_step::<N>(&x_k_ref, &(u_k - du), &(u_kp1 - du), tau, 4, &p);
            let fd = (xp - xm) / (2.0 * eps);
            for i in 0..7 {
                let diff = (fd[i] - total[(i, j)]).abs();
                let scale = fd[i].abs().max(total[(i, j)].abs()).max(1.0);
                assert!(
                    diff / scale < 1.0e-5,
                    "row {i}, col {j}: FD = {}, analytic = {}", fd[i], total[(i, j)]
                );
            }
        }
    }

    #[test]
    fn tau_sensitivity_matches_central_difference() {
        // s_k = ∂x_{k+1}/∂τ.  Verify by FD.
        const N: usize = 5;
        let p   = test_params();
        let mut x0 = SVector::<f64, 7>::zeros();
        x0[2] = 1000.0;
        x0[5] = -20.0;
        x0[6] = (700.0_f64).ln();
        let u0 = SVector::<f64, 3>::from_column_slice(&[100.0, 0.0, 3000.0]);
        let tau = 20.0;
        let traj = const_ref::<N>(x0, u0, tau);

        let mut lin = LinearizedDynamics::<N>::default();
        discretize_foh(&traj, &p, &mut lin, 4);

        let k = 0;
        let x_k_ref = col(&traj.x, k);
        let u_k     = col(&traj.u, k);
        let u_kp1   = col(&traj.u, k + 1);

        let h_tau = tau * 1.0e-6;
        let xp = nonlinear_step::<N>(&x_k_ref, &u_k, &u_kp1, tau + h_tau, 8, &p);
        let xm = nonlinear_step::<N>(&x_k_ref, &u_k, &u_kp1, tau - h_tau, 8, &p);
        let fd = (xp - xm) / (2.0 * h_tau);
        for i in 0..7 {
            let diff = (fd[i] - lin.s[k][i]).abs();
            let scale = fd[i].abs().max(lin.s[k][i].abs()).max(1.0);
            assert!(
                diff / scale < 1.0e-5,
                "i={i}: FD = {}, s_k = {}", fd[i], lin.s[k][i]
            );
        }
    }
}
