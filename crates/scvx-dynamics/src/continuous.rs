use libm::{exp, sqrt};
use nalgebra::SVector;
use scvx_core::PhysicalParams;

/// Drag-smoothing epsilon. Avoids the non-Lipschitz Jacobian of `‖v‖·v` at
/// `v = 0`. Units: `(m/s)²`. Bias is `O(ε)` near the origin and negligible
/// for typical landing velocities (`‖v‖ ≫ √ε ≈ 1 mm/s`).
/// See plan, "Honest risks: drag non-Lipschitz at v=0."
pub const VELOCITY_SMOOTH_EPS: f64 = 1.0e-6;

/// Thrust-magnitude smoothing epsilon. `‖u‖_ε = sqrt(‖u‖² + ε_u)` avoids
/// the same non-Lipschitz issue in the mass derivative when thrust is at
/// or near zero. Units: `N²`. Smaller than `VELOCITY_SMOOTH_EPS` because
/// thrust magnitudes are large relative to noise.
pub const THRUST_SMOOTH_EPS: f64 = 1.0e-4;

/// Continuous-time 3-DoF powered-descent dynamics with aerodynamic drag,
/// log-mass parameterization.
///
/// State `x = [r, v, z]` ∈ ℝ⁷:
/// - `r` (3): position in inertial frame, m
/// - `v` (3): velocity in inertial frame, m/s
/// - `z`    : `ln(m)`, dimensionless
///
/// Control `u` ∈ ℝ³: thrust vector in inertial frame, N.
///
/// Equations (with `α = e^{−z} = 1/m` and `‖·‖_ε = sqrt(‖·‖² + ε)`):
/// ```text
///   ṙ = v
///   v̇ = α·(u − D·‖v‖_ε·v) + g
///   ż = −α·‖u‖_ε / (Isp·g₀)
/// ```
/// where `D = ½·ρ·Cd·A` is the lumped drag coefficient.
pub fn f_continuous(
    x: &SVector<f64, 7>,
    u: &SVector<f64, 3>,
    p: &PhysicalParams,
) -> SVector<f64, 7> {
    // Unpack
    let v: SVector<f64, 3> = SVector::<f64, 3>::from_column_slice(&[x[3], x[4], x[5]]);
    let z = x[6];

    let alpha    = exp(-z);                                   // 1/m
    let v_norm_e = sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2] + VELOCITY_SMOOTH_EPS);
    let u_norm_e = sqrt(u[0]*u[0] + u[1]*u[1] + u[2]*u[2] + THRUST_SMOOTH_EPS);
    let drag     = 0.5 * p.rho * p.cd_a * v_norm_e;           // scalar drag coefficient

    let mut out = SVector::<f64, 7>::zeros();

    // ṙ = v
    out[0] = v[0];
    out[1] = v[1];
    out[2] = v[2];

    // v̇ = α·(u − drag·v) + g
    out[3] = alpha * (u[0] - drag * v[0]) + p.g[0];
    out[4] = alpha * (u[1] - drag * v[1]) + p.g[1];
    out[5] = alpha * (u[2] - drag * v[2]) + p.g[2];

    // ż = −α·‖u‖_ε / (Isp·g₀)
    out[6] = -alpha * u_norm_e / (p.isp * p.g0);

    out
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    extern crate std;

    use super::*;

    /// Reasonable Mars-landing-style physical parameters. Not flight-tuned,
    /// just well-conditioned numbers for tests.
    fn test_params() -> PhysicalParams {
        PhysicalParams {
            g:             [0.0, 0.0, -3.7114], // Mars, downward
            m_dry:          200.0,
            m_wet:         1000.0,
            isp:            225.0,              // hypergolic-like
            g0:               9.80665,
            t_min:         1000.0,
            t_max:         6000.0,
            cos_theta_max:    0.7660444,        // 40° pointing cone
            tan_gamma_gs:     1.0,              // 45° glide slope
            rho:              0.020,            // Mars surface, kg/m³
            cd_a:             1.0,              // m²
            tau_lo:           5.0,
            tau_hi:          50.0,
        }
    }

    #[test]
    fn rest_with_zero_thrust_falls_under_gravity() {
        // x = (r=0, v=0, z=ln(1000))   u = 0
        // Smoothed thrust ‖u‖_ε = √ε ≈ 0.01 N is tiny but nonzero, so
        // mass drips slightly. v̇ should be ≈ g. ṙ should be 0.
        let p = test_params();
        let mut x = SVector::<f64, 7>::zeros();
        x[6] = (1000.0_f64).ln();
        let u = SVector::<f64, 3>::zeros();
        let xdot = f_continuous(&x, &u, &p);

        // ṙ = 0
        assert!(xdot[0].abs() < 1e-12);
        assert!(xdot[1].abs() < 1e-12);
        assert!(xdot[2].abs() < 1e-12);
        // v̇ ≈ g (drag = 0 at v=0; thrust = 0)
        assert!((xdot[3] - p.g[0]).abs() < 1e-10);
        assert!((xdot[4] - p.g[1]).abs() < 1e-10);
        // Drag-smoothing at v=0 contributes -alpha*drag*0 = 0 exactly, so
        // v̇[2] must equal g[2] up to thrust-smoothing residual.
        // ‖u‖_ε = √(0+1e-4) = 0.01. alpha = 1/1000 = 1e-3.
        // ż = -1e-3 * 0.01 / (225 * 9.80665) ≈ -4.53e-9 — negligible.
        assert!((xdot[5] - p.g[2]).abs() < 1e-10);
        // ż small but negative
        assert!(xdot[6] < 0.0);
        assert!(xdot[6] > -1.0e-8);
    }

    #[test]
    fn hover_thrust_cancels_gravity() {
        // For 1000 kg on Mars with g_z = -3.7114, hover requires
        // u_z = m·|g_z| = 1000·3.7114 ≈ 3711.4 N upward.
        // At v=0, drag = 0. Then v̇ = α·u + g should be 0 in z.
        let p = test_params();
        let m: f64 = 1000.0;
        let mut x = SVector::<f64, 7>::zeros();
        x[6] = m.ln();
        let mut u = SVector::<f64, 3>::zeros();
        u[2] = -m * p.g[2]; // m * 3.7114
        let xdot = f_continuous(&x, &u, &p);

        assert!(xdot[3].abs() < 1e-10);
        assert!(xdot[4].abs() < 1e-10);
        assert!(xdot[5].abs() < 1e-9, "v̇_z = {}", xdot[5]);
        // ṁ < 0 (we're burning ~3711 N of thrust)
        assert!(xdot[6] < 0.0);
    }

    #[test]
    fn drag_opposes_velocity() {
        // At v in +x, drag should accelerate negatively in +x.
        let p = test_params();
        let mut x = SVector::<f64, 7>::zeros();
        x[3] = 100.0;             // v_x = 100 m/s
        x[6] = (500.0_f64).ln();  // m = 500 kg
        let u = SVector::<f64, 3>::zeros();
        let xdot = f_continuous(&x, &u, &p);

        // v̇_x = α·(0 − D·‖v‖_ε·v_x) = -α·D·‖v‖·v_x. With v_x > 0, must be < 0.
        assert!(xdot[3] < 0.0, "drag should slow positive x velocity, got {}", xdot[3]);
        // Other v components only feel gravity.
        assert!(xdot[4].abs() < 1e-10);
        assert!((xdot[5] - p.g[2]).abs() < 1e-9);
    }

    #[test]
    fn lighter_vehicle_has_larger_thrust_acceleration() {
        // Same thrust, half the mass → twice the acceleration.
        let p = test_params();
        let u = SVector::<f64, 3>::from_column_slice(&[0.0, 0.0, 5000.0]);

        let mut x1 = SVector::<f64, 7>::zeros();
        x1[6] = (1000.0_f64).ln();
        let mut x2 = SVector::<f64, 7>::zeros();
        x2[6] = (500.0_f64).ln();

        let xd1 = f_continuous(&x1, &u, &p);
        let xd2 = f_continuous(&x2, &u, &p);

        // Thrust contribution to v̇_z = α·u_z. Subtract gravity to isolate.
        let a1 = xd1[5] - p.g[2];
        let a2 = xd2[5] - p.g[2];
        assert!((a2 / a1 - 2.0).abs() < 1e-10, "ratio {}", a2 / a1);
    }
}
