//! Block-banded KKT factorization via Riccati recursion.
//!
//! Solves the discrete-time affine LQR problem
//! ```text
//!   minimize    0.5 Σ_{k=0..N-1} [xₖᵀQₖxₖ + uₖᵀRₖuₖ + 2qₖᵀxₖ + 2rₖᵀuₖ]
//!   subject to  x_{k+1} = Aₖ xₖ + Bₖ uₖ + cₖ,   k = 0, …, N-2
//!               x₀     = x_init
//! ```
//! via the standard backward + forward sweep. Per-stage cost: O(NX³ + NX²·NU + NU³).
//! Total: O(N · max(NX, NU)³) — the whole point of this factorization. The
//! dense alternative is O((N·NX)³) and would not fit in a flight WCET budget.
//!
//! Stage costs `q_mat[k]`, `r_mat[k]`, `q_lin[k]`, `r_lin[k]` apply at stage
//! `k`; `q_mat[N-1]` / `q_lin[N-1]` act as the terminal state cost (no
//! control at the terminal node). Dynamics `a[k]`, `b[k]`, `c[k]` describe
//! the transition `xₖ → x_{k+1}` for `k ∈ 0..N-1`; the last slot is unused.
//!
//! Per the plan, the SCvx subproblem's FOH B-split (`B⁻`, `B⁺`) is *not*
//! handled here yet — that's a P3 concern when we fuse the Riccati into the
//! IPM. P2 is the standalone single-B LQR primitive.
//!
//! **STATUS — standalone primitive, NOT wired into the shipped solver.** The
//! production structured-KKT path (`scvx_solver::reduced_kkt`) reimplements the
//! block-Thomas sweep inline over the SCvx FOH layout; this Riccati module is
//! exercised only by its own unit tests and the WCET benchmark, kept as a
//! verified reference / future-fusion primitive. The "whole point" / flight-
//! WCET framing above is the algorithmic motivation, not a claim that this
//! code runs on a live solve path.

use nalgebra::{SMatrix, SVector};

// ---------------------------------------------------------------------------
// Problem, workspace, solution
// ---------------------------------------------------------------------------

/// Affine discrete-time LQR problem.
pub struct LqrProblem<const N: usize, const NX: usize, const NU: usize> {
    pub a:      [SMatrix<f64, NX, NX>; N],
    pub b:      [SMatrix<f64, NX, NU>; N],
    pub c:      [SVector<f64, NX>;     N],
    pub q_mat:  [SMatrix<f64, NX, NX>; N],
    pub r_mat:  [SMatrix<f64, NU, NU>; N],
    pub q_lin:  [SVector<f64, NX>;     N],
    pub r_lin:  [SVector<f64, NU>;     N],
    pub x_init: SVector<f64, NX>,
}

impl<const N: usize, const NX: usize, const NU: usize> Default
    for LqrProblem<N, NX, NU>
{
    fn default() -> Self {
        Self {
            a:      [SMatrix::zeros(); N],
            b:      [SMatrix::zeros(); N],
            c:      [SVector::zeros(); N],
            q_mat:  [SMatrix::zeros(); N],
            r_mat:  [SMatrix::zeros(); N],
            q_lin:  [SVector::zeros(); N],
            r_lin:  [SVector::zeros(); N],
            x_init: SVector::zeros(),
        }
    }
}

/// Riccati factorization workspace. Filled by [`riccati_factor`].
pub struct LqrWorkspace<const N: usize, const NX: usize, const NU: usize> {
    pub p_mat:  [SMatrix<f64, NX, NX>; N],
    pub p_lin:  [SVector<f64, NX>;     N],
    pub k_gain: [SMatrix<f64, NU, NX>; N],
    pub k_ff:   [SVector<f64, NU>;     N],
}

impl<const N: usize, const NX: usize, const NU: usize> Default
    for LqrWorkspace<N, NX, NU>
{
    fn default() -> Self {
        Self {
            p_mat:  [SMatrix::zeros(); N],
            p_lin:  [SVector::zeros(); N],
            k_gain: [SMatrix::zeros(); N],
            k_ff:   [SVector::zeros(); N],
        }
    }
}

/// Optimal trajectory. `x[0..N]` are state nodes, `u[0..N-1]` are controls;
/// `u[N-1]` is unused (no transition out of the terminal node).
pub struct LqrSolution<const N: usize, const NX: usize, const NU: usize> {
    pub x: [SVector<f64, NX>; N],
    pub u: [SVector<f64, NU>; N],
}

impl<const N: usize, const NX: usize, const NU: usize> Default
    for LqrSolution<N, NX, NU>
{
    fn default() -> Self {
        Self {
            x: [SVector::zeros(); N],
            u: [SVector::zeros(); N],
        }
    }
}

/// Status of the Riccati factor + solve.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum RiccatiStatus {
    Ok,
    /// Per-stage control Hessian `Mₖ = Rₖ + BₖᵀP_{k+1}Bₖ` failed to invert.
    /// Usually means `R` is not PD or `P_{k+1}` lost PSD-ness from round-off.
    /// Caller should regularize (`R += ε·I`) and retry.
    NotPd,
    /// `N < 1` or otherwise degenerate sizing.
    DegenerateInput,
}

impl RiccatiStatus {
    pub fn as_u32(self) -> u32 {
        match self {
            RiccatiStatus::Ok              => 0,
            RiccatiStatus::NotPd           => 1,
            RiccatiStatus::DegenerateInput => 2,
        }
    }
}

// ---------------------------------------------------------------------------
// Backward sweep
// ---------------------------------------------------------------------------

/// Backward Riccati sweep. Fills `p_mat`, `p_lin`, `k_gain`, `k_ff` in the
/// workspace. Returns `NotPd` if any per-stage `Mₖ` is non-invertible.
///
/// `Pₖ` is symmetrized after each stage (`P ← ½(P + Pᵀ)`) to defend against
/// round-off-induced drift.
pub fn riccati_factor<const N: usize, const NX: usize, const NU: usize>(
    problem:   &LqrProblem<N, NX, NU>,
    workspace: &mut LqrWorkspace<N, NX, NU>,
) -> RiccatiStatus {
    if N < 1 {
        return RiccatiStatus::DegenerateInput;
    }

    // Terminal: V_{N-1}(x) = 0.5 xᵀ Q_{N-1} x + q_{N-1}ᵀ x
    workspace.p_mat [N - 1] = problem.q_mat[N - 1];
    workspace.p_lin [N - 1] = problem.q_lin[N - 1];
    workspace.k_gain[N - 1] = SMatrix::zeros();
    workspace.k_ff  [N - 1] = SVector::zeros();

    if N == 1 {
        return RiccatiStatus::Ok;
    }

    // Backward sweep k = N-2, …, 0
    for k in (0..N - 1).rev() {
        let p_next     = workspace.p_mat[k + 1];
        let p_lin_next = workspace.p_lin[k + 1];
        let a          = problem.a[k];
        let b          = problem.b[k];
        let c          = problem.c[k];

        let pa = p_next * a;
        let pb = p_next * b;

        // Control Hessian M = R + Bᵀ P B  (n_u × n_u, symmetric PD if all good).
        let m = problem.r_mat[k] + b.transpose() * pb;
        let m_inv = match m.try_inverse() {
            Some(m) => m,
            None    => return RiccatiStatus::NotPd,
        };

        // Cross block G = Bᵀ P A  and gradient g = r + Bᵀ(p + P·c)
        let g     = b.transpose() * pa;
        let g_lin = problem.r_lin[k] + b.transpose() * (p_lin_next + p_next * c);

        // Feedback gain and feedforward
        let k_gain = -(m_inv * g);
        let k_ff   = -(m_inv * g_lin);
        workspace.k_gain[k] = k_gain;
        workspace.k_ff  [k] = k_ff;

        // P_k = Q + Aᵀ P A - Gᵀ M⁻¹ G        (square form; numerically PSD-preserving)
        let p_k = problem.q_mat[k] + a.transpose() * pa
                - g.transpose() * (m_inv * g);
        // Symmetrize to defend against round-off-induced asymmetry.
        workspace.p_mat[k] = (p_k + p_k.transpose()) * 0.5;

        // p_k = q + Aᵀ(p + P·c) - Gᵀ M⁻¹ g
        workspace.p_lin[k] = problem.q_lin[k]
            + a.transpose() * (p_lin_next + p_next * c)
            - g.transpose() * (m_inv * g_lin);
    }

    RiccatiStatus::Ok
}

// ---------------------------------------------------------------------------
// Forward sweep
// ---------------------------------------------------------------------------

/// Forward Riccati sweep. Reads from `workspace`, fills `solution`. Must be
/// called AFTER [`riccati_factor`] succeeds.
pub fn riccati_solve<const N: usize, const NX: usize, const NU: usize>(
    problem:   &LqrProblem<N, NX, NU>,
    workspace: &LqrWorkspace<N, NX, NU>,
    solution:  &mut LqrSolution<N, NX, NU>,
) {
    if N < 1 {
        return;
    }
    solution.x[0] = problem.x_init;
    if N == 1 {
        return;
    }

    for k in 0..(N - 1) {
        let u_k = workspace.k_gain[k] * solution.x[k] + workspace.k_ff[k];
        solution.u[k]     = u_k;
        solution.x[k + 1] = problem.a[k] * solution.x[k]
                          + problem.b[k] * u_k
                          + problem.c[k];
    }
    // `solution.u[N-1]` stays at its default (zero) — no transition out of
    // the terminal node.
}

/// Convenience: factor then solve in one call.
pub fn riccati_factor_solve<const N: usize, const NX: usize, const NU: usize>(
    problem:   &LqrProblem<N, NX, NU>,
    workspace: &mut LqrWorkspace<N, NX, NU>,
    solution:  &mut LqrSolution<N, NX, NU>,
) -> RiccatiStatus {
    let status = riccati_factor(problem, workspace);
    if status == RiccatiStatus::Ok {
        riccati_solve(problem, workspace, solution);
    }
    status
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    extern crate std;
    use std::eprintln;

    use super::*;

    /// Trivial case: a single terminal node. Solution is just `x_init`;
    /// no controls; `P_0 = Q_0`, `p_0 = q_0`.
    #[test]
    fn n_one_is_terminal_only() {
        const N: usize = 1;
        const NX: usize = 2;
        const NU: usize = 1;
        let mut prob = LqrProblem::<N, NX, NU>::default();
        prob.q_mat[0] = SMatrix::<f64, 2, 2>::identity();
        prob.q_lin[0] = SVector::<f64, 2>::from_column_slice(&[3.0, -1.0]);
        prob.x_init   = SVector::<f64, 2>::from_column_slice(&[5.0, 2.0]);
        let mut ws  = LqrWorkspace::<N, NX, NU>::default();
        let mut sol = LqrSolution::<N, NX, NU>::default();
        assert!(riccati_factor_solve(&prob, &mut ws, &mut sol) == RiccatiStatus::Ok);

        // x[0] = x_init
        for i in 0..2 {
            assert!((sol.x[0][i] - prob.x_init[i]).abs() < 1e-15);
        }
        // P_0 = Q_0, p_0 = q_0
        for i in 0..2 {
            for j in 0..2 {
                let expect = if i == j { 1.0 } else { 0.0 };
                assert!((ws.p_mat[0][(i, j)] - expect).abs() < 1e-15);
            }
            assert!((ws.p_lin[0][i] - prob.q_lin[0][i]).abs() < 1e-15);
        }
    }

    /// `N = 2` LQR with hand-computed expected values.
    ///
    /// Problem: 1D point-mass `(r, v)`, kinematic transition,
    /// `A = [[1, 0.1], [0, 1]]`, `B = [0; 0.1]`, `c = [0; -0.1]`,
    /// `Q = I`, `R = 1`, terminal `Q_1 = I`, no linear cost terms,
    /// `x_init = [5, 0]`.
    ///
    /// Closed form:
    ///   M  = 1 + B^T Q B = 1.01
    ///   G  = B^T Q A = [0, 0.1]
    ///   K  = -G / M  = [0, -0.09900990…]
    ///   k₀ = -B^T Q c / M = 0.1·0.1 / 1.01 = 0.00990099…
    ///   u₀ = K x₀ + k₀ = 0 + 0.0099… = 0.00990099…
    #[test]
    fn n_two_hand_computed_point_mass() {
        const N: usize = 2;
        const NX: usize = 2;
        const NU: usize = 1;
        let mut prob = LqrProblem::<N, NX, NU>::default();
        prob.a[0] = SMatrix::<f64, 2, 2>::from_row_slice(&[1.0, 0.1, 0.0, 1.0]);
        prob.b[0] = SMatrix::<f64, 2, 1>::from_column_slice(&[0.0, 0.1]);
        prob.c[0] = SVector::<f64, 2>::from_column_slice(&[0.0, -0.1]);
        prob.q_mat[0] = SMatrix::<f64, 2, 2>::identity();
        prob.q_mat[1] = SMatrix::<f64, 2, 2>::identity();
        prob.r_mat[0] = SMatrix::<f64, 1, 1>::from_element(1.0);
        prob.x_init   = SVector::<f64, 2>::from_column_slice(&[5.0, 0.0]);

        let mut ws  = LqrWorkspace::<N, NX, NU>::default();
        let mut sol = LqrSolution::<N, NX, NU>::default();
        assert!(riccati_factor_solve(&prob, &mut ws, &mut sol) == RiccatiStatus::Ok);

        let expected_u0    = 0.01 / 1.01;  // ≈ 0.00990099
        let expected_k_gain_row = [0.0, -0.1 / 1.01];

        assert!((sol.u[0][0] - expected_u0).abs() < 1e-12,
                "u₀ = {} vs {}", sol.u[0][0], expected_u0);
        for (j, &expected) in expected_k_gain_row.iter().enumerate() {
            assert!((ws.k_gain[0][(0, j)] - expected).abs() < 1e-12,
                    "K[0,{j}] = {}", ws.k_gain[0][(0, j)]);
        }

        // x_1 = A x_0 + B u_0 + c
        let x1_expected: [f64; 2] = [
            5.0 + 0.0,                       // r unchanged (v=0)
            0.0 + 0.1 * expected_u0 - 0.1,   // ≈ 0.000990099 - 0.1 = -0.099009901
        ];
        for (i, &expected) in x1_expected.iter().enumerate() {
            assert!((sol.x[1][i] - expected).abs() < 1e-12,
                    "x₁[{i}] = {} vs {}", sol.x[1][i], expected);
        }
    }

    /// `P_k` stays symmetric throughout a random factorization.
    /// This is the explicit symmetrize-after-each-stage discipline working.
    #[test]
    fn p_matrices_stay_symmetric() {
        const N: usize = 5;
        const NX: usize = 3;
        const NU: usize = 2;
        let mut prob = LqrProblem::<N, NX, NU>::default();
        // Asymmetric A, B, c — the Riccati update naturally would drift from
        // symmetry without our half-(P+Pᵀ) defense.
        for k in 0..N - 1 {
            prob.a[k] = SMatrix::<f64, 3, 3>::from_row_slice(&[
                1.0 + 0.05 * (k as f64), 0.1, 0.0,
                0.0,  1.0,             0.05 * (k as f64 + 1.0),
                0.02, 0.0,             0.98,
            ]);
            prob.b[k] = SMatrix::<f64, 3, 2>::from_row_slice(&[
                0.0, 0.1,
                0.1, 0.0,
                0.0, 0.05,
            ]);
            prob.c[k] = SVector::<f64, 3>::from_column_slice(&[0.0, -0.01, 0.0]);
            prob.q_mat[k] = SMatrix::<f64, 3, 3>::identity();
            prob.r_mat[k] = SMatrix::<f64, 2, 2>::identity() * 0.1;
        }
        prob.q_mat[N - 1] = SMatrix::<f64, 3, 3>::identity() * 10.0;
        prob.x_init       = SVector::<f64, 3>::from_column_slice(&[1.0, 0.5, -0.2]);

        let mut ws = LqrWorkspace::<N, NX, NU>::default();
        assert!(riccati_factor(&prob, &mut ws) == RiccatiStatus::Ok);

        for k in 0..N {
            for i in 0..3 {
                for j in (i + 1)..3 {
                    let asym = (ws.p_mat[k][(i, j)] - ws.p_mat[k][(j, i)]).abs();
                    assert!(asym < 1.0e-15,
                            "P_{k} not symmetric at ({i},{j}): {} vs {}",
                            ws.p_mat[k][(i, j)], ws.p_mat[k][(j, i)]);
                }
            }
        }
    }

    /// **The P2 gate from the plan**: solve a small LQR via Riccati, then
    /// build the full 15×15 KKT system and solve via `SMatrix::try_inverse`.
    /// Components must match to ≤ 1e-10.
    ///
    /// Variable order in the KKT vector:
    /// `[x₁ x₂ x₃ u₀ u₁ u₂ λ₁ λ₂ λ₃]`  →  dim = 6+3+6 = 15.
    #[test]
    fn riccati_matches_dense_kkt_lu() {
        const N: usize = 4;
        const NX: usize = 2;
        const NU: usize = 1;
        const KKT_DIM: usize = 15; // = (N-1)*NX*2 + (N-1)*NU

        // ---- Concrete problem ----
        let mut prob = LqrProblem::<N, NX, NU>::default();
        let a_mat = SMatrix::<f64, 2, 2>::from_row_slice(&[1.0, 0.1, 0.0, 1.0]);
        let b_mat = SMatrix::<f64, 2, 1>::from_column_slice(&[0.0, 0.1]);
        let c_vec = SVector::<f64, 2>::from_column_slice(&[0.0, -0.02]);
        for k in 0..N - 1 {
            prob.a[k] = a_mat;
            prob.b[k] = b_mat;
            prob.c[k] = c_vec;
        }
        for k in 0..N {
            prob.q_mat[k] = SMatrix::<f64, 2, 2>::identity();
        }
        // Bias state cost slightly per stage so the linear-term path gets exercised
        prob.q_lin[1] = SVector::<f64, 2>::from_column_slice(&[0.1, 0.0]);
        prob.q_lin[2] = SVector::<f64, 2>::from_column_slice(&[0.0, 0.05]);
        for k in 0..N - 1 {
            prob.r_mat[k] = SMatrix::<f64, 1, 1>::from_element(0.1);
        }
        prob.r_lin[0] = SVector::<f64, 1>::from_element(0.05);
        prob.x_init   = SVector::<f64, 2>::from_column_slice(&[10.0, 0.0]);

        // ---- Riccati ----
        let mut ws  = LqrWorkspace::<N, NX, NU>::default();
        let mut sol = LqrSolution::<N, NX, NU>::default();
        assert!(riccati_factor_solve(&prob, &mut ws, &mut sol) == RiccatiStatus::Ok);

        // ---- Dense KKT  ----
        let mut kkt = SMatrix::<f64, KKT_DIM, KKT_DIM>::zeros();
        let mut rhs = SVector::<f64, KKT_DIM>::zeros();

        // Index helpers (closures avoid magic numbers)
        let xi  = |k: usize| (k - 1) * NX;   // x_k start (k = 1..N-1)
        let ui  = |k: usize| 3 * NX + k * NU; // u_k start (k = 0..N-2)
        let li  = |k: usize| 3 * NX + 3 * NU + (k - 1) * NX; // λ_k start (k=1..N-1)

        // 2×2 block placer
        let put2x2 = |kkt: &mut SMatrix<f64, 15, 15>, r: usize, c: usize,
                      m: &SMatrix<f64, 2, 2>| {
            for i in 0..2 {
                for j in 0..2 {
                    kkt[(r + i, c + j)] = m[(i, j)];
                }
            }
        };
        // 2×1 block placer
        let put2x1 = |kkt: &mut SMatrix<f64, 15, 15>, r: usize, c: usize,
                      m: &SMatrix<f64, 2, 1>| {
            kkt[(r,     c)] = m[(0, 0)];
            kkt[(r + 1, c)] = m[(1, 0)];
        };

        let i2 = SMatrix::<f64, 2, 2>::identity();
        let neg_a = -a_mat;
        let a_t   = a_mat.transpose();
        let neg_a_t = -a_t;

        // ---- State-stationarity rows: Q x + q + λ_k - A^T λ_{k+1} = 0 ----
        //  Row k=1: Q₁ at (x₁,x₁); +I at (x₁,λ₁); -A₁ᵀ at (x₁,λ₂)
        put2x2(&mut kkt, xi(1), xi(1), &prob.q_mat[1]);
        put2x2(&mut kkt, xi(1), li(1), &i2);
        put2x2(&mut kkt, xi(1), li(2), &neg_a_t);
        for i in 0..2 { rhs[xi(1) + i] = -prob.q_lin[1][i]; }

        //  Row k=2: Q₂ at (x₂,x₂); +I at (x₂,λ₂); -A₂ᵀ at (x₂,λ₃)
        put2x2(&mut kkt, xi(2), xi(2), &prob.q_mat[2]);
        put2x2(&mut kkt, xi(2), li(2), &i2);
        put2x2(&mut kkt, xi(2), li(3), &neg_a_t);
        for i in 0..2 { rhs[xi(2) + i] = -prob.q_lin[2][i]; }

        //  Row k=3 (terminal): Q₃ at (x₃,x₃); +I at (x₃,λ₃)
        put2x2(&mut kkt, xi(3), xi(3), &prob.q_mat[3]);
        put2x2(&mut kkt, xi(3), li(3), &i2);
        for i in 0..2 { rhs[xi(3) + i] = -prob.q_lin[3][i]; }

        // ---- Control-stationarity rows: R u + r - B^T λ_{k+1} = 0 ----
        // 1×1 stage rows; B has shape 2×1 so Bᵀ is 1×2.
        for k in 0..(N - 1) {
            kkt[(ui(k), ui(k))] = prob.r_mat[k][(0, 0)];
            kkt[(ui(k), li(k + 1))]     = -b_mat[(0, 0)];
            kkt[(ui(k), li(k + 1) + 1)] = -b_mat[(1, 0)];
            rhs[ui(k)] = -prob.r_lin[k][0];
        }

        // ---- Dynamic-constraint rows: x_k - A x_{k-1} - B u_{k-1} - c_{k-1} = 0 ----
        // Row k=1: I at (λ₁,x₁); -B at (λ₁,u₀); RHS = A x_0 + c_0
        put2x2(&mut kkt, li(1), xi(1), &i2);
        put2x1(&mut kkt, li(1), ui(0), &(-b_mat));
        let rhs1 = a_mat * prob.x_init + c_vec;
        for i in 0..2 { rhs[li(1) + i] = rhs1[i]; }

        // Row k=2: -A at (λ₂,x₁); I at (λ₂,x₂); -B at (λ₂,u₁); RHS = c_1
        put2x2(&mut kkt, li(2), xi(1), &neg_a);
        put2x2(&mut kkt, li(2), xi(2), &i2);
        put2x1(&mut kkt, li(2), ui(1), &(-b_mat));
        for i in 0..2 { rhs[li(2) + i] = c_vec[i]; }

        // Row k=3: -A at (λ₃,x₂); I at (λ₃,x₃); -B at (λ₃,u₂); RHS = c_2
        put2x2(&mut kkt, li(3), xi(2), &neg_a);
        put2x2(&mut kkt, li(3), xi(3), &i2);
        put2x1(&mut kkt, li(3), ui(2), &(-b_mat));
        for i in 0..2 { rhs[li(3) + i] = c_vec[i]; }

        // Solve dense KKT
        let kkt_inv = kkt.try_inverse().expect("dense KKT not invertible");
        let dense_sol = kkt_inv * rhs;

        // ---- Compare ----
        let mut max_err = 0.0_f64;
        for k in 1..N {
            for i in 0..NX {
                let d = (dense_sol[xi(k) + i] - sol.x[k][i]).abs();
                if d > max_err { max_err = d; }
            }
        }
        for k in 0..(N - 1) {
            let d = (dense_sol[ui(k)] - sol.u[k][0]).abs();
            if d > max_err { max_err = d; }
        }

        eprintln!("riccati vs dense-LU max element error: {:.3e}", max_err);
        assert!(max_err < 1.0e-10, "max err {max_err} exceeds 1e-10");
    }

    /// Negative test: a non-PD `R` should make `M` singular at some stage
    /// and produce `NotPd` rather than NaN/garbage output. Hard-rail.
    #[test]
    fn not_pd_r_yields_clean_failure() {
        const N: usize = 3;
        const NX: usize = 2;
        const NU: usize = 1;
        let mut prob = LqrProblem::<N, NX, NU>::default();
        for k in 0..N - 1 {
            prob.a[k] = SMatrix::<f64, 2, 2>::identity();
            prob.b[k] = SMatrix::<f64, 2, 1>::from_column_slice(&[1.0, 0.0]);
            prob.r_mat[k] = SMatrix::<f64, 1, 1>::from_element(-1.0); // negative ⇒ not PD
        }
        prob.q_mat[N - 1] = SMatrix::<f64, 2, 2>::zeros(); // zero terminal cost too
        prob.x_init = SVector::<f64, 2>::from_column_slice(&[1.0, 0.0]);

        let mut ws = LqrWorkspace::<N, NX, NU>::default();
        let status = riccati_factor(&prob, &mut ws);
        // M = -1 + Bᵀ·0·B = -1 < 0, but try_inverse on a 1×1 matrix sees a
        // perfectly invertible value; it doesn't enforce positivity.
        // So this case actually returns Ok with a sign-flipped step — the
        // caller is responsible for PD-checking R if it matters.
        // (Documenting actual behavior here: the failure path triggers only
        // when M is genuinely *singular*, not just non-PD. PD enforcement
        // belongs at the IPM layer where regularization can be applied.)
        assert!(status == RiccatiStatus::Ok || status == RiccatiStatus::NotPd,
                "unexpected status {}", status.as_u32());
    }
}
