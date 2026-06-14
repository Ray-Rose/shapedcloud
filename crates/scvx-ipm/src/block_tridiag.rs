//! Block-tridiagonal linear solver.
//!
//! Solves systems of the form
//! ```text
//!   | D_0   U_0                                  |   | x_0     |     | b_0     |
//!   | L_0   D_1   U_1                            |   | x_1     |     | b_1     |
//!   |       L_1   D_2   U_2                      |   | x_2     |  =  | b_2     |
//!   |               ...                          |   | ...     |     | ...     |
//!   |                  L_{n-2}  D_{n-1}          |   | x_{n-1} |     | b_{n-1} |
//! ```
//! where each block is `B × B` (`D_k`, `L_k`, `U_k`) or `B × 1` (`b_k`, `x_k`).
//!
//! Algorithm: **block-Thomas** (block-LU + forward/back substitution).
//! Cost: `O(N · B³)`. Memory: `O(N · B²)` (stored factors `D̃_k`, `Ũ_k`).
//!
//! **The whole point of this primitive**: the reduced-KKT Schur complement
//! `S = A·H⁻¹·Aᵀ` of an SCvx subproblem is block-tridiagonal, where the
//! block size is `NX` (the state dim) and the block count is roughly `N`
//! (the number of temporal nodes). The dense `try_inverse` path is
//! `O((N·NX)³)`; the block-tridiagonal path is `O(N·NX³)` — a flight WCET
//! must use the latter.
//!
//! `D_k` blocks are expected to be **symmetric PD** in normal use (they're
//! diagonal blocks of `A·H⁻¹·Aᵀ`). The factorization works as long as
//! `D̃_k = D_k - L_{k-1}·D̃_{k-1}⁻¹·U_{k-1}` is invertible at every stage.
//! Returns `BlockTridiagStatus::NotPd` on per-stage singularity.
//!
//! Per the design discipline: no `alloc`, no panic, const-generic over
//! `(N, B)`.

// `for k in 0..N` patterns dominate this module because we're walking
// per-stage arrays in lockstep with multiple sibling arrays (d, u, l,
// d_tilde, etc.). Rewriting as `iter().enumerate()` either requires
// multiple zip layers (less readable) or doesn't compose with the
// nalgebra const-generic types.
#![allow(clippy::needless_range_loop)]

use nalgebra::{SMatrix, SVector};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Block-tridiagonal linear system data.
///
/// `d[k]` is the `k`-th diagonal block (`B × B`). `u[k]` is the super-diagonal
/// block at row `k` column `k+1` (`B × B`), valid for `k = 0..N-1`. `l[k]` is
/// the sub-diagonal block at row `k+1` column `k` (`B × B`), valid for
/// `k = 0..N-1`. The last slot in `u` and `l` is unused.
pub struct BlockTridiag<const N: usize, const B: usize> {
    pub d: [SMatrix<f64, B, B>; N],
    pub u: [SMatrix<f64, B, B>; N],
    pub l: [SMatrix<f64, B, B>; N],
}

impl<const N: usize, const B: usize> Default for BlockTridiag<N, B> {
    fn default() -> Self {
        Self {
            d: [SMatrix::zeros(); N],
            u: [SMatrix::zeros(); N],
            l: [SMatrix::zeros(); N],
        }
    }
}

/// Stored factorization: the updated diagonal blocks `D̃_k = D_k − L_{k-1}·D̃_{k-1}⁻¹·U_{k-1}`,
/// their inverses (precomputed for the back-substitution), and the original
/// upper blocks (needed during back-sub).
pub struct BlockTridiagFactor<const N: usize, const B: usize> {
    pub d_tilde:     [SMatrix<f64, B, B>; N],
    pub d_tilde_inv: [SMatrix<f64, B, B>; N],
    pub u:           [SMatrix<f64, B, B>; N],
    pub l:           [SMatrix<f64, B, B>; N],
}

impl<const N: usize, const B: usize> Default for BlockTridiagFactor<N, B> {
    fn default() -> Self {
        Self {
            d_tilde:     [SMatrix::zeros(); N],
            d_tilde_inv: [SMatrix::zeros(); N],
            u:           [SMatrix::zeros(); N],
            l:           [SMatrix::zeros(); N],
        }
    }
}

/// Status of a block-tridiagonal factor/solve.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BlockTridiagStatus {
    Ok,
    /// Some `D̃_k` was non-invertible. Caller may regularize and retry.
    NotPd,
    /// `N < 1` — degenerate input.
    DegenerateInput,
}

impl BlockTridiagStatus {
    pub fn as_u32(self) -> u32 {
        match self {
            BlockTridiagStatus::Ok              => 0,
            BlockTridiagStatus::NotPd           => 1,
            BlockTridiagStatus::DegenerateInput => 2,
        }
    }
}

// ---------------------------------------------------------------------------
// Factor
// ---------------------------------------------------------------------------

/// Compute the block-LU factorization. Stores `D̃_k`, `D̃_k⁻¹`, and the original
/// `U_k`, `L_k` blocks in `factor` for use by [`block_tridiag_solve`].
///
/// **Block-Thomas recurrence** (forward elimination):
/// ```text
///   D̃_0 = D_0
///   D̃_k = D_k − L_{k-1} · D̃_{k-1}⁻¹ · U_{k-1}   for k = 1..N-1
/// ```
///
/// Returns `NotPd` if any `D̃_k` fails to invert.
pub fn block_tridiag_factor<const N: usize, const B: usize>(
    sys:    &BlockTridiag<N, B>,
    factor: &mut BlockTridiagFactor<N, B>,
) -> BlockTridiagStatus {
    if N < 1 {
        return BlockTridiagStatus::DegenerateInput;
    }

    // Stage 0: D̃_0 = D_0.
    factor.d_tilde[0] = sys.d[0];
    let d0_inv = match factor.d_tilde[0].try_inverse() {
        Some(m) => m,
        None    => return BlockTridiagStatus::NotPd,
    };
    factor.d_tilde_inv[0] = d0_inv;
    factor.u[0] = sys.u[0];
    factor.l[0] = sys.l[0];

    for k in 1..N {
        // D̃_k = D_k − L_{k-1} · D̃_{k-1}⁻¹ · U_{k-1}
        let lk_minus_1 = sys.l[k - 1];
        let uk_minus_1 = sys.u[k - 1];
        let d_tilde_k = sys.d[k]
            - lk_minus_1 * (factor.d_tilde_inv[k - 1] * uk_minus_1);
        let dk_inv = match d_tilde_k.try_inverse() {
            Some(m) => m,
            None    => return BlockTridiagStatus::NotPd,
        };
        factor.d_tilde    [k] = d_tilde_k;
        factor.d_tilde_inv[k] = dk_inv;
        factor.u          [k] = sys.u[k];
        factor.l          [k] = sys.l[k];
    }

    BlockTridiagStatus::Ok
}

// ---------------------------------------------------------------------------
// Solve
// ---------------------------------------------------------------------------

/// Solve `system · x = rhs` given a precomputed factorization.
///
/// Algorithm (forward then back substitution):
/// ```text
///   Forward:  b̃_0 = b_0
///             b̃_k = b_k − L_{k-1} · D̃_{k-1}⁻¹ · b̃_{k-1}   for k = 1..N-1
///   Back:     x_{N-1} = D̃_{N-1}⁻¹ · b̃_{N-1}
///             x_k     = D̃_k⁻¹ · (b̃_k − U_k · x_{k+1})    for k = N-2..0
/// ```
pub fn block_tridiag_solve<const N: usize, const B: usize>(
    factor: &BlockTridiagFactor<N, B>,
    rhs:    &[SVector<f64, B>; N],
    out:    &mut [SVector<f64, B>; N],
) {
    if N < 1 {
        return;
    }

    // Forward sub: reuse `out` as the b̃ buffer.
    out[0] = rhs[0];
    for k in 1..N {
        let solved_prev = factor.d_tilde_inv[k - 1] * out[k - 1];
        out[k] = rhs[k] - factor.l[k - 1] * solved_prev;
    }

    // Back sub.
    out[N - 1] = factor.d_tilde_inv[N - 1] * out[N - 1];
    if N == 1 {
        return;
    }
    for k in (0..(N - 1)).rev() {
        // x_k = D̃_k⁻¹ · (b̃_k − U_k · x_{k+1})
        let rhs_k = out[k] - factor.u[k] * out[k + 1];
        out[k]    = factor.d_tilde_inv[k] * rhs_k;
    }
}

/// Convenience: factor then solve in one call. Returns the factorization status.
pub fn block_tridiag_factor_solve<const N: usize, const B: usize>(
    sys:    &BlockTridiag<N, B>,
    factor: &mut BlockTridiagFactor<N, B>,
    rhs:    &[SVector<f64, B>; N],
    out:    &mut [SVector<f64, B>; N],
) -> BlockTridiagStatus {
    let status = block_tridiag_factor(sys, factor);
    if status == BlockTridiagStatus::Ok {
        block_tridiag_solve(factor, rhs, out);
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

    /// N=1 case: a single block. Solve reduces to `x_0 = D_0⁻¹·b_0`.
    #[test]
    fn n_one_is_simple_inverse() {
        const N: usize = 1;
        const B: usize = 3;
        let mut sys = BlockTridiag::<N, B>::default();
        sys.d[0] = SMatrix::<f64, 3, 3>::from_row_slice(&[
            4.0, 1.0, 0.0,
            1.0, 5.0, 2.0,
            0.0, 2.0, 6.0,
        ]);
        let rhs: [SVector<f64, 3>; 1] = [
            SVector::<f64, 3>::from_column_slice(&[1.0, 2.0, 3.0]),
        ];

        let mut factor = BlockTridiagFactor::<N, B>::default();
        let mut x:     [SVector<f64, 3>; 1] = [SVector::zeros()];
        assert_eq!(
            block_tridiag_factor_solve(&sys, &mut factor, &rhs, &mut x),
            BlockTridiagStatus::Ok,
        );

        // Verify D·x = b.
        let residual = sys.d[0] * x[0] - rhs[0];
        assert!(residual.norm() < 1.0e-13, "residual = {}", residual.norm());
    }

    /// N=2 case: 2 blocks with cross-coupling. Compare against direct
    /// 6×6 matrix solve.
    #[test]
    fn n_two_matches_dense_lu() {
        const N: usize = 2;
        const B: usize = 3;
        let mut sys = BlockTridiag::<N, B>::default();
        sys.d[0] = SMatrix::<f64, 3, 3>::from_row_slice(&[
            4.0, 1.0, 0.0,
            1.0, 5.0, 2.0,
            0.0, 2.0, 6.0,
        ]);
        sys.d[1] = SMatrix::<f64, 3, 3>::from_row_slice(&[
            7.0, 1.0, 1.0,
            1.0, 8.0, 2.0,
            1.0, 2.0, 9.0,
        ]);
        // Symmetric off-diagonal: U = Lᵀ (mirroring a Schur S = A·H⁻¹·Aᵀ).
        sys.u[0] = SMatrix::<f64, 3, 3>::from_row_slice(&[
            -1.0,  0.0,  0.0,
             0.0, -1.0,  0.0,
             0.0,  0.0, -1.0,
        ]);
        sys.l[0] = sys.u[0].transpose();

        let rhs: [SVector<f64, 3>; 2] = [
            SVector::<f64, 3>::from_column_slice(&[1.0, 0.0, 0.0]),
            SVector::<f64, 3>::from_column_slice(&[0.0, 1.0, 0.0]),
        ];
        let mut factor = BlockTridiagFactor::<N, B>::default();
        let mut x:     [SVector<f64, 3>; 2] = [SVector::zeros(); 2];
        assert_eq!(
            block_tridiag_factor_solve(&sys, &mut factor, &rhs, &mut x),
            BlockTridiagStatus::Ok,
        );

        // Build the dense 6×6 reference.
        let mut dense = SMatrix::<f64, 6, 6>::zeros();
        for i in 0..3 {
            for j in 0..3 {
                dense[(    i,     j)] = sys.d[0][(i, j)];
                dense[(3 + i, 3 + j)] = sys.d[1][(i, j)];
                dense[(    i, 3 + j)] = sys.u[0][(i, j)];
                dense[(3 + i,     j)] = sys.l[0][(i, j)];
            }
        }
        let mut b_dense = SVector::<f64, 6>::zeros();
        for i in 0..3 {
            b_dense[i]     = rhs[0][i];
            b_dense[3 + i] = rhs[1][i];
        }
        let x_dense = dense.try_inverse().unwrap() * b_dense;

        let mut max_err = 0.0_f64;
        for i in 0..3 {
            let e0 = (x[0][i] - x_dense[i]).abs();
            let e1 = (x[1][i] - x_dense[3 + i]).abs();
            if e0 > max_err { max_err = e0; }
            if e1 > max_err { max_err = e1; }
        }
        eprintln!("block-tridiag N=2 vs dense max err: {:.3e}", max_err);
        assert!(max_err < 1.0e-12, "max err {max_err} exceeds 1e-12");
    }

    /// **The P3b gate**: an `N=5`, `B=3` tridiagonal problem must match dense
    /// LU on a 15×15 system to ≤ 1e-10. Mirrors the
    /// `riccati_matches_dense_kkt_lu` pattern in `kkt.rs`.
    #[test]
    fn block_tridiag_matches_dense_lu() {
        const N: usize = 5;
        const B: usize = 3;
        const DIM: usize = N * B; // 15

        let mut sys = BlockTridiag::<N, B>::default();
        // Make diagonals PD and well-conditioned.
        for k in 0..N {
            let d = (k as f64) * 0.5;
            sys.d[k] = SMatrix::<f64, 3, 3>::from_row_slice(&[
                10.0 + d,  1.0,       0.5,
                 1.0,     11.0 + d,   0.5,
                 0.5,      0.5,      12.0 + d,
            ]);
        }
        // Off-diagonals: simple skew + identity-like, mirroring an `A·H⁻¹·Aᵀ`
        // Schur shape where the row coupling is from dynamics.
        for k in 0..(N - 1) {
            sys.u[k] = SMatrix::<f64, 3, 3>::from_row_slice(&[
                -1.0,  0.3,  0.0,
                 0.0, -1.0,  0.2,
                 0.1,  0.0, -1.0,
            ]);
            sys.l[k] = sys.u[k].transpose();  // Schur S is symmetric
        }

        let mut rhs: [SVector<f64, 3>; N] = [SVector::zeros(); N];
        for k in 0..N {
            rhs[k] = SVector::<f64, 3>::from_column_slice(&[
                1.0  + 0.1 * (k as f64),
                -0.3 + 0.2 * (k as f64),
                0.5  - 0.1 * (k as f64),
            ]);
        }

        let mut factor = BlockTridiagFactor::<N, B>::default();
        let mut x:     [SVector<f64, 3>; N] = [SVector::zeros(); N];
        assert_eq!(
            block_tridiag_factor_solve(&sys, &mut factor, &rhs, &mut x),
            BlockTridiagStatus::Ok,
        );

        // Build dense 15×15.
        let mut dense = SMatrix::<f64, DIM, DIM>::zeros();
        for k in 0..N {
            for i in 0..B {
                for j in 0..B {
                    dense[(k * B + i, k * B + j)] = sys.d[k][(i, j)];
                }
            }
        }
        for k in 0..(N - 1) {
            for i in 0..B {
                for j in 0..B {
                    dense[(    k     * B + i, (k + 1) * B + j)] = sys.u[k][(i, j)];
                    dense[((k + 1) * B + i,     k     * B + j)] = sys.l[k][(i, j)];
                }
            }
        }
        let mut b_dense = SVector::<f64, DIM>::zeros();
        for k in 0..N {
            for i in 0..B {
                b_dense[k * B + i] = rhs[k][i];
            }
        }
        let x_dense = dense.try_inverse().unwrap() * b_dense;

        let mut max_err = 0.0_f64;
        for k in 0..N {
            for i in 0..B {
                let d = (x[k][i] - x_dense[k * B + i]).abs();
                if d > max_err { max_err = d; }
            }
        }
        eprintln!("block-tridiag N=5 B=3 vs dense max err: {:.3e}", max_err);
        assert!(max_err < 1.0e-10, "max err {max_err} exceeds 1e-10");
    }

    /// Negative test: a deliberately singular `D_k` produces a clean `NotPd`
    /// (no panic, no NaN propagation).
    #[test]
    fn singular_d_yields_clean_failure() {
        const N: usize = 3;
        const B: usize = 2;
        let mut sys = BlockTridiag::<N, B>::default();
        sys.d[0] = SMatrix::<f64, 2, 2>::identity();
        // D_1 = 0 — guaranteed singular.
        sys.d[1] = SMatrix::<f64, 2, 2>::zeros();
        sys.d[2] = SMatrix::<f64, 2, 2>::identity();

        let mut factor = BlockTridiagFactor::<N, B>::default();
        let status = block_tridiag_factor(&sys, &mut factor);
        assert_eq!(status, BlockTridiagStatus::NotPd);
    }

    /// Solve linearity: `solve(L b1 + L b2) = L solve(b1) + solve(b2)`.
    /// (Helps catch off-by-one and forward/back-substitution bugs.)
    #[test]
    fn solver_is_linear() {
        const N: usize = 4;
        const B: usize = 3;
        let mut sys = BlockTridiag::<N, B>::default();
        for k in 0..N {
            sys.d[k] = SMatrix::<f64, 3, 3>::from_row_slice(&[
                5.0, 1.0, 0.0,
                1.0, 6.0, 0.0,
                0.0, 0.0, 7.0,
            ]);
        }
        for k in 0..(N - 1) {
            sys.u[k] = -SMatrix::<f64, 3, 3>::identity();
            sys.l[k] = -SMatrix::<f64, 3, 3>::identity();
        }
        let mut factor = BlockTridiagFactor::<N, B>::default();
        assert_eq!(block_tridiag_factor(&sys, &mut factor), BlockTridiagStatus::Ok);

        // Two distinct RHS.
        let mut b1: [SVector<f64, 3>; N] = [SVector::zeros(); N];
        let mut b2: [SVector<f64, 3>; N] = [SVector::zeros(); N];
        for k in 0..N {
            b1[k] = SVector::<f64, 3>::from_column_slice(&[1.0, 2.0,  3.0]);
            b2[k] = SVector::<f64, 3>::from_column_slice(&[0.5, -0.5, 1.0]);
        }
        let mut x1: [SVector<f64, 3>; N] = [SVector::zeros(); N];
        let mut x2: [SVector<f64, 3>; N] = [SVector::zeros(); N];
        block_tridiag_solve(&factor, &b1, &mut x1);
        block_tridiag_solve(&factor, &b2, &mut x2);

        let alpha = 3.0;
        let beta  = -2.0;
        let mut b_combined: [SVector<f64, 3>; N] = [SVector::zeros(); N];
        let mut x_combined: [SVector<f64, 3>; N] = [SVector::zeros(); N];
        for k in 0..N {
            b_combined[k] = b1[k] * alpha + b2[k] * beta;
        }
        block_tridiag_solve(&factor, &b_combined, &mut x_combined);

        // Verify x_combined = α·x1 + β·x2.
        let mut max_err = 0.0_f64;
        for k in 0..N {
            for i in 0..B {
                let expect = alpha * x1[k][i] + beta * x2[k][i];
                let d = (x_combined[k][i] - expect).abs();
                if d > max_err { max_err = d; }
            }
        }
        eprintln!("linearity check max err: {:.3e}", max_err);
        assert!(max_err < 1.0e-12, "linearity violated, max err {max_err}");
    }
}
