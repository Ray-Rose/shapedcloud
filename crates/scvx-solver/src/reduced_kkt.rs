//! SCvx-structured reduced-KKT solve via block-tridiagonal Schur.
//!
//! ## What this provides
//!
//! Given the SCvx subproblem's reduced KKT system at one IPM iterate
//! ```text
//!   [H   Aᵀ] [Δx]   [b_x]
//!   [A   0 ] [Δλ] = [b_a]
//! ```
//! where:
//! - `H = GᵀMG + reg·I` is the cone-scaled reduced Hessian (`M` block-diag
//!   per cone for AHO, `W²` for NT); **all SCvx cones are stage-local**, so
//!   `H` is exactly block-diagonal in the per-stage vars `z_k = (x_k, u_k,
//!   σ_k, ν_k, w_k)` of size `NZ = 19`.
//! - `A` is block-bidiagonal (dynamics couple stages `k-1` and `k`), plus
//!   the initial and terminal boundary rows.
//!
//! Then `S = A·H⁻¹·Aᵀ` is **block-tridiagonal**. This module:
//! 1. Extracts stage-wise `H_k` and dynamics blocks from a generic
//!    [`SocpProblem`] with SCvx layout.
//! 2. Builds the block-tridiagonal Schur via stage-local algebra
//!    (`O(N·NX³)`, not `O((N·NX)³)`).
//! 3. Factors + solves via [`scvx_ipm::block_tridiag`].
//! 4. Recovers `Δx` from `Δλ` stage-by-stage.
//!
//! ## Cost
//!
//! - Build `H_k` and per-stage products: `O(N · NZ²)` (cone rows × cones).
//! - Per-stage `H_k⁻¹`: `O(N · NZ³)`.
//! - Build Schur block-tridiag: `O(N · NX·NZ²)`.
//! - Factor + solve block-tridiag: `O(N · NX³)`.
//! - Recover Δx: `O(N · NZ·NX)`.
//!
//! Total: `O(N · NZ³)` where `NZ = NX + NU + 2 = 19` for fixed-tf SCvx.
//! Compare to dense `O((N·NZ)³)` — speedup grows as `N²`.
//!
//! ## Scope (fully wired — both time modes LANDED)
//!
//! - **Both fixed-tf and free-tf.** Free-tf's global `δτ` column makes `A`
//!   an "arrow" that breaks the strict block-bidiagonal structure; it is
//!   handled by a Sherman-Morrison rank-1 update on the Schur complement
//!   (the `*_free_tf` factor/solve pair below).
//! - Equality block structure assumes the canonical SCvx layout from
//!   [`crate::assemble::assemble_scvx_socp`] (initial, dynamics, terminal).
//! - **Wired into the IPM**: the structured drivers in
//!   [`crate::structured_socp`] call the `factor_*` / `solve_*_with_factor`
//!   pair here instead of a dense `h_mat.try_inverse()`. Per-step equivalence
//!   to the dense path is asserted to machine precision by the tests below.
//!
//! ## Verification
//!
//! The unit tests build a small SCvx subproblem, solve the reduced KKT
//! via this primitive and via dense `SMatrix::try_inverse`, and assert
//! componentwise equality to `≤ 1e-9`.

// `for k in 0..N` patterns dominate this module because we're walking
// per-stage arrays in lockstep with cone descriptors and dynamics blocks,
// and the stage index `k` is used to address multiple arrays + closures.
// Rewriting as `iter().enumerate()` hurts readability and forbids the
// closure captures we need.
#![allow(clippy::needless_range_loop)]

use nalgebra::{SMatrix, SVector};
use scvx_ipm::{ConeDesc, SocpProblem};

use crate::assemble::{NX, N_VARS_PER_NODE_SCVX};

/// Per-stage variable count: `(x, u, σ, ν, w) = 19`.
pub const NZ: usize = N_VARS_PER_NODE_SCVX;

/// Status of the SCvx reduced-KKT solve.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ReducedKktStatus {
    Ok,
    /// Per-stage `H_k` failed to invert.
    HNotPd,
    /// Schur block-tridiag factorization failed.
    SchurSingular,
    /// `N < 1` or other degenerate sizing.
    DegenerateInput,
}

impl ReducedKktStatus {
    pub fn as_u32(self) -> u32 {
        match self {
            ReducedKktStatus::Ok              => 0,
            ReducedKktStatus::HNotPd          => 1,
            ReducedKktStatus::SchurSingular   => 2,
            ReducedKktStatus::DegenerateInput => 3,
        }
    }
}

/// Per-stage indexing data for the SCvx layout. Cached once per IPM call;
/// caller passes the cone descs and the IPM scaling diagonals.
///
/// The cones list a `[ConeDesc; NCONES]` from the SOCP problem — this struct
/// just precomputes which **range of cone rows** touches stage `k`.
pub struct ScvxStageDesc<const NCONES: usize> {
    /// For each stage `k = 0..N-1`, the index range of cones in `prob.cones`
    /// that touch stage-k variables. SCvx has 8 cones per stage (no
    /// boundary cones in fixed-tf).
    pub cones_per_stage_lo: [usize; NCONES],
    pub cones_per_stage_hi: [usize; NCONES],
}

// ---------------------------------------------------------------------------
// H_k extraction
// ---------------------------------------------------------------------------

/// Build the per-stage diagonal block `H_k = (GᵀMG)_kk + reg·I` for stage
/// `k = 0..N-1`. `M` is the block-diagonal cone scaling (`arrow(s)⁻¹·arrow(y)`
/// for AHO, `W²` for NT) supplied per cone via the closure `m_block_for_cone`.
///
/// Implementation: for each cone `c` that touches stage `k`, accumulate
/// `(G_c)ᵀ · M_c · (G_c)` into the stage-k diagonal block.
///
/// **Why this is fast**: each cone touches `≤ NZ` columns in `G` (cones
/// are stage-local), and the per-cone matrix-product is `O(d²·NZ + d·NZ²)`
/// for a cone of dim `d`. Summing over 8 cones × N stages gives O(N·NZ³).
fn build_stage_h_blocks<
    const N: usize,
    const NP: usize,
    const NE: usize,
    const NCT: usize,
    const NCONES: usize,
>(
    prob:     &SocpProblem<NP, NE, NCT, NCONES>,
    m_diag:   &SVector<f64, NCT>,  // diagonal of M (block-diagonal cone scaling)
    reg:      f64,
    h_blocks: &mut [SMatrix<f64, NZ, NZ>; N],
) {
    debug_assert_eq!(NP, N * N_VARS_PER_NODE_SCVX,
                     "build_stage_h_blocks: NP must equal N · NZ");

    // For SCvx, M is supplied as its diagonal (since the IPM caller forms M
    // from per-cone scaling: AHO arrow(s)⁻¹·arrow(y) has nonzero off-diagonal
    // entries within a cone, but here we accept just the diagonal as an
    // approximation; a richer adapter would accept block-diagonal M).
    //
    // **Correctness note**: for the cone-row-scaled SCvx with NT scaling
    // (`M_c = W_c²`), each per-cone block IS dense (NT scaling is full
    // matrix). For AHO it's `arrow(s)⁻¹·arrow(y)`, also dense. So this
    // diagonal-only path is a simplification used for the standalone-test
    // case (verified against dense LU). Production wiring will need to
    // pass the full block-diagonal M.

    let zeros: SMatrix<f64, NZ, NZ> = SMatrix::zeros();
    for hb in h_blocks.iter_mut() { *hb = zeros; }

    // For each cone, add (G_c)ᵀ · M_c · (G_c) into the stage block.
    for cone in &prob.cones {
        let off = cone.offset;
        let d   = cone.dim;
        // Find which stage this cone belongs to (cone touches z_k iff its
        // G rows have nonzero columns in [k·NZ, (k+1)·NZ)). We detect by
        // scanning the first row of the cone's G slice for a nonzero col.
        let stage = match find_stage_for_cone::<N, NP, NE, NCT, NCONES>(prob, cone) {
            Some(s) => s,
            None    => continue,  // boundary cone (free-tf δτ) — skip
        };

        // Accumulate G_c row-by-row.
        // (G_c)ᵀ · M_c · (G_c) — diagonal M_c approximation: sum_i G_c[i,*]ᵀ · m_diag[off+i] · G_c[i,*]
        let col_lo = stage * NZ;
        for i in 0..d {
            let mi = m_diag[off + i];
            // Read G_c[i, col_lo..col_hi]
            let mut g_row = [0.0_f64; NZ];
            for c in 0..NZ {
                g_row[c] = prob.g_mat[(off + i, col_lo + c)];
            }
            // Outer product: H_k += mi · g_row · g_row^T
            for r in 0..NZ {
                if g_row[r] == 0.0 { continue; }
                let mr = mi * g_row[r];
                for c in 0..NZ {
                    h_blocks[stage][(r, c)] += mr * g_row[c];
                }
            }
        }
    }

    // Add regularization to every stage block's diagonal.
    for hb in h_blocks.iter_mut() {
        for i in 0..NZ {
            hb[(i, i)] += reg;
        }
    }
}

/// **Block-M variant**: accumulate `H_k = (G_k)ᵀ · M · (G_k)` where `M` is a
/// block-diagonal `NCT × NCT` matrix (per-cone dense blocks). This is the
/// path the IPM caller actually needs — both AHO (`M = arrow(s)⁻¹·arrow(y)`)
/// and NT (`M = W²`) produce block-dense per-cone scaling.
///
/// Note: passing the full `NCT × NCT` matrix is slightly wasteful (only
/// the per-cone diagonal blocks are nonzero), but `NCT ≤ 30·N` so for the
/// problem sizes we care about (`N ≤ 20`), `NCT² ≤ 360k` entries, well
/// under 3 MB of stack/heap. The build cost is `O(NCT²·NZ + NCT·NZ²)`
/// per stage — same `O(N·NZ³)` overall complexity.
fn build_stage_h_blocks_block_m<
    const N: usize,
    const NP: usize,
    const NE: usize,
    const NCT: usize,
    const NCONES: usize,
>(
    prob:     &SocpProblem<NP, NE, NCT, NCONES>,
    m_full:   &SMatrix<f64, NCT, NCT>,
    reg:      f64,
    h_blocks: &mut [SMatrix<f64, NZ, NZ>; N],
) {
    // Accepts both fixed-tf (NP = N·NZ) and free-tf (NP = N·NZ + 1)
    // layouts. The function reads only per-stage columns [0..N·NZ) — the
    // global δτ column (if present) is handled separately by the free-tf
    // caller via Sherman-Morrison.
    debug_assert!(NP == N * N_VARS_PER_NODE_SCVX
                  || NP == N * N_VARS_PER_NODE_SCVX + 1,
                  "build_stage_h_blocks_block_m: NP must equal N·NZ or N·NZ+1");

    let zeros: SMatrix<f64, NZ, NZ> = SMatrix::zeros();
    for hb in h_blocks.iter_mut() { *hb = zeros; }

    // For each cone, compute G_c (d × NZ), M_c (d × d), then
    //   H_k += G_c^T · M_c · G_c.
    for cone in &prob.cones {
        let off = cone.offset;
        let d   = cone.dim;
        let stage = match find_stage_for_cone::<N, NP, NE, NCT, NCONES>(prob, cone) {
            Some(s) => s,
            None    => continue,  // boundary cone (δτ) — skip
        };
        let col_lo = stage * NZ;

        // Compute g^T M g via two-stage inner product: first compute
        //   tmp[r, j] = Σ_i G[i, r] · M[i, j]    (for r ∈ NZ, j ∈ d)
        // then  H[r, c] += Σ_j tmp[r, j] · G[j, c].
        // (Hand-rolled to avoid materializing d×NZ submatrices via slices
        //  with awkward const-generic plumbing.)
        //
        // **Capacity contract**: `tmp[NZ][MAX_CONE_DIM_FOR_BLOCK_M]` is sized
        // for cone dims in the canonical SCvx layout `{1, 3, 4, 8, 11}`.
        // The max is 11 (trust-region cone: η + (x_k − x̄_k) + (u_k − ū_k)
        // for NX=7, NU=3 ⇒ 1 + 7 + 3 = 11). Any cone with `d > MAX` would
        // overflow `tmp[r][j]` silently in release — we forbid that case
        // outright: the cone is skipped (via `continue`) rather than
        // corrupting memory. Debug builds also assert the contract for
        // early diagnosis.
        const MAX_CONE_DIM_FOR_BLOCK_M: usize = 11;
        debug_assert!(d <= MAX_CONE_DIM_FOR_BLOCK_M,
                      "cone dim {d} exceeds MAX_CONE_DIM_FOR_BLOCK_M (={MAX_CONE_DIM_FOR_BLOCK_M})");
        if d > MAX_CONE_DIM_FOR_BLOCK_M {
            // Release-safe fallback: skip this cone's H contribution.
            // The caller (IPM) will likely fail to converge, but no UB.
            continue;
        }
        let mut tmp = [[0.0_f64; MAX_CONE_DIM_FOR_BLOCK_M]; NZ];
        for r in 0..NZ {
            for j in 0..d {
                let mut s = 0.0;
                for i in 0..d {
                    s += prob.g_mat[(off + i, col_lo + r)] * m_full[(off + i, off + j)];
                }
                tmp[r][j] = s;
            }
        }
        for r in 0..NZ {
            for c in 0..NZ {
                let mut s = 0.0;
                for j in 0..d {
                    s += tmp[r][j] * prob.g_mat[(off + j, col_lo + c)];
                }
                h_blocks[stage][(r, c)] += s;
            }
        }
    }

    // Regularization on diagonal.
    for hb in h_blocks.iter_mut() {
        for i in 0..NZ {
            hb[(i, i)] += reg;
        }
    }
}

/// Find the stage `k ∈ 0..N` that owns this cone (i.e., its G rows have
/// columns only in `[k·NZ, (k+1)·NZ)`). Returns `None` if the cone touches
/// variables outside the canonical SCvx per-stage layout (e.g., the global
/// `δτ` for free-tf).
///
/// **Comparison contract**: `prob.g_mat[(...)] == 0.0` is an exact-equality
/// check used only to detect the *sparsity pattern* (which columns a cone
/// touches), never a magnitude. This is sound because the only thing that
/// matters is **zero-ness**, and zero-ness is preserved along the entire
/// pipeline that runs before this function:
/// - [`crate::assemble::assemble_scvx_socp`] populates `g_mat` exclusively by
///   direct assignment of exact `0.0` (default) or exact non-zero constants
///   (`-1.0`, `+1.0`, `phys.cos_theta_max`, …); no accumulation.
/// - The upstream preconditioners ([`crate::precondition`]) then scale `g_mat`
///   by *positive* per-column / per-row diagonals. `0.0 · d == 0.0` and
///   `nonzero · positive` stays non-zero, so the sparsity pattern (hence the
///   stage detection) is invariant under preconditioning. The entries are now
///   fp-*products*, but their zero/non-zero classification is exact.
///
/// What would break this: a future transform that *subtracts/accumulates* into
/// `g_mat` (could cancel a structural non-zero to a tiny non-zero, or a
/// near-zero to exact zero), or a scale that underflows a true non-zero to a
/// subnormal/zero. If either is introduced, switch to `abs() > TINY` so the
/// detection keys on a small threshold rather than bit-exact zero.
fn find_stage_for_cone<
    const N: usize,
    const NP: usize,
    const NE: usize,
    const NCT: usize,
    const NCONES: usize,
>(
    prob: &SocpProblem<NP, NE, NCT, NCONES>,
    cone: &ConeDesc,
) -> Option<usize> {
    let off = cone.offset;
    let d   = cone.dim;
    let mut detected: Option<usize> = None;
    for i in 0..d {
        for col in 0..NP {
            if prob.g_mat[(off + i, col)] == 0.0 { continue; }
            let stage = col / NZ;
            if stage >= N { return None; }  // global var (δτ)
            match detected {
                None => detected = Some(stage),
                Some(s) if s == stage => {}
                _ => return None,  // cone touches multiple stages — not SCvx-local
            }
        }
    }
    detected
}

// ---------------------------------------------------------------------------
// Schur build
// ---------------------------------------------------------------------------

/// Per-stage dynamics block extracted from `prob.a_mat`. For a row block at
/// stage `k` (which constrains `x_{k+1}`), we collect:
/// - `C_k`: the NX × NZ block at (row_k, columns of stage k) — terms in `z_k`
/// - `D_k`: the NX × NZ block at (row_k, columns of stage k+1) — terms in `z_{k+1}`
///
/// Public to allow callers to hold a precomputed `ReducedKktFactor<N>` (which
/// stores `[DynamicsBlock; N]`) on their stack between `factor()` and
/// `solve_with_factor()` calls — the key enabler for the factor/apply split.
#[derive(Clone, Copy)]
pub struct DynamicsBlock {
    c_mat: SMatrix<f64, NX, NZ>,
    d_mat: SMatrix<f64, NX, NZ>,
}

/// Read the dynamics row blocks from `prob.a_mat`. The canonical SCvx layout
/// places dynamics rows at `[NX, NX + (N-1)·NX)`; each row block of size NX
/// touches stages `(k, k+1)`.
fn extract_dynamics_blocks<
    const N: usize,
    const NP: usize,
    const NE: usize,
    const NCT: usize,
    const NCONES: usize,
>(
    prob:   &SocpProblem<NP, NE, NCT, NCONES>,
    blocks: &mut [DynamicsBlock; N],  // we use only [0..N-1]
) {
    // Accepts both fixed-tf (NP = N·NZ) and free-tf (NP = N·NZ + 1) — only
    // reads per-stage columns [0..N·NZ).
    debug_assert!(NP == N * N_VARS_PER_NODE_SCVX
                  || NP == N * N_VARS_PER_NODE_SCVX + 1);

    let dyn_row_start = NX;
    for k in 0..(N - 1) {
        let row_lo = dyn_row_start + k * NX;
        let col_k_lo     = k * NZ;
        let col_kplus_lo = (k + 1) * NZ;
        for i in 0..NX {
            for j in 0..NZ {
                blocks[k].c_mat[(i, j)] = prob.a_mat[(row_lo + i, col_k_lo + j)];
                blocks[k].d_mat[(i, j)] = prob.a_mat[(row_lo + i, col_kplus_lo + j)];
            }
        }
    }
    // Last slot unused.
    blocks[N - 1].c_mat = SMatrix::zeros();
    blocks[N - 1].d_mat = SMatrix::zeros();
}

// ---------------------------------------------------------------------------
// Reduced KKT solve
// ---------------------------------------------------------------------------

/// Per-stage `Δz_k` and per-row-block `Δλ_k` returned by [`solve_reduced_kkt_scvx`].
///
/// `dz[k]` for `k = 0..N` carries stage `k`'s primal step.
/// `dlam_init` is the initial-condition multiplier (size NX).
/// `dlam_dyn[k]` for `k = 0..N-1` is the dynamics multiplier (size NX).
/// `dlam_term` is the terminal multiplier (size 6, the (r, v) constraint).
pub struct ReducedKktSolution<const N: usize> {
    pub dz:        [SVector<f64, NZ>; N],
    pub dlam_init: SVector<f64, NX>,
    pub dlam_dyn:  [SVector<f64, NX>; N],   // [0..N-1] used; last unused
    pub dlam_term: SVector<f64, 6>,
}

/// Static upper bound on the number of block-tridiag row blocks
/// (`N + 1` for SCvx: 1 initial + N-1 dynamics + 1 terminal). 64 covers
/// any flight-realistic N (≤63 stages).
pub const RB_MAX: usize = 64;

/// Cached factorization of the structured reduced KKT for an SCvx subproblem.
///
/// The factor data depends on `(prob, m_full, reg)` — i.e., the cone scaling
/// `M` and the regularization. It does **not** depend on the RHS `(b_x, b_a)`.
/// Building the factor once and applying it to multiple RHS vectors (the
/// IPM's predictor + corrector solves) gives a ~2× per-iter speedup.
///
/// Use [`factor_reduced_kkt_scvx_block_m`] to build, then
/// [`solve_reduced_kkt_scvx_with_factor`] to apply.
///
/// **Stack footprint**: roughly `N · (NZ² + 2·NX·NZ) · 8` bytes for the
/// per-stage data, plus `3 · RB_MAX · NX² · 8` bytes for the Schur tridiag
/// factor. For `N = 10`: ~70 KB; for `N = 20`: ~140 KB. Allocates entirely
/// on the caller's stack — no heap.
pub struct ReducedKktFactor<const N: usize> {
    /// Per-stage `H_k⁻¹` (NZ × NZ). Built by `factor_reduced_kkt_scvx_block_m`.
    pub h_inv:       [SMatrix<f64, NZ, NZ>; N],
    /// Dynamics blocks `(C_k, D_k)` extracted from `prob.a_mat`. Used by
    /// both factor + apply paths.
    pub dyn_blocks:  [DynamicsBlock; N],
    /// Schur D̃⁻¹ (the inverted forward-elim diagonal blocks).
    pub d_tilde_inv: [SMatrix<f64, NX, NX>; RB_MAX],
    /// Upper / lower off-diagonal Schur blocks (cached for back-sub).
    pub u_blocks:    [SMatrix<f64, NX, NX>; RB_MAX],
    pub l_blocks:    [SMatrix<f64, NX, NX>; RB_MAX],
    /// Number of row blocks actually used: `nrb = N + 1`.
    pub nrb:         usize,
}

impl<const N: usize> Default for ReducedKktFactor<N> {
    fn default() -> Self {
        Self {
            h_inv:       [SMatrix::zeros(); N],
            dyn_blocks:  [DynamicsBlock { c_mat: SMatrix::zeros(), d_mat: SMatrix::zeros() }; N],
            d_tilde_inv: [SMatrix::zeros(); RB_MAX],
            u_blocks:    [SMatrix::zeros(); RB_MAX],
            l_blocks:    [SMatrix::zeros(); RB_MAX],
            nrb:         0,
        }
    }
}

impl<const N: usize> Default for ReducedKktSolution<N> {
    fn default() -> Self {
        Self {
            dz:        [SVector::zeros(); N],
            dlam_init: SVector::zeros(),
            dlam_dyn:  [SVector::zeros(); N],
            dlam_term: SVector::zeros(),
        }
    }
}

/// **The Riccati-style structured reduced-KKT solve for SCvx**.
///
/// Solves
/// ```text
///   [H   Aᵀ] [Δz]   [b_x]
///   [A   0 ] [Δλ] = [b_a]
/// ```
/// for an SCvx-shaped problem **using block-tridiagonal Schur factorization**,
/// giving `O(N·NZ³)` cost vs dense `O((N·NZ)³)`.
///
/// `m_diag` is the diagonal of the IPM cone scaling `M` (one entry per cone
/// row of `G`). `reg` is the Tikhonov regularization added to `H`.
/// `b_x` and `b_a` are the RHS in stacked form (length `N·NZ` and `NE`).
///
/// **Limitations** (intentional for this iteration):
/// - Fixed-tf only (no `δτ` global variable).
/// - Diagonal-only cone scaling. For the full per-cone block-dense `M`
///   (the path AHO and NT actually produce), use
///   [`solve_reduced_kkt_scvx_block_m`].
/// - Returns `Ok` only if the per-stage `H_k` and Schur both factor cleanly.
///   IPM caller can regularize and retry on failure.
pub fn solve_reduced_kkt_scvx<
    const N: usize,
    const NP: usize,
    const NE: usize,
    const NCT: usize,
    const NCONES: usize,
>(
    prob:   &SocpProblem<NP, NE, NCT, NCONES>,
    m_diag: &SVector<f64, NCT>,
    reg:    f64,
    b_x:    &SVector<f64, NP>,
    b_a:    &SVector<f64, NE>,
    out:    &mut ReducedKktSolution<N>,
) -> ReducedKktStatus {
    debug_assert_eq!(NP, N * N_VARS_PER_NODE_SCVX,
                     "solve_reduced_kkt_scvx: NP must equal N · NZ");
    debug_assert_eq!(NE, N * NX + 6,
                     "solve_reduced_kkt_scvx: NE must equal N · NX + 6");

    if N < 1 {
        return ReducedKktStatus::DegenerateInput;
    }

    // ---- Build per-stage H_k blocks and invert ----
    let mut h_blocks: [SMatrix<f64, NZ, NZ>; N] = [SMatrix::zeros(); N];
    build_stage_h_blocks::<N, NP, NE, NCT, NCONES>(prob, m_diag, reg, &mut h_blocks);

    let mut h_inv: [SMatrix<f64, NZ, NZ>; N] = [SMatrix::zeros(); N];
    for k in 0..N {
        h_inv[k] = match h_blocks[k].try_inverse() {
            Some(m) => m,
            None    => return ReducedKktStatus::HNotPd,
        };
    }

    // ---- Extract dynamics blocks (C_k, D_k) ----
    let mut dyn_blocks: [DynamicsBlock; N] = core::array::from_fn(|_| DynamicsBlock {
        c_mat: SMatrix::zeros(),
        d_mat: SMatrix::zeros(),
    });
    extract_dynamics_blocks::<N, NP, NE, NCT, NCONES>(prob, &mut dyn_blocks);

    // ---- Eliminate boundary multipliers analytically ----
    //
    // The full A_mat row layout is:
    //   row block 0           : initial state x_0 = b_init (NX rows, touches z_0[x])
    //   row blocks 1..N-1     : dynamics for x_{k+1} = ... (NX rows, touches z_{k-1}, z_k)
    //   row block N           : terminal r_{N-1}, v_{N-1} = b_term (6 rows, touches z_{N-1}[x[0..6]])
    //
    // Solve the full block-tridiagonal Schur using the SAME stage-wise
    // Schur block `S_k = A_k·H_k⁻¹·A_kᵀ` decomposition, with the initial
    // and terminal rows folded into stages 0 and N-1 respectively.
    //
    // To keep the block-tridiag uniform (block size NX), we'll handle the
    // boundary rows explicitly: stage 0's row-block in S = (initial) + (dyn 0 leading edge);
    // stage N-1's = (dyn N-2 trailing) + (terminal).
    //
    // Cleanest approach for this **standalone** primitive: build the FULL
    // dense Schur S and solve, but **using the per-stage H_k⁻¹ blocks**
    // (we still get O(N·NZ³) for the H_k inversion). This proves the
    // structured H factorization works; the block-tridiag step on S
    // optimizes the **next** level. (For a small problem the savings are
    // already in H; for large N the block-tridiag on S dominates.)
    //
    // To exercise the block-tridiag primitive on S here, we collapse all
    // rows into per-stage NX blocks by treating initial and terminal as
    // their own "stages" (NX-block and 6-block respectively); we pad the
    // 6-block to NX with identity and zero RHS so block-tridiag operates
    // uniformly.

    solve_via_block_tridiag::<N, NP, NE, NCT, NCONES>(
        prob, &h_inv, &dyn_blocks, b_x, b_a, out,
    )
}

/// Inner driver: build the block-tridiagonal Schur and solve.
///
/// **Row blocks** (each NX-sized after padding):
/// - Row 0:     initial-state row block (touches z_0)
/// - Row k:     dynamics for x_{k+1} (touches z_{k-1}, z_k), for k = 1..N
/// - Row N+1:   terminal (touches z_{N-1}[x[0..6]]); padded to NX
///
/// Total row blocks: N + 1. Each NX × NX. So our block-tridiag is N+1 blocks.
///
/// **Stage column blocks**:
/// - Col k (k = 0..N-1): primal step Δz_k
///
/// **Block-M variant** of [`solve_reduced_kkt_scvx`] — accepts a full
/// `NCT × NCT` matrix `m_full` for the cone scaling (block-diagonal in
/// the cone offsets, with each per-cone block dense). This is the path
/// that AHO (`M = arrow(s)⁻¹·arrow(y)`) and NT (`M = W²`) actually
/// produce.
///
/// Same algorithm and limitations as `solve_reduced_kkt_scvx`; just a
/// different H accumulator. Cost remains `O(N·NZ³)`.
pub fn solve_reduced_kkt_scvx_block_m<
    const N: usize,
    const NP: usize,
    const NE: usize,
    const NCT: usize,
    const NCONES: usize,
>(
    prob:   &SocpProblem<NP, NE, NCT, NCONES>,
    m_full: &SMatrix<f64, NCT, NCT>,
    reg:    f64,
    b_x:    &SVector<f64, NP>,
    b_a:    &SVector<f64, NE>,
    out:    &mut ReducedKktSolution<N>,
) -> ReducedKktStatus {
    debug_assert_eq!(NP, N * N_VARS_PER_NODE_SCVX,
                     "solve_reduced_kkt_scvx_block_m: NP must equal N · NZ");
    debug_assert_eq!(NE, N * NX + 6,
                     "solve_reduced_kkt_scvx_block_m: NE must equal N · NX + 6");

    if N < 1 {
        return ReducedKktStatus::DegenerateInput;
    }

    // ---- Build per-stage H_k via the block-M accumulator ----
    let mut h_blocks: [SMatrix<f64, NZ, NZ>; N] = [SMatrix::zeros(); N];
    build_stage_h_blocks_block_m::<N, NP, NE, NCT, NCONES>(prob, m_full, reg, &mut h_blocks);

    let mut h_inv: [SMatrix<f64, NZ, NZ>; N] = [SMatrix::zeros(); N];
    for k in 0..N {
        h_inv[k] = match h_blocks[k].try_inverse() {
            Some(m) => m,
            None    => return ReducedKktStatus::HNotPd,
        };
    }

    let mut dyn_blocks: [DynamicsBlock; N] = core::array::from_fn(|_| DynamicsBlock {
        c_mat: SMatrix::zeros(),
        d_mat: SMatrix::zeros(),
    });
    extract_dynamics_blocks::<N, NP, NE, NCT, NCONES>(prob, &mut dyn_blocks);

    solve_via_block_tridiag::<N, NP, NE, NCT, NCONES>(
        prob, &h_inv, &dyn_blocks, b_x, b_a, out,
    )
}

// ---------------------------------------------------------------------------
// Helpers used by both the one-shot path and the factor/apply split.
// ---------------------------------------------------------------------------

/// For row block `rb_idx ∈ 0..=N`, return the stage(s) it touches:
/// - rb 0     (initial): touches stage 0 only
/// - rb k     (k = 1..N-1, dynamics): touches stages (k-1, k)
/// - rb N     (terminal): touches stage N-1 only
#[inline]
fn stages_of_rb_static<const N: usize>(rb_idx: usize) -> (Option<usize>, Option<usize>) {
    if rb_idx == 0 {
        (Some(0), None)
    } else if rb_idx < N {
        (Some(rb_idx - 1), Some(rb_idx))
    } else {
        // terminal
        (Some(N - 1), None)
    }
}

/// Extract `A_rb[stage]`: the NX × NZ block of `A_mat` at row block `rb_idx`,
/// column block `stage`. Pads terminal (6 rows) to NX with zeros.
///
/// Internal helper for the structured solve. Mirrors the inline closure that
/// previously lived inside `solve_via_block_tridiag`.
fn get_a_block_static<
    const N: usize,
    const NP: usize,
    const NE: usize,
    const NCT: usize,
    const NCONES: usize,
>(
    prob:       &SocpProblem<NP, NE, NCT, NCONES>,
    dyn_blocks: &[DynamicsBlock; N],
    rb_idx:     usize,
    stage:      usize,
) -> SMatrix<f64, NX, NZ> {
    let mut m = SMatrix::<f64, NX, NZ>::zeros();
    let col_lo = stage * NZ;
    if rb_idx == 0 {
        // Initial state.
        if stage == 0 {
            for i in 0..NX {
                for j in 0..NZ {
                    m[(i, j)] = prob.a_mat[(i, col_lo + j)];
                }
            }
        }
    } else if rb_idx < N {
        // Dynamics block for x_{rb_idx}.
        if stage == rb_idx - 1 {
            m = dyn_blocks[rb_idx - 1].c_mat;
        } else if stage == rb_idx {
            m = dyn_blocks[rb_idx - 1].d_mat;
        }
    } else {
        // Terminal: read first 6 rows from a_mat directly; rows 6..NX are 0.
        let row_lo = NX + (N - 1) * NX;
        if stage == N - 1 {
            for i in 0..6 {
                for j in 0..NZ {
                    m[(i, j)] = prob.a_mat[(row_lo + i, col_lo + j)];
                }
            }
        }
    }
    m
}

// ---------------------------------------------------------------------------
// Factor / apply split (Phase 6.7)
// ---------------------------------------------------------------------------

/// **Build the structured-KKT factor** from `(prob, m_full, reg)`. The factor
/// captures all RHS-independent work (per-stage H_k inversion, dynamics
/// block extraction, Schur block-tridiag forward elimination). Use
/// [`solve_reduced_kkt_scvx_with_factor`] to apply the factor to any RHS.
///
/// Cost: `O(N · NZ³)` (per-stage H_k inversion dominates).
///
/// Returns `HNotPd` if any per-stage `H_k` is non-invertible, or
/// `SchurSingular` if any Schur D̃_k fails to invert during forward elim.
/// `DegenerateInput` if `N < 1` or `N + 1 > RB_MAX`.
///
/// **Same scope as [`solve_reduced_kkt_scvx_block_m`]**: fixed-tf SCvx
/// layout only.
pub fn factor_reduced_kkt_scvx_block_m<
    const N: usize,
    const NP: usize,
    const NE: usize,
    const NCT: usize,
    const NCONES: usize,
>(
    prob:   &SocpProblem<NP, NE, NCT, NCONES>,
    m_full: &SMatrix<f64, NCT, NCT>,
    reg:    f64,
    factor: &mut ReducedKktFactor<N>,
) -> ReducedKktStatus {
    // The fixed-tf SCvx layout has `NP = N·NZ`; the free-tf layout
    // appends a single global `δτ` column, so `NP = N·NZ + 1`. This
    // factor builds only the per-stage `H_k` (stages 0..N-1); it
    // ignores any global columns past index `N·NZ`. So the assert
    // accepts both layouts. The free-tf caller
    // (`factor_reduced_kkt_scvx_block_m_free_tf`) handles the global
    // `δτ` separately via Sherman-Morrison.
    debug_assert!(NP == N * N_VARS_PER_NODE_SCVX
                  || NP == N * N_VARS_PER_NODE_SCVX + 1,
                  "factor_reduced_kkt_scvx_block_m: NP must equal N·NZ or N·NZ+1");
    debug_assert_eq!(NE, N * NX + 6,
                     "factor_reduced_kkt_scvx_block_m: NE must equal N · NX + 6");

    if N < 1 {
        return ReducedKktStatus::DegenerateInput;
    }
    let nrb = N + 1;
    if nrb > RB_MAX {
        return ReducedKktStatus::DegenerateInput;
    }
    factor.nrb = nrb;

    // ---- Per-stage H_k and inverses ----
    let mut h_blocks: [SMatrix<f64, NZ, NZ>; N] = [SMatrix::zeros(); N];
    build_stage_h_blocks_block_m::<N, NP, NE, NCT, NCONES>(prob, m_full, reg, &mut h_blocks);
    for k in 0..N {
        factor.h_inv[k] = match h_blocks[k].try_inverse() {
            Some(m) => m,
            None    => return ReducedKktStatus::HNotPd,
        };
    }

    // ---- Dynamics blocks ----
    extract_dynamics_blocks::<N, NP, NE, NCT, NCONES>(prob, &mut factor.dyn_blocks);

    // ---- Schur diagonal + off-diagonal blocks ----
    let mut d_blocks = [SMatrix::<f64, NX, NX>::zeros(); RB_MAX];
    for rb in 0..nrb {
        let mut d = SMatrix::<f64, NX, NX>::zeros();
        let (s0, s1) = stages_of_rb_static::<N>(rb);
        if let Some(s) = s0 {
            let a = get_a_block_static::<N, NP, NE, NCT, NCONES>(prob, &factor.dyn_blocks, rb, s);
            d += a * (factor.h_inv[s] * a.transpose());
        }
        if let Some(s) = s1 {
            let a = get_a_block_static::<N, NP, NE, NCT, NCONES>(prob, &factor.dyn_blocks, rb, s);
            d += a * (factor.h_inv[s] * a.transpose());
        }
        // Pad terminal block (rows 6..NX) with identity so it stays invertible.
        if rb == N {
            for i in 6..NX {
                d[(i, i)] = 1.0;
            }
        }
        d_blocks[rb] = d;
    }
    // Off-diagonal Schur entries: U[rb] couples rb to rb+1 via their shared stage.
    for rb in 0..(nrb - 1) {
        let shared_stage = if rb == 0 { 0 } else { rb };
        let a_lo = get_a_block_static::<N, NP, NE, NCT, NCONES>(
            prob, &factor.dyn_blocks, rb,     shared_stage,
        );
        let a_hi = get_a_block_static::<N, NP, NE, NCT, NCONES>(
            prob, &factor.dyn_blocks, rb + 1, shared_stage,
        );
        factor.u_blocks[rb] = a_lo * (factor.h_inv[shared_stage] * a_hi.transpose());
        factor.l_blocks[rb] = factor.u_blocks[rb].transpose();
    }

    // ---- Forward elim: D̃_k = D_k − L·D̃⁻¹·U, store D̃⁻¹ ----
    // Local scratch (d_tilde itself is not needed after we invert; we
    // only keep d_tilde_inv in the factor).
    let mut d_tilde = [SMatrix::<f64, NX, NX>::zeros(); RB_MAX];
    d_tilde[0] = d_blocks[0];
    factor.d_tilde_inv[0] = match d_tilde[0].try_inverse() {
        Some(m) => m,
        None    => return ReducedKktStatus::SchurSingular,
    };
    for k in 1..nrb {
        let mm = factor.l_blocks[k - 1] * (factor.d_tilde_inv[k - 1] * factor.u_blocks[k - 1]);
        d_tilde[k] = d_blocks[k] - mm;
        factor.d_tilde_inv[k] = match d_tilde[k].try_inverse() {
            Some(im) => im,
            None     => return ReducedKktStatus::SchurSingular,
        };
    }

    ReducedKktStatus::Ok
}

/// **Apply a pre-built structured-KKT factor** to a specific `(b_x, b_a)`
/// RHS. Cheaper than `solve_reduced_kkt_scvx_block_m` because the factor
/// work (per-stage H_k inversion + Schur forward elim) is amortized across
/// multiple RHS solves.
///
/// Cost: `O(N · NX²)` for the back-sub plus `O(N · NX · NZ)` for the Δz
/// recovery — dominated by the latter; total `O(N · NX · NZ)` per apply.
///
/// **Contract**: `factor` must have been built by
/// [`factor_reduced_kkt_scvx_block_m`] on the **same** `prob`. Using a
/// factor built on a different problem produces garbage Δz.
pub fn solve_reduced_kkt_scvx_with_factor<
    const N:      usize,
    const NP:     usize,
    const NE:     usize,
    const NCT:    usize,
    const NCONES: usize,
>(
    prob:   &SocpProblem<NP, NE, NCT, NCONES>,
    factor: &ReducedKktFactor<N>,
    b_x:    &SVector<f64, NP>,
    b_a:    &SVector<f64, NE>,
    out:    &mut ReducedKktSolution<N>,
) -> ReducedKktStatus {
    debug_assert!(NP == N * N_VARS_PER_NODE_SCVX
                  || NP == N * N_VARS_PER_NODE_SCVX + 1,
                  "solve_reduced_kkt_scvx_with_factor: NP must equal N·NZ or N·NZ+1");
    debug_assert_eq!(NE, N * NX + 6,
                     "solve_reduced_kkt_scvx_with_factor: NE must equal N · NX + 6");

    if N < 1 {
        return ReducedKktStatus::DegenerateInput;
    }
    let nrb = factor.nrb;
    if nrb != N + 1 || nrb > RB_MAX {
        return ReducedKktStatus::DegenerateInput;
    }

    // ---- Build Schur RHS  d_rhs = A·H⁻¹·b_x − b_a  per row block ----
    let b_x_stage = |k: usize| -> SVector<f64, NZ> {
        let mut v = SVector::<f64, NZ>::zeros();
        for j in 0..NZ {
            v[j] = b_x[k * NZ + j];
        }
        v
    };
    let b_a_rb = |rb_idx: usize| -> SVector<f64, NX> {
        let mut v = SVector::<f64, NX>::zeros();
        if rb_idx == 0 {
            for i in 0..NX { v[i] = b_a[i]; }
        } else if rb_idx < N {
            let lo = NX + (rb_idx - 1) * NX;
            for i in 0..NX { v[i] = b_a[lo + i]; }
        } else {
            let lo = NX + (N - 1) * NX;
            for i in 0..6 { v[i] = b_a[lo + i]; }
        }
        v
    };
    let mut d_rhs = [SVector::<f64, NX>::zeros(); RB_MAX];
    for rb in 0..nrb {
        let mut r = SVector::<f64, NX>::zeros();
        let (s0, s1) = stages_of_rb_static::<N>(rb);
        if let Some(s) = s0 {
            let a = get_a_block_static::<N, NP, NE, NCT, NCONES>(
                prob, &factor.dyn_blocks, rb, s,
            );
            r += a * (factor.h_inv[s] * b_x_stage(s));
        }
        if let Some(s) = s1 {
            let a = get_a_block_static::<N, NP, NE, NCT, NCONES>(
                prob, &factor.dyn_blocks, rb, s,
            );
            r += a * (factor.h_inv[s] * b_x_stage(s));
        }
        r -= b_a_rb(rb);
        d_rhs[rb] = r;
    }

    // ---- Forward sub of RHS through the cached D̃⁻¹ / L factors ----
    let mut b_tilde = [SVector::<f64, NX>::zeros(); RB_MAX];
    b_tilde[0] = d_rhs[0];
    for k in 1..nrb {
        let bm = factor.l_blocks[k - 1] * (factor.d_tilde_inv[k - 1] * b_tilde[k - 1]);
        b_tilde[k] = d_rhs[k] - bm;
    }

    // ---- Back-sub: dlam[N] = D̃⁻¹·b̃[N], dlam[k] = D̃⁻¹·(b̃[k] − U·dlam[k+1]) ----
    let mut dlam = [SVector::<f64, NX>::zeros(); RB_MAX];
    dlam[nrb - 1] = factor.d_tilde_inv[nrb - 1] * b_tilde[nrb - 1];
    for k in (0..(nrb - 1)).rev() {
        let rhs = b_tilde[k] - factor.u_blocks[k] * dlam[k + 1];
        dlam[k] = factor.d_tilde_inv[k] * rhs;
    }

    // ---- Pack dlam into boundary-aware output ----
    out.dlam_init = dlam[0];
    if N >= 2 {
        out.dlam_dyn[..(N - 1)].copy_from_slice(&dlam[1..N]);
    }
    for i in 0..6 {
        out.dlam_term[i] = dlam[N][i];
    }

    // ---- Recover Δz_k = H⁻¹·(b_x_k − Σ_{rb} A_rbk^T · Δλ_rb) ----
    for k in 0..N {
        let mut acc = b_x_stage(k);
        for rb in 0..nrb {
            let (s0, s1) = stages_of_rb_static::<N>(rb);
            if s0 == Some(k) || s1 == Some(k) {
                let a = get_a_block_static::<N, NP, NE, NCT, NCONES>(
                    prob, &factor.dyn_blocks, rb, k,
                );
                acc -= a.transpose() * dlam[rb];
            }
        }
        out.dz[k] = factor.h_inv[k] * acc;
    }

    ReducedKktStatus::Ok
}

// ===========================================================================
// Free-tf via Sherman-Morrison (Phase 6.8)
// ===========================================================================
//
// **Math derivation**:
//
// The free-tf SCvx subproblem augments the primal vector with a global
// `δτ` scalar, giving `NP_free = N·NZ + 1`. The δτ variable couples to
// every dynamics row of `A` (via `−s_k·δτ` terms) but has NO cross-coupling
// to other primals in the Hessian `H` (because all cones touching δτ —
// the two bound cones — are δτ-only). So `H` decomposes as:
//
//   H = blkdiag(H_0, ..., H_{N-1}, H_δτ)
//
// where `H_δτ` is a 1×1 scalar (sum of bound-cone contributions + reg).
//
// `A`'s structure: same block-bidiagonal pattern in z columns, plus one
// dense δτ column with nonzeros only in dynamics row blocks:
//
//   A_δτ_rb_0     = 0          (initial)
//   A_δτ_rb_k     = −s_{k−1}    (dynamics for x_k, k = 1..N−1)
//   A_δτ_rb_N     = 0          (terminal)
//
// The Schur complement `S = A·H⁻¹·Aᵀ` splits as:
//
//   S = A_z·H_z⁻¹·A_zᵀ  +  (1/H_δτ)·a_δτ·a_δτᵀ
//     = S_tridiag         +  α·u·uᵀ          (rank-1 update)
//
// where `α = 1/H_δτ` and `u = a_δτ` (the δτ column of A, flattened over
// row blocks). `S_tridiag` is exactly the fixed-tf Schur — we reuse the
// existing factor for that.
//
// **Sherman-Morrison-Woodbury** to solve `S·Δλ = rhs`:
//
//   Δλ = S_tridiag⁻¹·rhs  −  [α·(uᵀ·S_tridiag⁻¹·rhs) / (1 + α·uᵀ·S_tridiag⁻¹·u)] · S_tridiag⁻¹·u
//
// We precompute `v = S_tridiag⁻¹·u` (once per iter) and `γ = α / (1 + α·uᵀ·v)`
// (the SMW scaling factor). Then each RHS solve is:
//
//   1. Solve `y₀ = S_tridiag⁻¹·rhs` via the existing block-tridiag back-sub.
//   2. Compute `Δλ = y₀ − γ·(uᵀ·y₀)·v`.
//
// **Primal recovery** with δτ:
//
//   Δz_k = H_k⁻¹·(b_x_k − Σ A_rbkᵀ·Δλ_rb)   (per stage, same as fixed-tf)
//   Δδτ  = H_δτ⁻¹·(b_x_δτ − Σ a_δτ_rb·Δλ_rb)  (the new piece)
//
// **RHS construction** for the Schur:
//
//   d_rhs_rb = A_z·H_z⁻¹·b_z_rb  +  a_δτ_rb·H_δτ⁻¹·b_x_δτ  −  b_a_rb

/// Extended factor for free-tf SCvx subproblems. Wraps the fixed-tf
/// [`ReducedKktFactor<N>`] with the additional data needed for the
/// Sherman-Morrison rank-1 correction:
/// - `h_inv_dtau`: scalar 1 / H_δτ
/// - `a_dtau`: δτ column of `A` packed by row block (NX-sized blocks, padded for terminal)
/// - `v_smw`: cached `S_tridiag⁻¹ · a_dtau` (the rank-1 direction)
/// - `gamma`: the SMW correction scaling `α / (1 + α·uᵀ·v)`
pub struct ReducedKktFactorFreeTf<const N: usize> {
    pub base:       ReducedKktFactor<N>,
    pub h_inv_dtau: f64,
    pub a_dtau:     [SVector<f64, NX>; RB_MAX],
    pub v_smw:      [SVector<f64, NX>; RB_MAX],
    pub gamma:      f64,
}

impl<const N: usize> Default for ReducedKktFactorFreeTf<N> {
    fn default() -> Self {
        Self {
            base:       ReducedKktFactor::default(),
            h_inv_dtau: 0.0,
            a_dtau:     [SVector::zeros(); RB_MAX],
            v_smw:      [SVector::zeros(); RB_MAX],
            gamma:      0.0,
        }
    }
}

/// Extended solution for free-tf SCvx. Wraps the fixed-tf
/// [`ReducedKktSolution<N>`] with the optimal δτ step.
pub struct ReducedKktSolutionFreeTf<const N: usize> {
    pub base:         ReducedKktSolution<N>,
    pub dz_delta_tau: f64,
}

impl<const N: usize> Default for ReducedKktSolutionFreeTf<N> {
    fn default() -> Self {
        Self {
            base:         ReducedKktSolution::default(),
            dz_delta_tau: 0.0,
        }
    }
}

/// Inner block-tridiag back-sub: solve `S_tridiag · y = rhs` given a factored
/// `ReducedKktFactor` and a stacked NX-block rhs. Used by the free-tf
/// apply path twice (once for the SMW correction direction, once for each
/// IPM RHS). Same algorithm as the body of `solve_reduced_kkt_scvx_with_factor`
/// but takes the rhs already packed by row block and returns just `Δλ`.
fn block_tridiag_back_sub<const N: usize>(
    factor: &ReducedKktFactor<N>,
    rhs:    &[SVector<f64, NX>; RB_MAX],
    out:    &mut [SVector<f64, NX>; RB_MAX],
) {
    let nrb = factor.nrb;
    // Forward sub of RHS.
    let mut b_tilde = [SVector::<f64, NX>::zeros(); RB_MAX];
    b_tilde[0] = rhs[0];
    for k in 1..nrb {
        let bm = factor.l_blocks[k - 1] * (factor.d_tilde_inv[k - 1] * b_tilde[k - 1]);
        b_tilde[k] = rhs[k] - bm;
    }
    // Back-sub.
    out[nrb - 1] = factor.d_tilde_inv[nrb - 1] * b_tilde[nrb - 1];
    for k in (0..(nrb - 1)).rev() {
        let rhs_k = b_tilde[k] - factor.u_blocks[k] * out[k + 1];
        out[k] = factor.d_tilde_inv[k] * rhs_k;
    }
}

/// **Build the structured-KKT factor for a free-tf SCvx subproblem**.
///
/// Same algorithm as [`factor_reduced_kkt_scvx_block_m`] for the per-stage
/// blocks, plus:
/// 1. Build `H_δτ` from the two δτ bound cones' M contributions
/// 2. Pack the δτ column of `A` into per-row-block NX-vectors (only
///    dynamics row blocks are nonzero; initial and terminal have zero)
/// 3. Solve `v = S_tridiag⁻¹·a_dτ` via the cached factor (one back-sub)
/// 4. Compute the SMW denominator `γ = α / (1 + α·uᵀ·v)`
///
/// **Layout contract**: `NP` MUST equal `N·NZ + 1`, `NCT` MUST equal
/// `N·N_CONE_DIM_PER_NODE_SCVX + 2`. Debug-asserted; release trusts caller.
pub fn factor_reduced_kkt_scvx_block_m_free_tf<
    const N: usize,
    const NP: usize,
    const NE: usize,
    const NCT: usize,
    const NCONES: usize,
>(
    prob:   &SocpProblem<NP, NE, NCT, NCONES>,
    m_full: &SMatrix<f64, NCT, NCT>,
    reg:    f64,
    factor: &mut ReducedKktFactorFreeTf<N>,
) -> ReducedKktStatus {
    debug_assert_eq!(NP, N * N_VARS_PER_NODE_SCVX + 1,
                     "factor_reduced_kkt_scvx_block_m_free_tf: NP must equal N·NZ + 1");
    debug_assert_eq!(NE, N * NX + 6,
                     "factor_reduced_kkt_scvx_block_m_free_tf: NE must equal N·NX + 6");

    if N < 1 {
        return ReducedKktStatus::DegenerateInput;
    }

    // ---- 1. Build the per-stage factor (same as fixed-tf) ----
    //
    // The fixed-tf `factor_reduced_kkt_scvx_block_m` builds H_k from cones
    // that `find_stage_for_cone` maps to a specific stage. Boundary cones
    // (δτ-only) are skipped by that helper — exactly what we want here:
    // their contributions go into H_δτ separately, not into any H_k.
    let st = factor_reduced_kkt_scvx_block_m::<N, NP, NE, NCT, NCONES>(
        prob, m_full, reg, &mut factor.base,
    );
    if st != ReducedKktStatus::Ok {
        return st;
    }

    // ---- 2. Compute H_δτ from δτ bound cones ----
    //
    // The two bound cones each contribute `(G_c column for δτ)² · M_c[0,0]`
    // to H_δτ (with G_c = ±1 for both bounds). Plus regularization.
    let dtau_idx = N * N_VARS_PER_NODE_SCVX;
    let mut h_dtau = 0.0_f64;
    for cone in &prob.cones {
        let off = cone.offset;
        let d   = cone.dim;
        // Determine if this cone touches the δτ column (and only δτ).
        //
        // **Intentional exhaustive column scan**: we scan all `NP` columns
        // rather than just `dtau_idx` so this stays correct even if a
        // future assemble path produces a cone that couples δτ to a stage
        // variable (which would disqualify it from the δτ-only H block).
        // For the canonical SCvx layout there are exactly two such cones
        // (the τ bound cones, each dim 1), so the cost is `O(2·NP)` ≈
        // `O(N)` per factor call — negligible against the `O(N·NZ³)` H
        // inversion. The `break 'col_scan` fast-fails on the first
        // non-δτ nonzero, so stage cones bail after one column hit.
        let mut touches_dtau_only = false;
        let mut other_touch = false;
        'col_scan: for i in 0..d {
            for col in 0..NP {
                if prob.g_mat[(off + i, col)] != 0.0 {
                    if col == dtau_idx {
                        touches_dtau_only = true;
                    } else {
                        other_touch = true;
                        break 'col_scan;
                    }
                }
            }
        }
        if touches_dtau_only && !other_touch {
            // Accumulate: H_δτ += g_c^T · M_c · g_c for the δτ direction.
            for i in 0..d {
                let g_i = prob.g_mat[(off + i, dtau_idx)];
                if g_i == 0.0 { continue; }
                for j in 0..d {
                    let g_j = prob.g_mat[(off + j, dtau_idx)];
                    if g_j == 0.0 { continue; }
                    h_dtau += g_i * m_full[(off + i, off + j)] * g_j;
                }
            }
        }
    }
    h_dtau += reg;
    if !h_dtau.is_finite() || h_dtau <= 0.0 {
        return ReducedKktStatus::HNotPd;
    }
    factor.h_inv_dtau = 1.0 / h_dtau;

    // ---- 3. Pack a_δτ column of A into per-row-block NX-vectors ----
    //
    // Row blocks:
    //   rb 0     (initial, rows 0..NX): a_δτ = 0
    //   rb k     (dynamics, rows NX+(k-1)·NX..NX+k·NX, k=1..N-1): from A column dtau_idx
    //   rb N     (terminal, rows N·NX..N·NX+6): a_δτ = 0
    let nrb = N + 1;
    for rb in 0..RB_MAX {
        factor.a_dtau[rb] = SVector::zeros();
    }
    for rb in 1..N {
        let row_lo = NX + (rb - 1) * NX;
        let mut v = SVector::<f64, NX>::zeros();
        for i in 0..NX {
            v[i] = prob.a_mat[(row_lo + i, dtau_idx)];
        }
        factor.a_dtau[rb] = v;
    }

    // ---- 4. Solve v = S_tridiag⁻¹ · a_δτ (cached for both predictor + corrector) ----
    block_tridiag_back_sub::<N>(&factor.base, &factor.a_dtau, &mut factor.v_smw);

    // ---- 5. Compute SMW denominator γ = α / (1 + α · uᵀ · v) ----
    //
    // u^T · v = Σ_rb a_dtau[rb] · v[rb]
    //
    // **PD structure note**: `v = S_tridiag⁻¹·u`, so `uᵀv = uᵀ·S_tridiag⁻¹·u`
    // is a quadratic form in the PD inverse `S_tridiag⁻¹` ⇒ `uᵀv ≥ 0`
    // exactly. With `α = 1/H_δτ > 0` (checked above), `denom = 1 + α·uᵀv`
    // is therefore `≥ 1` in exact arithmetic — the SMW update is always
    // well-posed for the canonical SCvx problem. The guards below are
    // defense-in-depth against round-off that nudges the *computed* `uᵀv`
    // slightly negative (a near-singular S_tridiag could do this), or a
    // pathological caller-supplied `M` that breaks PD-ness.
    let alpha = factor.h_inv_dtau;
    let mut utv = 0.0_f64;
    for rb in 0..nrb {
        utv += factor.a_dtau[rb].dot(&factor.v_smw[rb]);
    }
    let denom = 1.0 + alpha * utv;
    if !denom.is_finite() || denom.abs() < 1.0e-300 {
        return ReducedKktStatus::SchurSingular;
    }
    factor.gamma = alpha / denom;
    // Even though `denom` cleared the `> 1e-300` screen, a `denom` barely
    // above the threshold against a large `alpha` could push `gamma` to
    // ~1e300, which would overflow to ±∞ when multiplied into the SMW
    // correction `scale = γ·(uᵀy₀)`. Reject at the factor stage so the IPM
    // never applies a non-finite rank-1 correction. (The apply-path
    // step-finite guard would also catch it downstream, but failing at the
    // source gives a cleaner SchurSingular status than a generic
    // NumericalError.)
    if !factor.gamma.is_finite() {
        return ReducedKktStatus::SchurSingular;
    }

    ReducedKktStatus::Ok
}

/// **Apply a pre-built free-tf factor** to a specific `(b_x, b_a)` RHS.
///
/// `b_x` is the full primal RHS (length `N·NZ + 1`), with `b_x[N·NZ]` being
/// the δτ component. Recovers both `Δz_k` (per stage) and `Δδτ` (scalar).
///
/// **Algorithm**:
/// 1. Pack `b_x` per row block and construct the Schur RHS:
///    `d_rhs_rb = Σ_stages A_rb·H_k⁻¹·b_x_k + α·a_δτ_rb·b_x_δτ − b_a_rb`
/// 2. Solve `y₀ = S_tridiag⁻¹·d_rhs` via cached back-sub
/// 3. SMW correction: `Δλ = y₀ − γ·(a_δτᵀ·y₀)·v_smw`
/// 4. Per-stage recovery: `Δz_k = H_k⁻¹·(b_x_k − Σ A_rbkᵀ·Δλ_rb)`
/// 5. δτ recovery: `Δδτ = α·(b_x_δτ − Σ a_δτ_rb·Δλ_rb)`
pub fn solve_reduced_kkt_scvx_with_factor_free_tf<
    const N: usize,
    const NP: usize,
    const NE: usize,
    const NCT: usize,
    const NCONES: usize,
>(
    prob:   &SocpProblem<NP, NE, NCT, NCONES>,
    factor: &ReducedKktFactorFreeTf<N>,
    b_x:    &SVector<f64, NP>,
    b_a:    &SVector<f64, NE>,
    out:    &mut ReducedKktSolutionFreeTf<N>,
) -> ReducedKktStatus {
    debug_assert_eq!(NP, N * N_VARS_PER_NODE_SCVX + 1,
                     "solve_reduced_kkt_scvx_with_factor_free_tf: NP must equal N·NZ + 1");
    debug_assert_eq!(NE, N * NX + 6,
                     "solve_reduced_kkt_scvx_with_factor_free_tf: NE must equal N·NX + 6");

    if N < 1 {
        return ReducedKktStatus::DegenerateInput;
    }
    let nrb = factor.base.nrb;
    if nrb != N + 1 || nrb > RB_MAX {
        return ReducedKktStatus::DegenerateInput;
    }

    let dtau_idx = N * N_VARS_PER_NODE_SCVX;
    let b_x_dtau = b_x[dtau_idx];
    let alpha = factor.h_inv_dtau;

    // ---- Pack b_x per stage and build Schur RHS ----
    let b_x_stage = |k: usize| -> SVector<f64, NZ> {
        let mut v = SVector::<f64, NZ>::zeros();
        for j in 0..NZ {
            v[j] = b_x[k * NZ + j];
        }
        v
    };
    let b_a_rb = |rb_idx: usize| -> SVector<f64, NX> {
        let mut v = SVector::<f64, NX>::zeros();
        if rb_idx == 0 {
            for i in 0..NX { v[i] = b_a[i]; }
        } else if rb_idx < N {
            let lo = NX + (rb_idx - 1) * NX;
            for i in 0..NX { v[i] = b_a[lo + i]; }
        } else {
            let lo = NX + (N - 1) * NX;
            for i in 0..6 { v[i] = b_a[lo + i]; }
        }
        v
    };
    let mut d_rhs = [SVector::<f64, NX>::zeros(); RB_MAX];
    for rb in 0..nrb {
        let mut r = SVector::<f64, NX>::zeros();
        let (s0, s1) = stages_of_rb_static::<N>(rb);
        if let Some(s) = s0 {
            let a = get_a_block_static::<N, NP, NE, NCT, NCONES>(
                prob, &factor.base.dyn_blocks, rb, s,
            );
            r += a * (factor.base.h_inv[s] * b_x_stage(s));
        }
        if let Some(s) = s1 {
            let a = get_a_block_static::<N, NP, NE, NCT, NCONES>(
                prob, &factor.base.dyn_blocks, rb, s,
            );
            r += a * (factor.base.h_inv[s] * b_x_stage(s));
        }
        // δτ contribution: α · a_δτ_rb · b_x_δτ
        r += factor.a_dtau[rb] * (alpha * b_x_dtau);
        r -= b_a_rb(rb);
        d_rhs[rb] = r;
    }

    // ---- Solve y₀ = S_tridiag⁻¹·d_rhs (cached factor) ----
    let mut y0 = [SVector::<f64, NX>::zeros(); RB_MAX];
    block_tridiag_back_sub::<N>(&factor.base, &d_rhs, &mut y0);

    // ---- SMW correction: Δλ = y₀ − γ·(uᵀ·y₀)·v_smw ----
    let mut utv = 0.0_f64;
    for rb in 0..nrb {
        utv += factor.a_dtau[rb].dot(&y0[rb]);
    }
    let scale = factor.gamma * utv;
    let mut dlam = [SVector::<f64, NX>::zeros(); RB_MAX];
    for rb in 0..nrb {
        dlam[rb] = y0[rb] - factor.v_smw[rb] * scale;
    }

    // ---- Pack dlam into output ----
    out.base.dlam_init = dlam[0];
    if N >= 2 {
        out.base.dlam_dyn[..(N - 1)].copy_from_slice(&dlam[1..N]);
    }
    for i in 0..6 {
        out.base.dlam_term[i] = dlam[N][i];
    }

    // ---- Per-stage Δz_k recovery ----
    for k in 0..N {
        let mut acc = b_x_stage(k);
        for rb in 0..nrb {
            let (s0, s1) = stages_of_rb_static::<N>(rb);
            if s0 == Some(k) || s1 == Some(k) {
                let a = get_a_block_static::<N, NP, NE, NCT, NCONES>(
                    prob, &factor.base.dyn_blocks, rb, k,
                );
                acc -= a.transpose() * dlam[rb];
            }
        }
        out.base.dz[k] = factor.base.h_inv[k] * acc;
    }

    // ---- δτ recovery: Δδτ = α·(b_x_δτ − Σ a_δτ_rb·Δλ_rb) ----
    let mut at_dl = 0.0_f64;
    for rb in 0..nrb {
        at_dl += factor.a_dtau[rb].dot(&dlam[rb]);
    }
    out.dz_delta_tau = alpha * (b_x_dtau - at_dl);

    ReducedKktStatus::Ok
}

/// The Schur is on the **dual variables**, indexed by row block. The
/// primal Δz is recovered from Δλ in a final stage-wise pass.
#[allow(clippy::too_many_arguments)]
fn solve_via_block_tridiag<
    const N: usize,
    const NP: usize,
    const NE: usize,
    const NCT: usize,
    const NCONES: usize,
>(
    prob:       &SocpProblem<NP, NE, NCT, NCONES>,
    h_inv:      &[SMatrix<f64, NZ, NZ>; N],
    dyn_blocks: &[DynamicsBlock; N],
    b_x:        &SVector<f64, NP>,
    b_a:        &SVector<f64, NE>,
    out:        &mut ReducedKktSolution<N>,
) -> ReducedKktStatus {
    // The dual layout is:
    //   λ_init      (NX, row block 0)
    //   λ_dyn[k]    (NX, row block k+1 for k = 0..N-2)
    //   λ_term_pad  (NX, row block N — first 6 entries are terminal, last NX-6 = 1 are zero/identity-padded)
    //
    // Total row blocks: N+1, each NX × NX.

    const RB_NX: usize = NX;

    // We're using nalgebra const generics; we need a concrete const for the
    // block count. Wrap into a helper that takes N+1 as a const generic
    // via an intermediate function.
    //
    // To keep this self-contained without const-generic gymnastics, we
    // exploit the fact that N is known at the call site. We'll pin the
    // "padded terminal" as the (N+1)-th block of size NX.
    //
    // Rust doesn't let us write `N + 1` directly as a const generic param
    // through a single call chain without nightly features (`generic_const_exprs`).
    // The cleanest workaround: stop using a single block-tridiag-sized
    // const and instead build the Schur ourselves and call block-tridiag
    // with the next-up power-of-2-rounded N. But that wastes stack.
    //
    // For this PR, we hand-roll the block-Thomas inline since N is small
    // (we test at N=3) and avoid the const-generic awkwardness. Same
    // algorithm as block_tridiag_factor_solve, specialized to N+1 blocks
    // of size NX.

    let nrb = N + 1; // number of row blocks
    /// Static upper bound on the number of block-tridiag row blocks
    /// (`N + 1` for SCvx, with the +1 for the terminal-pad). 64 covers
    /// any flight-realistic N (≤63 stages — the project ships N≤10
    /// today, and the largest scale-up benchmark targets N≤20). If a
    /// caller passes a larger problem, the release-safe path below
    /// returns `DegenerateInput` rather than corrupting stack memory.
    const RB_MAX: usize = 64;
    debug_assert!(nrb <= RB_MAX, "N+1 exceeds RB_MAX in block-tridiag inline");
    if nrb > RB_MAX {
        // Release-safe: refuse to write out of bounds. The IPM caller
        // will see an error status and bail rather than corrupt the
        // stack-allocated work arrays below.
        return ReducedKktStatus::DegenerateInput;
    }

    // Schur diagonal blocks D_k = A_k·H⁻¹·A_kᵀ summed over stages.
    let mut d_blocks   = [SMatrix::<f64, RB_NX, RB_NX>::zeros(); RB_MAX];
    let mut u_blocks   = [SMatrix::<f64, RB_NX, RB_NX>::zeros(); RB_MAX]; // super-diag (rb k → rb k+1)
    let mut l_blocks   = [SMatrix::<f64, RB_NX, RB_NX>::zeros(); RB_MAX]; // sub-diag

    // For each row block, store the **row-block-times-stage-col** view of A_mat as
    // a NZ × NX matrix `aT_rb_at_stage[rb_idx][k]`. (Stage k's column block.)
    // Used both to build Schur AND to recover Δz from Δλ.
    // Build per stage on the fly to save memory.

    // ---- Build Schur D, U, L ----
    //
    // Each row block `i` has nonzero column overlap with at most TWO stages.
    // Determine these once (the SCvx layout pins them):
    //   rb 0 (initial)   → stages [0]
    //   rb k (k=1..N-1)  → stages [k-1, k]    (dynamics)
    //   rb N (terminal)  → stages [N-1]

    // Helper: extract A_rb[col_stage] : NX × NZ for row block `rb_idx` viewed at stage `s`.
    // We pad terminal to NX with rows 7..NX = 0 (terminal only has 6 rows).
    // Inline closures because of nested const generics.
    let get_a_block = |rb_idx: usize, stage: usize| -> SMatrix<f64, NX, NZ> {
        let mut m = SMatrix::<f64, NX, NZ>::zeros();
        let col_lo = stage * NZ;
        if rb_idx == 0 {
            // Initial state: NX rows starting at row 0, touches z_0.
            if stage == 0 {
                for i in 0..NX {
                    for j in 0..NZ {
                        m[(i, j)] = prob.a_mat[(i, col_lo + j)];
                    }
                }
            }
        } else if rb_idx < N {
            // Dynamics block for x_{rb_idx} = ...
            //   row range [NX + (rb_idx-1)·NX, NX + rb_idx·NX)
            //   stage rb_idx-1 (C) or rb_idx (D).
            let row_lo = NX + (rb_idx - 1) * NX;
            if stage == rb_idx - 1 {
                m = dyn_blocks[rb_idx - 1].c_mat;
                // Sanity check that it matches what we'd read from prob.a_mat
                debug_assert!({
                    let mut ok = true;
                    'outer: for i in 0..NX {
                        for j in 0..NZ {
                            if (prob.a_mat[(row_lo + i, col_lo + j)] - m[(i, j)]).abs() > 1e-14 {
                                ok = false;
                                break 'outer;
                            }
                        }
                    }
                    ok
                });
            } else if stage == rb_idx {
                m = dyn_blocks[rb_idx - 1].d_mat;
            }
            // Suppress unused warning when debug_assert is compiled out
            let _ = row_lo;
        } else {
            // Terminal (rb_idx == N): 6 rows at the end of A_mat, touches z_{N-1}.
            let row_lo = NX + (N - 1) * NX;  // start of terminal block
            if stage == N - 1 {
                for i in 0..6 {
                    for j in 0..NZ {
                        m[(i, j)] = prob.a_mat[(row_lo + i, col_lo + j)];
                    }
                }
                // Last NX-6 = 1 row is padding (zeros, so block-tridiag sees
                // an identity-padded block diagonal — we add the identity
                // below in the D block construction explicitly).
            }
        }
        m
    };

    // Compute D_blocks[rb] = Σ over stages it touches of  A_rb[stage] · H⁻¹_stage · A_rb[stage]ᵀ
    //         U_blocks[rb] = A_rb[shared_stage] · H⁻¹_shared · A_{rb+1}[shared_stage]ᵀ  (for consecutive blocks)
    //         L_blocks[rb] = U_blocks[rb]ᵀ  (symmetric Schur)

    // Helper: stages owned by row block rb_idx.
    let stages_of_rb = |rb_idx: usize| -> (Option<usize>, Option<usize>) {
        if rb_idx == 0 {
            (Some(0), None)
        } else if rb_idx < N {
            (Some(rb_idx - 1), Some(rb_idx))
        } else {
            // terminal
            (Some(N - 1), None)
        }
    };

    // For each row block, accumulate the diagonal Schur entry.
    for rb in 0..nrb {
        let mut d = SMatrix::<f64, NX, NX>::zeros();
        let (s0, s1) = stages_of_rb(rb);
        if let Some(s) = s0 {
            let a = get_a_block(rb, s);
            d += a * (h_inv[s] * a.transpose());
        }
        if let Some(s) = s1 {
            let a = get_a_block(rb, s);
            d += a * (h_inv[s] * a.transpose());
        }
        // For the terminal block, pad the unused row(s) with identity so the
        // block stays invertible. (Effectively: λ_term_pad[6..] are dummy
        // variables that don't appear in any equation; we add I to make D
        // PD, and the RHS for those rows is zero, so they stay zero.)
        if rb == N {
            for i in 6..NX {
                d[(i, i)] = 1.0;
            }
        }
        d_blocks[rb] = d;
    }

    // Off-diagonal: U_blocks[rb] couples rb to rb+1. They share at most one
    // stage. For canonical SCvx:
    //   rb 0 (initial, s=0) couples to rb 1 (dynamics 0, s=0 and s=1): shared s=0
    //   rb k (dyn, s=k-1,k) couples to rb k+1 (dyn or term, s=k,k+1): shared s=k
    //   rb N-1 couples to rb N (terminal, s=N-1): shared s=N-1
    for rb in 0..(nrb - 1) {
        let shared_stage = if rb == 0 { 0 } else { rb };
        let a_lo = get_a_block(rb,     shared_stage);
        let a_hi = get_a_block(rb + 1, shared_stage);
        u_blocks[rb] = a_lo * (h_inv[shared_stage] * a_hi.transpose());
        l_blocks[rb] = u_blocks[rb].transpose();
    }

    // ---- Build Schur RHS  d_rhs = A·H⁻¹·b_x − b_a ----
    // For each row block: r_rb = Σ_stages A_rb[stage] · H⁻¹_stage · b_x_stage  −  b_a_rb
    let mut d_rhs   = [SVector::<f64, RB_NX>::zeros(); RB_MAX];
    let b_x_stage = |k: usize| -> SVector<f64, NZ> {
        let mut v = SVector::<f64, NZ>::zeros();
        for j in 0..NZ {
            v[j] = b_x[k * NZ + j];
        }
        v
    };
    // Slice b_a per row block:
    //   rb 0: b_a[0..NX]
    //   rb k (1..=N-1): b_a[NX + (k-1)·NX .. NX + k·NX]
    //   rb N: b_a[N·NX .. N·NX + 6], padded zeros for rows 6..NX
    let b_a_rb = |rb_idx: usize| -> SVector<f64, NX> {
        let mut v = SVector::<f64, NX>::zeros();
        if rb_idx == 0 {
            for i in 0..NX { v[i] = b_a[i]; }
        } else if rb_idx < N {
            let lo = NX + (rb_idx - 1) * NX;
            for i in 0..NX { v[i] = b_a[lo + i]; }
        } else {
            let lo = NX + (N - 1) * NX;
            for i in 0..6 { v[i] = b_a[lo + i]; }
        }
        v
    };
    for rb in 0..nrb {
        let mut r = SVector::<f64, NX>::zeros();
        let (s0, s1) = stages_of_rb(rb);
        if let Some(s) = s0 {
            let a = get_a_block(rb, s);
            r += a * (h_inv[s] * b_x_stage(s));
        }
        if let Some(s) = s1 {
            let a = get_a_block(rb, s);
            r += a * (h_inv[s] * b_x_stage(s));
        }
        // Schur RHS:  A·H⁻¹·b_x  +  b_a  (sign matches the IPM convention
        // where the Schur step is  Δλ = S⁻¹·(A·H⁻¹·b_x − b_a)  for our b_a = -r_a;
        // here we pass `b_a` directly and the caller has already negated as needed.)
        r -= b_a_rb(rb);
        d_rhs[rb] = r;
    }

    // ---- Inline block-Thomas on (D, U, L) ----
    //
    // Use the same algorithm as block_tridiag_factor_solve but inline so we
    // can size by `nrb` at runtime within `RB_MAX`.

    let mut d_tilde     = [SMatrix::<f64, NX, NX>::zeros(); RB_MAX];
    let mut d_tilde_inv = [SMatrix::<f64, NX, NX>::zeros(); RB_MAX];
    let mut b_tilde     = [SVector::<f64, NX>::zeros();   RB_MAX];

    d_tilde[0] = d_blocks[0];
    d_tilde_inv[0] = match d_tilde[0].try_inverse() {
        Some(m) => m,
        None    => return ReducedKktStatus::SchurSingular,
    };
    b_tilde[0] = d_rhs[0];
    for k in 1..nrb {
        let m = l_blocks[k - 1] * (d_tilde_inv[k - 1] * u_blocks[k - 1]);
        d_tilde[k] = d_blocks[k] - m;
        d_tilde_inv[k] = match d_tilde[k].try_inverse() {
            Some(im) => im,
            None     => return ReducedKktStatus::SchurSingular,
        };
        let bm = l_blocks[k - 1] * (d_tilde_inv[k - 1] * b_tilde[k - 1]);
        b_tilde[k] = d_rhs[k] - bm;
    }

    let mut dlam = [SVector::<f64, NX>::zeros(); RB_MAX];
    dlam[nrb - 1] = d_tilde_inv[nrb - 1] * b_tilde[nrb - 1];
    for k in (0..(nrb - 1)).rev() {
        let rhs = b_tilde[k] - u_blocks[k] * dlam[k + 1];
        dlam[k] = d_tilde_inv[k] * rhs;
    }

    // ---- Pack dlam into the boundary-aware output ----
    out.dlam_init = dlam[0];
    if N >= 2 {
        out.dlam_dyn[..(N - 1)].copy_from_slice(&dlam[1..N]);
    }
    // Terminal: first 6 entries from dlam[N] (last NX-6 entries are padding).
    for i in 0..6 {
        out.dlam_term[i] = dlam[N][i];
    }

    // ---- Recover Δz_k = H⁻¹_k · (b_x_k − Σ_{rb} A_rbk^T · Δλ_rb) ----
    //
    // For each stage k, sum the contributions from all row blocks that
    // touch stage k:
    //   rb 0 touches stage 0 only.
    //   rb k (1..N-1) touches stages k-1 and k.
    //   rb N touches stage N-1 only.
    for k in 0..N {
        let mut acc = b_x_stage(k);
        // rb that contain stage k: { rb : k ∈ stages_of_rb(rb) }.
        //   k = 0: rb 0 (s=0) and rb 1 (s=0 via C-block)
        //   1 ≤ k ≤ N-2: rb k (s=k via D-block) and rb k+1 (s=k via C-block)
        //   k = N-1: rb N-1 (s=N-1 via D-block) and rb N (s=N-1 via boundary)
        for rb in 0..nrb {
            let (s0, s1) = stages_of_rb(rb);
            if s0 == Some(k) || s1 == Some(k) {
                let a = get_a_block(rb, k);
                acc -= a.transpose() * dlam[rb];
            }
        }
        out.dz[k] = h_inv[k] * acc;
    }

    ReducedKktStatus::Ok
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    extern crate std;
    use std::eprintln;

    use super::*;
    use crate::assemble::{assemble_scvx_socp, TerminalCondition, N_CONE_DIM_PER_NODE_SCVX};
    use nalgebra::{SMatrix, SVector};
    use scvx_core::{PhysicalParams, Trajectory};
    use scvx_dynamics::discretize_foh;

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

    /// **The P3b structural test**: build an SCvx subproblem with `N=3`,
    /// solve the reduced KKT via the **structured Riccati-style block
    /// tridiagonal solver**, and via dense `SMatrix::try_inverse` on the
    /// full augmented KKT, and assert componentwise equality to ≤ 1e-9.
    ///
    /// This is the gate that proves the structured solver matches the
    /// canonical dense LU oracle, mirroring `riccati_matches_dense_kkt_lu`
    /// in `kkt.rs`.
    #[test]
    fn structured_reduced_kkt_matches_dense_lu() {
        const N: usize = 3;
        const NP: usize = N * N_VARS_PER_NODE_SCVX;        // 57
        const NE: usize = N * NX + 6;                       // 27
        const NCT: usize = N * N_CONE_DIM_PER_NODE_SCVX;   // 90
        const NCONES: usize = N * 8;                        // 24

        let phys = mars_params();
        let mut x_init = SVector::<f64, 7>::zeros();
        x_init[0] = 100.0;
        x_init[2] = 1000.0;     // 1 km altitude
        x_init[5] = -50.0;      // descending
        x_init[6] = libm::log(800.0);  // log-mass

        let traj = hover_reference::<N>(x_init, 800.0, 12.0);
        let mut lin = scvx_dynamics::LinearizedDynamics::<N>::default();
        discretize_foh(&traj, &phys, &mut lin, 4);

        let term = TerminalCondition { r: [0.0, 0.0, 0.0], v: [0.0, 0.0, 0.0] };
        let mut prob: SocpProblem<NP, NE, NCT, NCONES> = SocpProblem::default();
        assemble_scvx_socp::<N, NP, NE, NCT, NCONES>(
            &traj, &lin, &phys, &x_init, &term,
            /*trust_eta=*/ 1.0e3,
            /*virt_weight=*/ 1.0e4,
            /*use_free_tf=*/ false,
            &mut prob,
        );

        // For this test, M = identity (no IPM scaling) and reg = small.
        // The structured solver should match dense regardless of M.
        let m_diag = SVector::<f64, NCT>::from_element(1.0);
        let reg: f64 = 1.0e-6;

        // Build a synthetic RHS.
        let mut b_x = SVector::<f64, NP>::zeros();
        for i in 0..NP {
            b_x[i] = libm::sin(0.13 * (i as f64 + 1.0));
        }
        let mut b_a = SVector::<f64, NE>::zeros();
        for i in 0..NE {
            b_a[i] = libm::cos(0.07 * (i as f64 + 1.0));
        }

        // ---- Structured solve ----
        let mut sol = ReducedKktSolution::<N>::default();
        let status = solve_reduced_kkt_scvx::<N, NP, NE, NCT, NCONES>(
            &prob, &m_diag, reg, &b_x, &b_a, &mut sol,
        );
        assert_eq!(status, ReducedKktStatus::Ok, "structured solve failed");

        // ---- Dense oracle ----
        // Build H = (G^T·diag(M)·G) + reg·I  (NP × NP)
        let mut h_dense = SMatrix::<f64, NP, NP>::zeros();
        for i in 0..NCT {
            for c1 in 0..NP {
                let g1 = prob.g_mat[(i, c1)];
                if g1 == 0.0 { continue; }
                let weighted = m_diag[i] * g1;
                for c2 in 0..NP {
                    let g2 = prob.g_mat[(i, c2)];
                    if g2 == 0.0 { continue; }
                    h_dense[(c1, c2)] += weighted * g2;
                }
            }
        }
        for i in 0..NP {
            h_dense[(i, i)] += reg;
        }

        // Full KKT (NP+NE) × (NP+NE).
        const DIM: usize = NP + NE;
        let mut k_dense = SMatrix::<f64, DIM, DIM>::zeros();
        for i in 0..NP {
            for j in 0..NP {
                k_dense[(i, j)] = h_dense[(i, j)];
            }
        }
        for i in 0..NE {
            for j in 0..NP {
                k_dense[(NP + i, j)] = prob.a_mat[(i, j)];
                k_dense[(j, NP + i)] = prob.a_mat[(i, j)];
            }
        }
        let mut rhs_dense = SVector::<f64, DIM>::zeros();
        for i in 0..NP { rhs_dense[i]      = b_x[i]; }
        for i in 0..NE { rhs_dense[NP + i] = b_a[i]; }

        let k_inv = k_dense.try_inverse().expect("dense KKT not invertible");
        let sol_dense = k_inv * rhs_dense;

        // ---- Compare ----
        let mut max_dz_err = 0.0_f64;
        for k in 0..N {
            for j in 0..NZ {
                let d = (sol.dz[k][j] - sol_dense[k * NZ + j]).abs();
                if d > max_dz_err { max_dz_err = d; }
            }
        }
        eprintln!("structured-vs-dense max Δz err: {:.3e}", max_dz_err);
        assert!(max_dz_err < 1.0e-9, "max Δz err {max_dz_err} exceeds 1e-9");

        // Also spot-check the multipliers.
        let mut max_dl_err = 0.0_f64;
        for i in 0..NX {
            let d = (sol.dlam_init[i] - sol_dense[NP + i]).abs();
            if d > max_dl_err { max_dl_err = d; }
        }
        for k in 0..(N - 1) {
            for i in 0..NX {
                let d = (sol.dlam_dyn[k][i] - sol_dense[NP + NX + k * NX + i]).abs();
                if d > max_dl_err { max_dl_err = d; }
            }
        }
        for i in 0..6 {
            let d = (sol.dlam_term[i] - sol_dense[NP + N * NX + i]).abs();
            if d > max_dl_err { max_dl_err = d; }
        }
        eprintln!("structured-vs-dense max Δλ err: {:.3e}", max_dl_err);
        assert!(max_dl_err < 1.0e-7, "max Δλ err {max_dl_err} exceeds 1e-7");
    }

    /// Stress with a slightly larger problem (N=5) to verify the Schur
    /// block-tridiagonal recursion stays accurate over more stages.
    #[test]
    fn structured_reduced_kkt_n5_matches_dense() {
        const N: usize = 5;
        const NP: usize = N * N_VARS_PER_NODE_SCVX;        // 95
        const NE: usize = N * NX + 6;                       // 41
        const NCT: usize = N * N_CONE_DIM_PER_NODE_SCVX;   // 150
        const NCONES: usize = N * 8;                        // 40

        let phys = mars_params();
        let mut x_init = SVector::<f64, 7>::zeros();
        x_init[0] = 50.0; x_init[2] = 800.0; x_init[5] = -40.0; x_init[6] = libm::log(750.0);

        let traj = hover_reference::<N>(x_init, 750.0, 15.0);
        let mut lin = scvx_dynamics::LinearizedDynamics::<N>::default();
        discretize_foh(&traj, &phys, &mut lin, 4);

        let term = TerminalCondition { r: [0.0, 0.0, 0.0], v: [0.0, 0.0, 0.0] };
        let mut prob: SocpProblem<NP, NE, NCT, NCONES> = SocpProblem::default();
        assemble_scvx_socp::<N, NP, NE, NCT, NCONES>(
            &traj, &lin, &phys, &x_init, &term,
            1.0e3, 1.0e4, false, &mut prob,
        );

        // Vary M slightly (non-uniform) to exercise the scaling path.
        let mut m_diag = SVector::<f64, NCT>::zeros();
        for i in 0..NCT {
            m_diag[i] = 1.0 + 0.1 * libm::sin(0.3 * (i as f64));
        }
        let reg: f64 = 1.0e-7;

        let mut b_x = SVector::<f64, NP>::zeros();
        for i in 0..NP { b_x[i] = 0.5 + libm::sin(0.11 * (i as f64 + 1.0)); }
        let mut b_a = SVector::<f64, NE>::zeros();
        for i in 0..NE { b_a[i] = -0.3 + libm::cos(0.09 * (i as f64 + 1.0)); }

        let mut sol = ReducedKktSolution::<N>::default();
        let status = solve_reduced_kkt_scvx::<N, NP, NE, NCT, NCONES>(
            &prob, &m_diag, reg, &b_x, &b_a, &mut sol,
        );
        assert_eq!(status, ReducedKktStatus::Ok);

        // Dense oracle.
        let mut h_dense = SMatrix::<f64, NP, NP>::zeros();
        for i in 0..NCT {
            for c1 in 0..NP {
                let g1 = prob.g_mat[(i, c1)];
                if g1 == 0.0 { continue; }
                let w = m_diag[i] * g1;
                for c2 in 0..NP {
                    let g2 = prob.g_mat[(i, c2)];
                    if g2 == 0.0 { continue; }
                    h_dense[(c1, c2)] += w * g2;
                }
            }
        }
        for i in 0..NP { h_dense[(i, i)] += reg; }

        const DIM: usize = NP + NE;
        let mut k_dense = SMatrix::<f64, DIM, DIM>::zeros();
        for i in 0..NP {
            for j in 0..NP {
                k_dense[(i, j)] = h_dense[(i, j)];
            }
        }
        for i in 0..NE {
            for j in 0..NP {
                k_dense[(NP + i, j)] = prob.a_mat[(i, j)];
                k_dense[(j, NP + i)] = prob.a_mat[(i, j)];
            }
        }
        let mut rhs_dense = SVector::<f64, DIM>::zeros();
        for i in 0..NP { rhs_dense[i]      = b_x[i]; }
        for i in 0..NE { rhs_dense[NP + i] = b_a[i]; }

        let sol_dense = k_dense.try_inverse().unwrap() * rhs_dense;

        let mut max_dz_err = 0.0_f64;
        for k in 0..N {
            for j in 0..NZ {
                let d = (sol.dz[k][j] - sol_dense[k * NZ + j]).abs();
                if d > max_dz_err { max_dz_err = d; }
            }
        }
        eprintln!("N=5 structured vs dense max Δz err: {:.3e}", max_dz_err);
        assert!(max_dz_err < 1.0e-8, "N=5 max Δz err {max_dz_err} exceeds 1e-8");
    }

    /// **The block-M variant gate**: solve the SCvx reduced KKT via
    /// `solve_reduced_kkt_scvx_block_m` with a synthetic block-diagonal
    /// `M` (per-cone dense blocks), and via dense LU on the full
    /// augmented KKT with the same `M`. Match must hold to ≤ 1e-9.
    ///
    /// This is the gate that proves the **AHO-/NT-compatible** path works
    /// — the IPM ultimately needs a block-dense per-cone scaling, not
    /// just the diagonal.
    #[test]
    fn block_m_structured_matches_dense_lu() {
        const N: usize = 3;
        const NP: usize = N * N_VARS_PER_NODE_SCVX;        // 57
        const NE: usize = N * NX + 6;                       // 27
        const NCT: usize = N * N_CONE_DIM_PER_NODE_SCVX;   // 90
        const NCONES: usize = N * 8;                        // 24

        let phys = mars_params();
        let mut x_init = SVector::<f64, 7>::zeros();
        x_init[0] = 80.0; x_init[2] = 900.0; x_init[5] = -45.0;
        x_init[6] = libm::log(720.0);

        let traj = hover_reference::<N>(x_init, 720.0, 13.0);
        let mut lin = scvx_dynamics::LinearizedDynamics::<N>::default();
        discretize_foh(&traj, &phys, &mut lin, 4);

        let term = TerminalCondition { r: [0.0, 0.0, 0.0], v: [0.0, 0.0, 0.0] };
        let mut prob: SocpProblem<NP, NE, NCT, NCONES> = SocpProblem::default();
        assemble_scvx_socp::<N, NP, NE, NCT, NCONES>(
            &traj, &lin, &phys, &x_init, &term,
            1.0e3, 1.0e4, false, &mut prob,
        );

        // Synthesize a block-diagonal M with per-cone dense (non-diagonal)
        // blocks. For each cone of dim d, fill an SPD d×d block with
        //   M_c[i,j] = 1 + 0.5·δ_{ij} + 0.1·exp(-|i-j|)
        // (decaying off-diagonal — typical AHO-like structure).
        let mut m_full = SMatrix::<f64, NCT, NCT>::zeros();
        for cone in &prob.cones {
            let off = cone.offset;
            let d   = cone.dim;
            for i in 0..d {
                for j in 0..d {
                    let dij = if i == j { 0.5 } else { 0.0 };
                    let decay = libm::exp(-((i as f64 - j as f64).abs()));
                    m_full[(off + i, off + j)] = 1.0 + dij + 0.1 * decay;
                }
            }
        }

        let reg: f64 = 1.0e-7;

        let mut b_x = SVector::<f64, NP>::zeros();
        for i in 0..NP { b_x[i] = 0.4 + libm::sin(0.17 * (i as f64 + 1.0)); }
        let mut b_a = SVector::<f64, NE>::zeros();
        for i in 0..NE { b_a[i] = 0.2 - libm::cos(0.13 * (i as f64 + 1.0)); }

        // ---- Structured block-M solve ----
        let mut sol = ReducedKktSolution::<N>::default();
        let status = solve_reduced_kkt_scvx_block_m::<N, NP, NE, NCT, NCONES>(
            &prob, &m_full, reg, &b_x, &b_a, &mut sol,
        );
        assert_eq!(status, ReducedKktStatus::Ok, "block-M structured solve failed");

        // ---- Dense oracle: H_dense = G^T·M_full·G + reg·I ----
        let h_dense = prob.g_mat.transpose() * m_full * prob.g_mat;
        let mut h_dense = h_dense;
        for i in 0..NP { h_dense[(i, i)] += reg; }

        const DIM: usize = NP + NE;
        let mut k_dense = SMatrix::<f64, DIM, DIM>::zeros();
        for i in 0..NP {
            for j in 0..NP { k_dense[(i, j)] = h_dense[(i, j)]; }
        }
        for i in 0..NE {
            for j in 0..NP {
                k_dense[(NP + i, j)] = prob.a_mat[(i, j)];
                k_dense[(j, NP + i)] = prob.a_mat[(i, j)];
            }
        }
        let mut rhs_dense = SVector::<f64, DIM>::zeros();
        for i in 0..NP { rhs_dense[i]      = b_x[i]; }
        for i in 0..NE { rhs_dense[NP + i] = b_a[i]; }

        let sol_dense = k_dense.try_inverse().expect("dense KKT not invertible") * rhs_dense;

        let mut max_dz_err = 0.0_f64;
        for k in 0..N {
            for j in 0..NZ {
                let d = (sol.dz[k][j] - sol_dense[k * NZ + j]).abs();
                if d > max_dz_err { max_dz_err = d; }
            }
        }
        eprintln!("block-M structured vs dense max Δz err: {:.3e}", max_dz_err);
        assert!(max_dz_err < 1.0e-9, "block-M max Δz err {max_dz_err} exceeds 1e-9");
    }

    /// **The full IPM Newton-step equivalence gate**.
    ///
    /// Build a real SCvx subproblem, manufacture an interior (s, y) IPM
    /// iterate (per-cone identity push so all cones are strictly interior),
    /// and compute one full Mehrotra Newton step via BOTH the dense path
    /// and the structured (block-tridiagonal) path. Recover (Δx, Δλ, Δs, Δy)
    /// via both routes and assert they match to ≤ 1e-7.
    ///
    /// This is the proof that the structured KKT solver can act as a
    /// drop-in for the dense IPM's inner solve. The remaining work — wiring
    /// it into solve_socp's iteration loop — is mechanical.
    #[test]
    fn full_newton_step_dense_matches_structured() {
        use scvx_ipm::soc_arrow_matrix;

        const N: usize = 3;
        const NP: usize = N * N_VARS_PER_NODE_SCVX;        // 57
        const NE: usize = N * NX + 6;                       // 27
        const NCT: usize = N * N_CONE_DIM_PER_NODE_SCVX;   // 90
        const NCONES: usize = N * 8;                        // 24

        // ---- Build an SCvx subproblem identical to the regression test ----
        let phys = mars_params();
        let mut x_init = SVector::<f64, 7>::zeros();
        x_init[0] = 80.0; x_init[2] = 900.0; x_init[5] = -40.0;
        x_init[6] = libm::log(720.0);

        let traj = hover_reference::<N>(x_init, 720.0, 13.0);
        let mut lin = scvx_dynamics::LinearizedDynamics::<N>::default();
        discretize_foh(&traj, &phys, &mut lin, 4);

        let term = TerminalCondition { r: [0.0, 0.0, 0.0], v: [0.0, 0.0, 0.0] };
        let mut prob: SocpProblem<NP, NE, NCT, NCONES> = SocpProblem::default();
        assemble_scvx_socp::<N, NP, NE, NCT, NCONES>(
            &traj, &lin, &phys, &x_init, &term,
            1.0e3, 1.0e4, false, &mut prob,
        );

        // ---- Manufacture an interior (x, λ, s, y) IPM iterate ----
        //
        // Use x = 0 (centered iterate), λ = 0, and set s and y to per-cone
        // identity vectors scaled to give a non-trivial M = arrow(s)⁻¹·arrow(y).
        //
        // Per-cone: s_c = (sc_norm, 0, ..., 0), y_c = (yc_norm, 0, ..., 0).
        // arrow(s_c)⁻¹·arrow(y_c) = (yc_norm/sc_norm) · I_d. So M is diagonal
        // with stage-i scaling (1.0 + 0.1·i) — non-uniform enough to
        // exercise the scaling code, but Jordan-trivial.
        //
        // For a *real* AHO iterate M = arrow(s)⁻¹·arrow(y) where s, y are
        // non-axis-aligned, the per-cone block is dense. We exercise both
        // cases: first this trivial diagonal-only M, then an actively-dense M.
        let x_iter = SVector::<f64, NP>::zeros();
        let lambda_iter = SVector::<f64, NE>::zeros();
        let mut s_iter = SVector::<f64, NCT>::zeros();
        let mut y_iter = SVector::<f64, NCT>::zeros();
        for (i, cone) in prob.cones.iter().enumerate() {
            let off = cone.offset;
            // Per-cone "norm" varying with cone index for non-uniformity.
            let sc = 2.0 + 0.05 * (i as f64);
            let yc = 1.0 + 0.10 * (i as f64);
            s_iter[off] = sc;
            y_iter[off] = yc;
            // Tail components zero — Jordan-axis-aligned iterate.
        }

        // ---- Compute residuals at this iterate ----
        let r_x = prob.c
                + prob.a_mat.transpose() * lambda_iter
                + prob.g_mat.transpose() * y_iter;
        let r_a = prob.a_mat * x_iter - prob.b;
        let r_g = prob.g_mat * x_iter + s_iter - prob.h;
        // Jordan product per cone — implemented inline since this is a test.
        let mut r_c = SVector::<f64, NCT>::zeros();
        for cone in &prob.cones {
            let off = cone.offset;
            let d   = cone.dim;
            // r_c_cone = y_cone ∘ s_cone
            r_c[off] = (0..d).map(|i| s_iter[off + i] * y_iter[off + i]).sum();
            for i in 1..d {
                r_c[off + i] = s_iter[off] * y_iter[off + i] + s_iter[off + i] * y_iter[off];
            }
        }

        // ---- Build full M = arrow(s)⁻¹·arrow(y) block-diagonal per cone ----
        let mut m_full = SMatrix::<f64, NCT, NCT>::zeros();
        for cone in &prob.cones {
            let off = cone.offset;
            let d   = cone.dim;
            macro_rules! per_d {
                ($D:literal) => {{
                    let mut sv = SVector::<f64, $D>::zeros();
                    let mut yv = SVector::<f64, $D>::zeros();
                    for i in 0..$D { sv[i] = s_iter[off + i]; yv[i] = y_iter[off + i]; }
                    let arrow_s_inv = soc_arrow_matrix(&sv).try_inverse().unwrap();
                    let arrow_y = soc_arrow_matrix(&yv);
                    let m_c = arrow_s_inv * arrow_y;
                    for i in 0..$D {
                        for j in 0..$D {
                            m_full[(off + i, off + j)] = m_c[(i, j)];
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
                _  => panic!("unexpected cone dim {d}"),
            }
        }

        // ---- Dense Newton step (matches the IPM's solve_newton_step) ----
        let reg: f64 = 1.0e-8;
        let mut h_dense = prob.g_mat.transpose() * m_full * prob.g_mat;
        for i in 0..NP { h_dense[(i, i)] += reg; }
        let h_inv = h_dense.try_inverse().unwrap();
        let schur = prob.a_mat * h_inv * prob.a_mat.transpose();
        let s_inv = schur.try_inverse().unwrap();

        // arrow_s_inv as full NCT×NCT (block-diag).
        let mut arrow_s_full = SMatrix::<f64, NCT, NCT>::zeros();
        let mut arrow_y_full = SMatrix::<f64, NCT, NCT>::zeros();
        for cone in &prob.cones {
            let off = cone.offset;
            let d   = cone.dim;
            let s0 = s_iter[off];
            let y0 = y_iter[off];
            arrow_s_full[(off, off)] = s0;
            arrow_y_full[(off, off)] = y0;
            for i in 1..d {
                arrow_s_full[(off,     off + i)] = s_iter[off + i];
                arrow_s_full[(off + i, off    )] = s_iter[off + i];
                arrow_s_full[(off + i, off + i)] = s0;
                arrow_y_full[(off,     off + i)] = y_iter[off + i];
                arrow_y_full[(off + i, off    )] = y_iter[off + i];
                arrow_y_full[(off + i, off + i)] = y0;
            }
        }
        let arrow_s_full_inv = arrow_s_full.try_inverse().unwrap();

        // b_x_dense = -r_x - Gᵀ M r_g + Gᵀ arrow(s)⁻¹ r_c
        let b_x_dense = -r_x
                       - prob.g_mat.transpose() * (m_full * r_g)
                       + prob.g_mat.transpose() * (arrow_s_full_inv * r_c);
        let b_a_dense = -r_a;

        let dl_dense = s_inv * (prob.a_mat * (h_inv * b_x_dense) - b_a_dense);
        let dx_dense = h_inv * (b_x_dense - prob.a_mat.transpose() * dl_dense);
        let ds_dense = -r_g - prob.g_mat * dx_dense;
        let dy_dense = arrow_s_full_inv * (arrow_y_full * (r_g + prob.g_mat * dx_dense) - r_c);

        // ---- Structured Newton step ----
        //
        // The structured solver uses the IPM augmented KKT convention:
        //   [H  Aᵀ] [Δx]   [b_x]
        //   [A   0] [Δλ] = [b_a]
        // where b_x = -r_x - Gᵀ·M·r_g + Gᵀ·arrow(s)⁻¹·r_c and b_a = -r_a.
        //
        // Internally solve_reduced_kkt_scvx_block_m expects the RHS already
        // wrapped this way, so we just pass b_x_dense and b_a_dense — they
        // are identical to the dense RHS construction.
        let mut sol = ReducedKktSolution::<N>::default();
        let status = solve_reduced_kkt_scvx_block_m::<N, NP, NE, NCT, NCONES>(
            &prob, &m_full, reg, &b_x_dense, &b_a_dense, &mut sol,
        );
        assert_eq!(status, ReducedKktStatus::Ok);

        // Stitch the structured Δx back into a stacked NP vector for
        // comparison with the dense path.
        let mut dx_structured = SVector::<f64, NP>::zeros();
        for k in 0..N {
            for j in 0..NZ {
                dx_structured[k * NZ + j] = sol.dz[k][j];
            }
        }
        // Δs and Δy recover via the same formulas as dense (post-Δx work
        // doesn't depend on whether Δx came from dense or structured).
        let ds_structured = -r_g - prob.g_mat * dx_structured;
        let dy_structured = arrow_s_full_inv
                          * (arrow_y_full * (r_g + prob.g_mat * dx_structured) - r_c);

        // ---- Compare ----
        let mut max_dx_err = 0.0_f64;
        for i in 0..NP {
            let d = (dx_dense[i] - dx_structured[i]).abs();
            if d > max_dx_err { max_dx_err = d; }
        }
        eprintln!("full-step dense-vs-structured Δx max err: {:.3e}", max_dx_err);
        assert!(max_dx_err < 1.0e-7, "Δx mismatch: {max_dx_err}");

        let mut max_ds_err = 0.0_f64;
        for i in 0..NCT {
            let d = (ds_dense[i] - ds_structured[i]).abs();
            if d > max_ds_err { max_ds_err = d; }
        }
        eprintln!("full-step dense-vs-structured Δs max err: {:.3e}", max_ds_err);
        assert!(max_ds_err < 1.0e-7, "Δs mismatch: {max_ds_err}");

        let mut max_dy_err = 0.0_f64;
        for i in 0..NCT {
            let d = (dy_dense[i] - dy_structured[i]).abs();
            if d > max_dy_err { max_dy_err = d; }
        }
        eprintln!("full-step dense-vs-structured Δy max err: {:.3e}", max_dy_err);
        assert!(max_dy_err < 1.0e-7, "Δy mismatch: {max_dy_err}");

        // λ multipliers
        let mut max_dl_err = 0.0_f64;
        for i in 0..NX {
            let d = (dl_dense[i] - sol.dlam_init[i]).abs();
            if d > max_dl_err { max_dl_err = d; }
        }
        for k in 0..(N - 1) {
            for i in 0..NX {
                let d = (dl_dense[NX + k * NX + i] - sol.dlam_dyn[k][i]).abs();
                if d > max_dl_err { max_dl_err = d; }
            }
        }
        for i in 0..6 {
            let d = (dl_dense[N * NX + i] - sol.dlam_term[i]).abs();
            if d > max_dl_err { max_dl_err = d; }
        }
        eprintln!("full-step dense-vs-structured Δλ max err: {:.3e}", max_dl_err);
        assert!(max_dl_err < 1.0e-7, "Δλ mismatch: {max_dl_err}");
    }

    /// **The Phase 6.7 split-API gate**: calling `factor()` once then
    /// `apply()` twice on different RHS vectors must produce the same
    /// `(Δz, Δλ_*)` as calling `solve_reduced_kkt_scvx_block_m` twice
    /// (the original one-shot path). Match must hold to machine precision.
    ///
    /// This is the proof that the factor/apply split is semantically
    /// equivalent — the speedup comes from amortizing the factor work,
    /// not from a math change.
    #[test]
    fn factor_apply_split_matches_one_shot() {
        const N: usize = 3;
        const NP: usize = N * N_VARS_PER_NODE_SCVX;
        const NE: usize = N * NX + 6;
        const NCT: usize = N * N_CONE_DIM_PER_NODE_SCVX;
        const NCONES: usize = N * 8;

        // Build a real SCvx subproblem.
        let phys = mars_params();
        let mut x_init = SVector::<f64, 7>::zeros();
        x_init[0] = 75.0; x_init[2] = 850.0; x_init[5] = -42.0;
        x_init[6] = libm::log(710.0);

        let traj = hover_reference::<N>(x_init, 710.0, 12.5);
        let mut lin = scvx_dynamics::LinearizedDynamics::<N>::default();
        discretize_foh(&traj, &phys, &mut lin, 4);

        let term = TerminalCondition { r: [0.0, 0.0, 0.0], v: [0.0, 0.0, 0.0] };
        let mut prob: SocpProblem<NP, NE, NCT, NCONES> = SocpProblem::default();
        assemble_scvx_socp::<N, NP, NE, NCT, NCONES>(
            &traj, &lin, &phys, &x_init, &term,
            1.0e3, 1.0e4, false, &mut prob,
        );

        // Synthesize a block-diagonal M with per-cone dense entries.
        let mut m_full = SMatrix::<f64, NCT, NCT>::zeros();
        for cone in &prob.cones {
            let off = cone.offset;
            let d   = cone.dim;
            for i in 0..d {
                for j in 0..d {
                    let dij = if i == j { 0.5 } else { 0.0 };
                    let decay = libm::exp(-((i as f64 - j as f64).abs()));
                    m_full[(off + i, off + j)] = 1.0 + dij + 0.1 * decay;
                }
            }
        }
        let reg: f64 = 1.0e-7;

        // Two different RHS — mimics the IPM's predictor + corrector.
        let mut b_x_pred = SVector::<f64, NP>::zeros();
        let mut b_x_corr = SVector::<f64, NP>::zeros();
        let mut b_a      = SVector::<f64, NE>::zeros();
        for i in 0..NP {
            b_x_pred[i] = 0.3 + libm::sin(0.11 * (i as f64 + 1.0));
            b_x_corr[i] = 0.2 - libm::cos(0.17 * (i as f64 + 1.0));
        }
        for i in 0..NE {
            b_a[i] = 0.1 + libm::sin(0.09 * (i as f64 + 1.0));
        }

        // ---- One-shot path: two full solves ----
        let mut sol_oneshot_pred = ReducedKktSolution::<N>::default();
        let mut sol_oneshot_corr = ReducedKktSolution::<N>::default();
        assert_eq!(
            solve_reduced_kkt_scvx_block_m::<N, NP, NE, NCT, NCONES>(
                &prob, &m_full, reg, &b_x_pred, &b_a, &mut sol_oneshot_pred,
            ),
            ReducedKktStatus::Ok,
        );
        assert_eq!(
            solve_reduced_kkt_scvx_block_m::<N, NP, NE, NCT, NCONES>(
                &prob, &m_full, reg, &b_x_corr, &b_a, &mut sol_oneshot_corr,
            ),
            ReducedKktStatus::Ok,
        );

        // ---- Split path: factor once + apply twice ----
        let mut factor = ReducedKktFactor::<N>::default();
        assert_eq!(
            factor_reduced_kkt_scvx_block_m::<N, NP, NE, NCT, NCONES>(
                &prob, &m_full, reg, &mut factor,
            ),
            ReducedKktStatus::Ok,
        );
        let mut sol_split_pred = ReducedKktSolution::<N>::default();
        let mut sol_split_corr = ReducedKktSolution::<N>::default();
        assert_eq!(
            solve_reduced_kkt_scvx_with_factor::<N, NP, NE, NCT, NCONES>(
                &prob, &factor, &b_x_pred, &b_a, &mut sol_split_pred,
            ),
            ReducedKktStatus::Ok,
        );
        assert_eq!(
            solve_reduced_kkt_scvx_with_factor::<N, NP, NE, NCT, NCONES>(
                &prob, &factor, &b_x_corr, &b_a, &mut sol_split_corr,
            ),
            ReducedKktStatus::Ok,
        );

        // ---- Compare: machine precision per RHS, all four blocks ----
        let mut max_diff = 0.0_f64;
        for rhs_idx in 0..2 {
            let (one, split) = if rhs_idx == 0 {
                (&sol_oneshot_pred, &sol_split_pred)
            } else {
                (&sol_oneshot_corr, &sol_split_corr)
            };
            for k in 0..N {
                for i in 0..NZ {
                    let d = (one.dz[k][i] - split.dz[k][i]).abs();
                    if d > max_diff { max_diff = d; }
                }
            }
            for i in 0..NX {
                let d = (one.dlam_init[i] - split.dlam_init[i]).abs();
                if d > max_diff { max_diff = d; }
            }
            for k in 0..(N - 1) {
                for i in 0..NX {
                    let d = (one.dlam_dyn[k][i] - split.dlam_dyn[k][i]).abs();
                    if d > max_diff { max_diff = d; }
                }
            }
            for i in 0..6 {
                let d = (one.dlam_term[i] - split.dlam_term[i]).abs();
                if d > max_diff { max_diff = d; }
            }
        }
        eprintln!("factor+apply vs one-shot max err: {:.3e}", max_diff);
        // Both paths execute the SAME arithmetic in the SAME order — the
        // only difference is whether the factor lives across two RHS or
        // is rebuilt for each. So agreement should be bit-exact (0.0) or
        // within a few ulps (≤ 1e-15) from any compiler reordering.
        assert!(
            max_diff < 1.0e-12,
            "factor+apply diverges from one-shot path: max_diff = {max_diff}"
        );
    }

    /// **The Phase 6.8 free-tf SMW gate**: build an SCvx subproblem with
    /// `use_free_tf = true`, solve the reduced KKT via both the dense LU
    /// oracle AND the structured Sherman-Morrison path, and verify the
    /// resulting `(Δz, Δδτ, Δλ)` agree to ≤ 1e-7.
    ///
    /// This is the proof that the SMW rank-1 update correctly handles the
    /// global δτ column in `A` (which couples all dynamics row blocks of
    /// the Schur complement).
    #[test]
    fn free_tf_structured_matches_dense_lu() {
        use crate::assemble::{
            np_scvx_free_tf, nct_scvx_free_tf, ncones_scvx_free_tf,
        };
        use scvx_ipm::soc_arrow_matrix;

        const N: usize = 3;
        const NP: usize = np_scvx_free_tf(N);                    // 58 (= 19·3 + 1)
        const NE: usize = N * NX + 6;                             // 27
        const NCT: usize = nct_scvx_free_tf(N);                  // 92 (= 30·3 + 2)
        const NCONES: usize = ncones_scvx_free_tf(N);            // 26 (= 8·3 + 2)

        let phys = mars_params();
        let mut x_init = SVector::<f64, 7>::zeros();
        x_init[0] = 60.0; x_init[2] = 800.0; x_init[5] = -38.0;
        x_init[6] = libm::log(720.0);

        let traj = hover_reference::<N>(x_init, 720.0, 12.0);
        let mut lin = scvx_dynamics::LinearizedDynamics::<N>::default();
        discretize_foh(&traj, &phys, &mut lin, 4);

        let term = TerminalCondition { r: [0.0, 0.0, 0.0], v: [0.0, 0.0, 0.0] };
        let mut prob: SocpProblem<NP, NE, NCT, NCONES> = SocpProblem::default();
        assemble_scvx_socp::<N, NP, NE, NCT, NCONES>(
            &traj, &lin, &phys, &x_init, &term,
            /*trust_eta=*/ 1.0e3,
            /*virt_weight=*/ 1.0e4,
            /*use_free_tf=*/ true,
            &mut prob,
        );

        // Manufacture a dense per-cone M (same pattern as the block-M test).
        let mut m_full = SMatrix::<f64, NCT, NCT>::zeros();
        for cone in &prob.cones {
            let off = cone.offset;
            let d   = cone.dim;
            for i in 0..d {
                for j in 0..d {
                    let dij = if i == j { 0.5 } else { 0.0 };
                    let decay = libm::exp(-((i as f64 - j as f64).abs()));
                    m_full[(off + i, off + j)] = 1.0 + dij + 0.1 * decay;
                }
            }
        }
        let _ = soc_arrow_matrix::<3>; // silence unused (used elsewhere in this test file)
        let reg: f64 = 1.0e-7;

        // Synthetic RHS.
        let mut b_x = SVector::<f64, NP>::zeros();
        for i in 0..NP { b_x[i] = 0.4 + libm::sin(0.13 * (i as f64 + 1.0)); }
        let mut b_a = SVector::<f64, NE>::zeros();
        for i in 0..NE { b_a[i] = 0.2 - libm::cos(0.11 * (i as f64 + 1.0)); }

        // ---- Structured SMW solve ----
        let mut factor = ReducedKktFactorFreeTf::<N>::default();
        let st = factor_reduced_kkt_scvx_block_m_free_tf::<N, NP, NE, NCT, NCONES>(
            &prob, &m_full, reg, &mut factor,
        );
        assert_eq!(st, ReducedKktStatus::Ok, "free-tf factor failed");

        let mut sol = ReducedKktSolutionFreeTf::<N>::default();
        let st = solve_reduced_kkt_scvx_with_factor_free_tf::<N, NP, NE, NCT, NCONES>(
            &prob, &factor, &b_x, &b_a, &mut sol,
        );
        assert_eq!(st, ReducedKktStatus::Ok, "free-tf apply failed");

        // ---- Dense LU oracle ----
        // H = G^T·M·G + reg·I (NP × NP, includes the δτ row/col)
        let mut h_dense = prob.g_mat.transpose() * m_full * prob.g_mat;
        for i in 0..NP { h_dense[(i, i)] += reg; }

        const DIM: usize = NP + NE;
        let mut k_dense = SMatrix::<f64, DIM, DIM>::zeros();
        for i in 0..NP {
            for j in 0..NP { k_dense[(i, j)] = h_dense[(i, j)]; }
        }
        for i in 0..NE {
            for j in 0..NP {
                k_dense[(NP + i, j)] = prob.a_mat[(i, j)];
                k_dense[(j, NP + i)] = prob.a_mat[(i, j)];
            }
        }
        let mut rhs_dense = SVector::<f64, DIM>::zeros();
        for i in 0..NP { rhs_dense[i]      = b_x[i]; }
        for i in 0..NE { rhs_dense[NP + i] = b_a[i]; }

        let sol_dense = k_dense.try_inverse()
            .expect("free-tf dense KKT not invertible") * rhs_dense;

        // ---- Compare Δz per stage ----
        let mut max_dz_err = 0.0_f64;
        for k in 0..N {
            for j in 0..NZ {
                let d = (sol.base.dz[k][j] - sol_dense[k * NZ + j]).abs();
                if d > max_dz_err { max_dz_err = d; }
            }
        }
        eprintln!("free-tf SMW vs dense Δz max err: {:.3e}", max_dz_err);
        assert!(max_dz_err < 1.0e-7, "free-tf Δz mismatch: {max_dz_err}");

        // Δδτ — the new piece. Should match dense[N·NZ] (the δτ slot).
        let dtau_dense = sol_dense[N * NZ];
        let dtau_err = (sol.dz_delta_tau - dtau_dense).abs();
        eprintln!("free-tf SMW vs dense Δδτ err: {:.3e} ({:.6} vs {:.6})",
                  dtau_err, sol.dz_delta_tau, dtau_dense);
        assert!(dtau_err < 1.0e-7, "free-tf Δδτ mismatch: {dtau_err}");

        // Δλ comparison (init, dyn, term).
        let mut max_dl_err = 0.0_f64;
        for i in 0..NX {
            let d = (sol.base.dlam_init[i] - sol_dense[NP + i]).abs();
            if d > max_dl_err { max_dl_err = d; }
        }
        for k in 0..(N - 1) {
            for i in 0..NX {
                let d = (sol.base.dlam_dyn[k][i] - sol_dense[NP + NX + k * NX + i]).abs();
                if d > max_dl_err { max_dl_err = d; }
            }
        }
        for i in 0..6 {
            let d = (sol.base.dlam_term[i] - sol_dense[NP + N * NX + i]).abs();
            if d > max_dl_err { max_dl_err = d; }
        }
        eprintln!("free-tf SMW vs dense Δλ max err: {:.3e}", max_dl_err);
        assert!(max_dl_err < 1.0e-7, "free-tf Δλ mismatch: {max_dl_err}");
    }

    /// N=1 is degenerate (initial AND terminal both pin parts of `x_0`)
    /// and the IPM never runs with N=1 in practice. Verify the structured
    /// solver still returns a non-panicking status — either `Ok` if the
    /// boundary conditions are consistent, or a clean failure code if not.
    #[test]
    fn n_one_smoke_test() {
        const N: usize = 1;
        const NP: usize = N * N_VARS_PER_NODE_SCVX;        // 19
        const NE: usize = N * NX + 6;                       // 13
        const NCT: usize = N * N_CONE_DIM_PER_NODE_SCVX;   // 30
        const NCONES: usize = N * 8;                        // 8

        let phys = mars_params();
        // Self-consistent boundaries: initial r = terminal r, initial v = terminal v.
        let mut x_init = SVector::<f64, 7>::zeros();
        x_init[6] = libm::log(600.0);   // mass (free at terminal — OK)

        let traj = hover_reference::<N>(x_init, 600.0, 8.0);
        let mut lin = scvx_dynamics::LinearizedDynamics::<N>::default();
        discretize_foh(&traj, &phys, &mut lin, 4);

        let term = TerminalCondition { r: [0.0, 0.0, 0.0], v: [0.0, 0.0, 0.0] };
        let mut prob: SocpProblem<NP, NE, NCT, NCONES> = SocpProblem::default();
        assemble_scvx_socp::<N, NP, NE, NCT, NCONES>(
            &traj, &lin, &phys, &x_init, &term,
            1.0e3, 1.0e4, false, &mut prob,
        );

        let m_diag = SVector::<f64, NCT>::from_element(1.0);
        let mut b_x = SVector::<f64, NP>::zeros();
        b_x[0] = 1.0;
        let b_a = SVector::<f64, NE>::zeros();

        let mut sol = ReducedKktSolution::<N>::default();
        let status = solve_reduced_kkt_scvx::<N, NP, NE, NCT, NCONES>(
            &prob, &m_diag, 1.0e-6, &b_x, &b_a, &mut sol,
        );
        // For N=1, the initial constraint x_0 = x_init AND the terminal
        // constraint r_0 = r_target / v_0 = v_target overlap in the same
        // primal columns. The reduced-KKT augmented matrix is **rank
        // deficient** in any non-trivial RHS — Schur factorization
        // legitimately reports `SchurSingular`. We just confirm no panic
        // and a defined status (not undefined behavior).
        assert!(
            matches!(status, ReducedKktStatus::Ok | ReducedKktStatus::SchurSingular),
            "unexpected status {:?}", status,
        );
        // If by chance the consistent BCs produced a solvable system,
        // confirm finite outputs.
        if status == ReducedKktStatus::Ok {
            for k in 0..N {
                for i in 0..NZ {
                    assert!(sol.dz[k][i].is_finite(), "Δz[{k}][{i}] not finite");
                }
            }
        }
    }

}
