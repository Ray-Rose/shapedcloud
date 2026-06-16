//! Second-order cone primitives.
//!
//! `K_D = { z ∈ ℝ^D : z₀ ≥ ‖z̄‖₂ }` where `z = (z₀, z̄)`, `z̄ ∈ ℝ^(D−1)`.
//!
//! Slice-based scalar / Jordan-algebra ops (`soc_det`, `soc_jordan_product`,
//! `soc_project`, `soc_max_step`, …) handle every cone dimension uniformly
//! with no monomorphization blow-up — callers slice the per-cone window out
//! of the big primal/dual vectors.
//!
//! Const-generic matrix-form NT scaling lives below the slice primitives:
//! `soc_arrow_matrix`, `soc_arrow_inv_sqrt`, `soc_nt_scaling_matrix`. These
//! are usable today as building blocks; integrating the resulting symmetric
//! `M = W⁻²` into the IPM's Newton system is the P1b lift (requires re-
//! deriving the complementarity residual in scaled coordinates — not just
//! swapping the scaling matrix).
//!
//! References for the Jordan-algebra view of SOC:
//! - Alizadeh & Goldfarb, "Second-order cone programming", Math. Prog. 2003.
//! - Sturm, "Implementation of interior point methods for mixed semidefinite
//!   and second order cone optimization problems", Opt. Methods & Software 2002.

use libm::sqrt;
use nalgebra::{SMatrix, SVector};

// ---------------------------------------------------------------------------
// Scalar invariants
// ---------------------------------------------------------------------------

/// SOC determinant `det(z) = z₀² − ‖z̄‖²`. Positive in `int(K)`; zero on the
/// boundary; negative outside.
#[inline]
pub fn soc_det(z: &[f64]) -> f64 {
    let z0 = z[0];
    let mut sq = 0.0;
    for &x in &z[1..] {
        sq += x * x;
    }
    z0 * z0 - sq
}

/// `‖z̄‖₂` — Euclidean norm of the tail of `z`.
#[inline]
pub fn soc_tail_norm(z: &[f64]) -> f64 {
    let mut sq = 0.0;
    for &x in &z[1..] {
        sq += x * x;
    }
    sqrt(sq)
}

/// Returns `true` iff `z` lies in `K_D` (closure, not interior).
#[inline]
pub fn soc_in_cone(z: &[f64]) -> bool {
    z[0] >= soc_tail_norm(z)
}

/// Returns `true` iff `z` lies in the interior of `K_D`.
#[inline]
pub fn soc_in_interior(z: &[f64]) -> bool {
    z[0] > soc_tail_norm(z)
}

// ---------------------------------------------------------------------------
// Jordan algebra primitives
// ---------------------------------------------------------------------------

/// Write the Jordan-algebra identity `e = (1, 0, …, 0)` into `out`.
#[inline]
pub fn soc_e(out: &mut [f64]) {
    out[0] = 1.0;
    for slot in &mut out[1..] {
        *slot = 0.0;
    }
}

/// Jordan product `u ∘ v = (uᵀv, u₀ v̄ + v₀ ū)`. Lengths must match.
///
/// `arrow(u) · v` and `u ∘ v` coincide — this is the matrix-free form.
pub fn soc_jordan_product(u: &[f64], v: &[f64], out: &mut [f64]) {
    let mut dot = 0.0;
    for k in 0..u.len() {
        dot += u[k] * v[k];
    }
    out[0] = dot;
    let u0 = u[0];
    let v0 = v[0];
    for k in 1..out.len() {
        out[k] = u0 * v[k] + v0 * u[k];
    }
}

/// SOC square root: given `s ∈ int(K_D)`, write `s^(1/2)` into `out`.
///
/// Closed form: `s^(1/2) = (t,  s̄ / (2t))` where `t = sqrt((s₀ + γ)/2)`
/// and `γ = sqrt(det(s))`.
pub fn soc_sqrt(s: &[f64], out: &mut [f64]) {
    // `.max(0.0)` is robust against tiny negative `det(s)` from round-off
    // when `s` sits on the boundary. f64::max is an intrinsic, no_std-safe.
    let gamma     = sqrt(soc_det(s).max(0.0));
    let t         = sqrt(((s[0] + gamma) * 0.5).max(0.0));
    let inv_two_t = if t > 0.0 { 0.5 / t } else { 0.0 };
    out[0] = t;
    for k in 1..s.len() {
        out[k] = s[k] * inv_two_t;
    }
}

// ---------------------------------------------------------------------------
// Projection & step length
// ---------------------------------------------------------------------------

/// Project `z` onto the closure of `K_D`, writing the result into `out`.
///
/// Three cases:
/// - `‖z̄‖ ≤ z₀`:       `out ← z`        (already in cone)
/// - `‖z̄‖ ≤ −z₀`:      `out ← 0`        (in `−K`; closest point is origin)
/// - otherwise:        blend onto the boundary by Moreau decomposition.
pub fn soc_project(z: &[f64], out: &mut [f64]) {
    let z0        = z[0];
    let zbar_norm = soc_tail_norm(z);

    if zbar_norm <= z0 {
        out.copy_from_slice(z);
    } else if zbar_norm <= -z0 {
        for slot in out.iter_mut() {
            *slot = 0.0;
        }
    } else {
        let scale  = (z0 + zbar_norm) * 0.5;
        let factor = scale / zbar_norm;
        out[0] = scale;
        for k in 1..out.len() {
            out[k] = factor * z[k];
        }
    }
}

/// Maximum step `α ≥ 0` such that `z + α·dz` stays in the closed cone
/// `K_D`, given `z ∈ int(K_D)`.
///
/// Solves `det(z + α dz) ≥ 0  ∧  z₀ + α dz₀ ≥ 0`. Returns `f64::INFINITY`
/// when the cone never binds along this ray.
pub fn soc_max_step(z: &[f64], dz: &[f64]) -> f64 {
    // a α² + 2β α + c = 0
    //   a = det(dz)
    //   β = z₀ dz₀ − z̄ · dz̄
    //   c = det(z) > 0   (we assume z ∈ int K)
    let a    = soc_det(dz);
    let z0   = z[0];
    let dz0  = dz[0];
    let mut zb_dot_db = 0.0;
    for k in 1..z.len() {
        zb_dot_db += z[k] * dz[k];
    }
    let beta = z0 * dz0 - zb_dot_db;
    let c    = soc_det(z);

    // Pick the smallest positive root of  a α² + 2β α + c = 0.
    let mut alpha_det = f64::INFINITY;
    if a.abs() > 0.0 {
        let disc = beta * beta - a * c;
        if disc >= 0.0 {
            let sd = sqrt(disc);
            // Two candidate roots
            let r1 = (-beta - sd) / a;
            let r2 = (-beta + sd) / a;
            // Smallest strictly-positive
            for &r in &[r1, r2] {
                if r > 0.0 && r < alpha_det {
                    alpha_det = r;
                }
            }
        }
    } else if beta.abs() > 0.0 {
        // Linear: 2β α + c = 0  ⇒  α = −c/(2β)
        let r = -c / (2.0 * beta);
        if r > 0.0 {
            alpha_det = r;
        }
    }

    // Linear half-space  z₀ + α dz₀ ≥ 0.
    let alpha_lin = if dz0 < 0.0 { -z0 / dz0 } else { f64::INFINITY };

    alpha_det.min(alpha_lin)
}

// ---------------------------------------------------------------------------
// Arrow operator (matrix form)
// ---------------------------------------------------------------------------

/// Build the arrow matrix `arrow(z) = [[z₀, z̄ᵀ], [z̄, z₀·I_{D-1}]]`.
///
/// This is the matrix representation of "multiply by z" in the SOC Jordan
/// algebra: `arrow(z)·v = z ∘ v`. Symmetric; PD iff `z ∈ int(K)`.
pub fn soc_arrow_matrix<const D: usize>(z: &SVector<f64, D>) -> SMatrix<f64, D, D> {
    let mut m = SMatrix::<f64, D, D>::zeros();
    let z0 = z[0];
    m[(0, 0)] = z0;
    for i in 1..D {
        m[(0, i)] = z[i];
        m[(i, 0)] = z[i];
        m[(i, i)] = z0;
    }
    m
}

// ---------------------------------------------------------------------------
// Nesterov-Todd scaling (matrix form)
//
// An earlier vector-form `W = η·arrow(w_n)` parameterization only matches
// the true NT scaling when `s_bar ∥ y_bar` — for misaligned bars in D ≥ 3
// the SOC Jordan algebra is non-associative
// (`arrow(a)·arrow(b)·v ≠ arrow(a·b)·v`), so `arrow(w_n)²` cannot represent
// the true `W²` in general. That implementation was removed; the matrix
// form below (`W² = arrow(s)⁻¹ᐟ²·arrow(y)·arrow(s)⁻¹ᐟ²`) is correct in the
// "operator geometric mean" sense — symmetric PD always, exact NT when
// `arrow(s)` and `arrow(y)` commute, an excellent approximation otherwise.
// ---------------------------------------------------------------------------

/// Build the matrix `arrow(s)^{−1/2}` for `s ∈ int(SOC^D)`.
///
/// Uses the closed-form spectral decomposition of `arrow(s)`. The matrix
/// has three distinct eigenvalues:
/// - `λ_1 = s_0 + ‖s_bar‖`         (mult 1, eigvec aligned with `(1, û)/√2`)
/// - `λ_2 = s_0 − ‖s_bar‖`         (mult 1, eigvec aligned with `(1, −û)/√2`)
/// - `λ_3 = s_0`                    (mult D−2, eigvecs in `(0, û^⊥)`)
///
/// Returns `None` if `s` is on or outside the cone (`λ_2 ≤ 0`).
///
/// **Why this matters:** the AHO direction uses `arrow(s)^{−1}·arrow(y)`,
/// which is asymmetric. The Nesterov-Todd direction uses
/// `W² = arrow(s)^{−1/2}·arrow(y)·arrow(s)^{−1/2}`, which is symmetric PD.
/// Symmetric H = G^T·W^{−2}·G is dramatically better conditioned at the
/// cone boundary, eliminating the AHO endgame degeneracy.
pub fn soc_arrow_inv_sqrt<const D: usize>(
    s: &nalgebra::SVector<f64, D>,
) -> Option<nalgebra::SMatrix<f64, D, D>> {
    let s0 = s[0];
    let mut bar_norm_sq = 0.0;
    for i in 1..D {
        bar_norm_sq += s[i] * s[i];
    }
    let bar_norm = sqrt(bar_norm_sq);

    let lambda1 = s0 + bar_norm; // largest
    let lambda2 = s0 - bar_norm; // smallest
    if lambda1 <= 0.0 || lambda2 <= 0.0 || !lambda1.is_finite() || !lambda2.is_finite() {
        return None;
    }

    let l1_pow = 1.0 / sqrt(lambda1);
    let l2_pow = 1.0 / sqrt(lambda2);

    let mut result = nalgebra::SMatrix::<f64, D, D>::zeros();

    if D == 1 {
        result[(0, 0)] = 1.0 / sqrt(s0);
        return Some(result);
    }

    if bar_norm < 1.0e-15 {
        // arrow(s) = s_0·I  ⇒  arrow(s)^{−1/2} = (1/√s_0)·I
        let val = 1.0 / sqrt(s0);
        for i in 0..D {
            result[(i, i)] = val;
        }
        return Some(result);
    }

    let l3_pow = 1.0 / sqrt(s0); // eigval λ_3 = s_0 (mult D−2)

    // `û = s_bar / ‖s_bar‖`, with components `û_i = s[i+1] / bar_norm` for
    // `i ∈ 0..D−1` in bar-space (using 0-based bar indexing).
    //
    // arrow(s)^{−1/2}[0, 0]    = (l1_pow + l2_pow)/2
    // arrow(s)^{−1/2}[0, j≥1]  = (l1_pow − l2_pow)/2 · û[j−1]
    // arrow(s)^{−1/2}[i≥1, j≥1] = ((l1_pow + l2_pow)/2 − l3_pow)·û[i−1]·û[j−1]
    //                            + l3_pow · δ_{ij}
    let half_sum = (l1_pow + l2_pow) * 0.5;
    let half_diff = (l1_pow - l2_pow) * 0.5;
    let coef = half_sum - l3_pow;
    let inv_norm = 1.0 / bar_norm;

    result[(0, 0)] = half_sum;
    for j in 1..D {
        let u_j = s[j] * inv_norm;
        result[(0, j)] = half_diff * u_j;
        result[(j, 0)] = half_diff * u_j;
    }
    for i in 1..D {
        let u_i = s[i] * inv_norm;
        for j in 1..D {
            let u_j = s[j] * inv_norm;
            let off_diag = coef * u_i * u_j;
            result[(i, j)] = if i == j { off_diag + l3_pow } else { off_diag };
        }
    }
    Some(result)
}

/// Compute the per-cone symmetric scaling matrix `M = W^{−2}` for the IPM
/// (NT-style symmetric replacement for the asymmetric AHO scaling).
///
/// `M = (arrow(s)^{−1/2}·arrow(y)·arrow(s)^{−1/2})^{−1}`. This is the
/// **operator geometric mean** of `arrow(s)^{−1}` and `arrow(y)`, which is:
///
/// 1. **Symmetric PD** (the key property that fixes AHO's endgame
///    degeneracy and gives `H = GᵀMG` clean conditioning).
/// 2. **Exactly** equal to the true NT scaling when `arrow(s)` and `arrow(y)`
///    commute (i.e., `s_bar` ∥ `y_bar`).
/// 3. **Approximately** the NT scaling otherwise — residual `‖M⁻¹·s − y‖`
///    typically `< 1e-3` for cones in practice, where `s` and `y` are
///    correlated through the centering condition `s∘y = μe`.
///
/// The strict-NT closed form for SOC with misaligned bars exists (Sturm
/// 1999, Tütüncü-Toh-Todd 2003) but is substantially more code. The
/// symmetric-PD property of this geometric-mean form is sufficient to
/// unblock the AHO-endgame issue that motivated NT in the first place.
///
/// Returns `None` if `s` or `y` is non-interior or any intermediate matrix
/// is non-invertible.
pub fn soc_nt_scaling_matrix<const D: usize>(
    s: &SVector<f64, D>,
    y: &SVector<f64, D>,
) -> Option<SMatrix<f64, D, D>> {
    let s_inv_half = soc_arrow_inv_sqrt::<D>(s)?;
    let arrow_y = soc_arrow_matrix(y);
    let w_sq = s_inv_half * arrow_y * s_inv_half;
    w_sq.try_inverse()
}

/// Build `W²` per cone: the operator-geometric-mean scaling matrix
/// `W² = arrow(s)⁻¹ᐟ²·arrow(y)·arrow(s)⁻¹ᐟ²`. Symmetric PD when `s, y` are
/// strictly interior. Returns `None` if `s` is non-interior.
///
/// This is the matrix that appears as `H = Gᵀ·W²·G` in the NT-direction
/// reduced KKT — symmetric PD, in contrast to AHO's asymmetric
/// `Gᵀ·(arrow(s)⁻¹·arrow(y))·G`.
pub fn soc_w_squared<const D: usize>(
    s: &SVector<f64, D>,
    y: &SVector<f64, D>,
) -> Option<SMatrix<f64, D, D>> {
    let s_inv_half = soc_arrow_inv_sqrt::<D>(s)?;
    let arrow_y = soc_arrow_matrix(y);
    Some(s_inv_half * arrow_y * s_inv_half)
}

/// Build both `W` and `W⁻¹` per cone via the Denman-Beavers matrix-sqrt
/// iteration applied to `W²`.
///
/// **Why not eigendecomposition:** `nalgebra::SymmetricEigen::try_new`
/// requires `Const<D>: DimSub<U1>`, which doesn't hold for generic const-`D`
/// in stable Rust 1.94. Denman-Beavers avoids that constraint entirely:
///   `Y_{k+1} = ½(Y_k + Z_k⁻¹)`,   `Z_{k+1} = ½(Z_k + Y_k⁻¹)`
/// starting from `Y_0 = W²`, `Z_0 = I`. `Y_k → W`, `Z_k → W⁻¹` quadratically.
///
/// **Conditioning limit:** plain DB is robust for well-conditioned `W²`
/// (`κ < ~10⁶`) which covers iterates not yet near the cone boundary. The
/// textbook Higham scaling (`γ_k = (|det Z_k|/|det Y_k|)^(1/(2D))`) is more
/// robust but requires `SMatrix::determinant`, which needs
/// `Const<D>: ToTypenum` (implemented per fixed `D` only — same limitation
/// that blocks `SymmetricEigen` from being usable in our generic context).
/// On SCvx subproblems whose iterates drift toward the boundary, plain DB
/// fails after a few iterations and the IPM correctly bails to
/// `NumericalError`. Fixing this is logged as future work and would either
/// require dispatched-per-D specialization with `SymmetricEigen`, or a
/// custom slice-based matrix-sqrt implementation.
///
/// For our cones (`D ≤ 11`) and well-conditioned iterates, 10–15 iterations
/// reach machine precision. Bounded WCET: at most `MAX_ITERS = 30` matrix-
/// inversions of a `D×D` matrix.
///
/// Used by the NT-direction Newton step to convert between scaled
/// coordinates (`s̃ = W·s`, `ỹ = W⁻¹·y`) and original. By the NT property,
/// `s̃ = ỹ` at the iterate (defining identity), but we still need to apply
/// `W` and `W⁻¹` to other vectors during the Newton step.
///
/// Returns `None` if `s` is non-interior, `arrow(y)` is non-PD, or the
/// iteration fails to converge / encounters a singular Y/Z.
pub fn soc_nt_w_and_inverse<const D: usize>(
    s: &SVector<f64, D>,
    y: &SVector<f64, D>,
) -> Option<(SMatrix<f64, D, D>, SMatrix<f64, D, D>)> {
    let w_sq = soc_w_squared(s, y)?;
    const MAX_ITERS: usize = 30;
    const TOL: f64        = 1.0e-13;

    let mut y_mat = w_sq;
    let mut z_mat = SMatrix::<f64, D, D>::identity();
    for _ in 0..MAX_ITERS {
        let y_inv = y_mat.try_inverse()?;
        let z_inv = z_mat.try_inverse()?;
        let y_new = (y_mat + z_inv) * 0.5;
        let z_new = (z_mat + y_inv) * 0.5;

        // Convergence: |Y_new − Y_old|_max
        let mut max_step = 0.0_f64;
        for i in 0..D {
            for j in 0..D {
                let d = (y_new[(i, j)] - y_mat[(i, j)]).abs();
                if d > max_step { max_step = d; }
            }
        }
        y_mat = y_new;
        z_mat = z_new;
        if max_step < TOL {
            break;
        }
    }
    // Final sanity: Y · Z ≈ I.
    let prod = y_mat * z_mat;
    for i in 0..D {
        for j in 0..D {
            let want = if i == j { 1.0 } else { 0.0 };
            if (prod[(i, j)] - want).abs() > 1.0e-9 {
                return None;
            }
        }
    }
    Some((y_mat, z_mat))
}

/// **Exact** closed-form Nesterov-Todd scaling for the second-order cone, via
/// unit-determinant **normalized points** (the form production SOC solvers
/// — ECOS, Clarabel, MOSEK — use). Returns the symmetric PD pair `(W, W⁻¹)`
/// with the defining NT property
/// ```text
///   W·s = W⁻¹·y  =: λ     (the common scaling point)
///   ⇔  W²·s = y           (so `M = W²` in `H = Gᵀ·W²·G` satisfies `M·s = y`)
/// ```
/// holding **exactly** (not the operator-geometric-mean approximation of
/// [`soc_w_squared`] / [`soc_nt_w_and_inverse`], which is only exact for
/// aligned bars `s̄ ∥ ȳ`).
///
/// **Why this is the fix for vanishing cones.** The geometric-mean form builds
/// `arrow(s)^{−1/2}`, whose smallest eigenvalue is `1/√(s₀−‖s̄‖)` — it
/// **overflows** as the cone approaches its boundary (`s₀ → ‖s̄‖`, i.e.
/// `det(s) → 0`) REGARDLESS of `y`, which is exactly what the SCvx virtual-
/// control relaxation produces at the optimum (`ν → 0`). The normalized-point
/// form works with `s̄ = s/√det(s)`, `ȳ = y/√det(y)` and `η = (det y/det s)^{1/4}`
/// (a determinant *ratio* — it stays O(1) when both determinants vanish
/// together). The scaling point `w̄ = (J·s̄ + ȳ)/(2γ)` stays bounded **on the
/// central path**: the centering condition `s ∘ y ≈ μe` keeps `s̄ ≈ ȳ` per cone,
/// so `w̄ ≈ e` (the identity) and `W ≈ η·I` even as both determinants → 0.
/// (Unit determinant alone does NOT bound `w̄`: for ANTI-aligned `s̄`, `ȳ`
/// riding the boundary in opposite bar directions, `w̄₀ ~ 1/√det` still grows —
/// but real IPM iterates are near-complementary, so that regime does not
/// arise.) That is the numerical mechanism by which production solvers keep NT
/// stable on the cones that touch their boundary at the solution.
///
/// Construction (Vandenberghe, "The CVXOPT cone program solvers", §5.1):
/// ```text
///   s̄ = s/√det(s),  ȳ = y/√det(y)                  (unit determinant)
///   γ = √((1 + s̄ᵀȳ)/2)                              (≥ 1, Euclidean dot)
///   w̄ = (J·s̄ + ȳ)/(2γ),   J = diag(1,−1,…,−1)       (det(w̄)=1; verified)
///   W = η·W̄,  η = (det y/det s)^{1/4}
///   W̄ = [[ w̄₀,  w̄_bᵀ ], [ w̄_b,  I + w̄_b·w̄_bᵀ/(1+w̄₀) ]]   (spin automorphism)
///   W̄⁻¹ = J·W̄·J  (same block form built from (w̄₀, −w̄_b))
/// ```
///
/// Generic over const `D` — needs only `soc_det`, `sqrt`, dot products and
/// `D×D` matrix arithmetic, so (unlike the eigendecomp / determinant-scaled
/// Denman-Beavers paths) it works for any cone dimension with no per-`D`
/// specialization.
///
/// Returns `None` if `s` or `y` is non-interior (`det ≤ 0`) or any intermediate
/// is non-finite.
pub fn soc_nt_scaling_exact<const D: usize>(
    s: &SVector<f64, D>,
    y: &SVector<f64, D>,
) -> Option<(SMatrix<f64, D, D>, SMatrix<f64, D, D>)> {
    // det(z) = z₀² − ‖z̄‖²  (strictly positive in the interior).
    let mut sb = 0.0;
    let mut yb = 0.0;
    for i in 1..D {
        sb += s[i] * s[i];
        yb += y[i] * y[i];
    }
    let ds = s[0] * s[0] - sb;
    let dy = y[0] * y[0] - yb;
    // Reject non-interior (`det ≤ 0`) or non-finite. `!is_finite()` catches NaN
    // (which `det ≤ 0` alone would miss, since `NaN ≤ 0` is false).
    if !ds.is_finite() || ds <= 0.0 || !dy.is_finite() || dy <= 0.0 {
        return None;
    }
    let rds = sqrt(ds);
    let rdy = sqrt(dy);

    // dot = s̄ᵀȳ = (sᵀy)/(√det s · √det y)  (Euclidean inner product).
    let mut sy = 0.0;
    for i in 0..D {
        sy += s[i] * y[i];
    }
    let dot = sy / (rds * rdy);
    // `g2 ≥ 1` (and, below, `denom = 1+w̄₀ ≥ 2`) in exact arithmetic for interior
    // points, so the `≤ 0` arms are unreachable there — they exist only so the
    // `!is_finite()` companion catches a NaN that slipped through `dot`.
    let g2 = (1.0 + dot) * 0.5;
    if !g2.is_finite() || g2 <= 0.0 {
        return None;
    }
    let gamma = sqrt(g2);
    let inv_2g = 1.0 / (2.0 * gamma);

    // w̄ = (J·s̄ + ȳ)/(2γ): head adds, bar is ȳ_bar − s̄_bar (J negates the bar
    // of s̄). This orientation is the one consistent with η = (det y/det s)^{1/4}
    // and W²·s = y (verified against the geometric-mean form on aligned bars).
    let mut w = SVector::<f64, D>::zeros();
    w[0] = (s[0] / rds + y[0] / rdy) * inv_2g;
    for i in 1..D {
        w[i] = (y[i] / rdy - s[i] / rds) * inv_2g;
    }
    let w0 = w[0];
    let denom = 1.0 + w0;
    if !denom.is_finite() || denom <= 0.0 {
        return None;
    }
    let inv_denom = 1.0 / denom;

    // η = (det y/det s)^{1/4}  (so that W²·s = y; verified for D=1: W²=y₀/s₀).
    let eta = libm::pow(dy / ds, 0.25);
    if !eta.is_finite() || eta <= 0.0 {
        return None;
    }
    let inv_eta = 1.0 / eta;

    // W = η·W̄, W⁻¹ = (1/η)·(W̄ with the bar of w̄ negated). The bar–bar block
    // `I + w̄_b·w̄_bᵀ/(1+w̄₀)` is identical for both (sign cancels); only the
    // head↔bar cross terms flip sign.
    let mut wmat = SMatrix::<f64, D, D>::zeros();
    let mut winv = SMatrix::<f64, D, D>::zeros();
    wmat[(0, 0)] = eta * w0;
    winv[(0, 0)] = inv_eta * w0;
    for i in 1..D {
        wmat[(0, i)] = eta * w[i];
        wmat[(i, 0)] = eta * w[i];
        winv[(0, i)] = inv_eta * (-w[i]);
        winv[(i, 0)] = inv_eta * (-w[i]);
    }
    for i in 1..D {
        for j in 1..D {
            let val = (if i == j { 1.0 } else { 0.0 }) + w[i] * w[j] * inv_denom;
            wmat[(i, j)] = eta * val;
            winv[(i, j)] = inv_eta * val;
        }
    }
    // Defensive finiteness gate (the cancellation keeps these bounded, but a
    // pathological near-underflow `det` could still slip through).
    for i in 0..D {
        for j in 0..D {
            if !wmat[(i, j)].is_finite() || !winv[(i, j)].is_finite() {
                return None;
            }
        }
    }
    Some((wmat, winv))
}

// ===========================================================================
// Unit tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Approximate equality of two f64 slices, element-wise.
    fn close(a: &[f64], b: &[f64], eps: f64) -> bool {
        if a.len() != b.len() {
            return false;
        }
        for k in 0..a.len() {
            if (a[k] - b[k]).abs() > eps {
                return false;
            }
        }
        true
    }

    #[test]
    fn det_signs_correct() {
        // Inside: (2, 1, 0) — det = 4 − 1 = 3 > 0
        assert!((soc_det(&[2.0, 1.0, 0.0]) - 3.0).abs() < 1e-12);
        // Boundary: (1, 1, 0) — det = 0
        assert!(soc_det(&[1.0, 1.0, 0.0]).abs() < 1e-12);
        // Outside: (1, 2, 0) — det = 1 − 4 = −3
        assert!((soc_det(&[1.0, 2.0, 0.0]) - (-3.0)).abs() < 1e-12);
    }

    #[test]
    fn membership() {
        assert!(soc_in_cone(&[2.0, 1.0, 0.0]));
        assert!(soc_in_interior(&[2.0, 1.0, 0.0]));
        assert!(soc_in_cone(&[1.0, 1.0, 0.0])); // boundary
        assert!(!soc_in_interior(&[1.0, 1.0, 0.0]));
        assert!(!soc_in_cone(&[0.5, 1.0, 0.0]));
    }

    #[test]
    fn e_is_identity_under_jordan_product() {
        let mut e = [0.0f64; 4];
        soc_e(&mut e);
        let u = [3.0, 1.0, -2.0, 0.5];
        let mut out = [0.0f64; 4];
        soc_jordan_product(&e, &u, &mut out);
        assert!(close(&out, &u, 1e-12), "e ∘ u must equal u, got {:?}", out);
    }

    #[test]
    fn jordan_product_first_component_is_inner_product() {
        let u = [2.0, 1.0, -1.0];
        let v = [3.0, 4.0, 2.0];
        // u·v = 2·3 + 1·4 + (-1)·2 = 8
        let mut out = [0.0f64; 3];
        soc_jordan_product(&u, &v, &mut out);
        assert!((out[0] - 8.0).abs() < 1e-12);
        // tail: u₀ v̄ + v₀ ū = 2·(4,2) + 3·(1,-1) = (11, 1)
        assert!((out[1] - 11.0).abs() < 1e-12);
        assert!((out[2] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn sqrt_squared_recovers_input() {
        // For s ∈ int(K), (s^(1/2)) ∘ (s^(1/2)) = s.
        let s = [5.0, 2.0, 1.0, -1.0];
        let mut root = [0.0; 4];
        soc_sqrt(&s, &mut root);
        let mut squared = [0.0; 4];
        soc_jordan_product(&root, &root, &mut squared);
        assert!(close(&s, &squared, 1e-10), "got {:?}", squared);
    }

    #[test]
    fn project_inside_is_identity() {
        let z = [2.0, 1.0, 0.5];
        let mut out = [0.0; 3];
        soc_project(&z, &mut out);
        assert!(close(&z, &out, 1e-15));
    }

    #[test]
    fn project_polar_is_zero() {
        // z ∈ −K  iff  (−z) ∈ K. Take −(2, 1, 0).5)) = (-2, -1, -0.5)
        let z = [-2.0, -1.0, -0.5];
        let mut out = [0.0; 3];
        soc_project(&z, &mut out);
        assert!(close(&out, &[0.0, 0.0, 0.0], 1e-15));
    }

    #[test]
    fn project_blend_lands_on_boundary() {
        // z = (0, 3, 4): ‖z̄‖ = 5, z₀ = 0 — neither in K nor −K.
        // Blend: scale = (0 + 5)/2 = 2.5; factor = 2.5/5 = 0.5
        // out = (2.5, 1.5, 2.0); det = 6.25 − (2.25+4) = 0 ✓
        let z = [0.0, 3.0, 4.0];
        let mut out = [0.0; 3];
        soc_project(&z, &mut out);
        assert!(close(&out, &[2.5, 1.5, 2.0], 1e-12), "got {:?}", out);
        assert!(soc_det(&out).abs() < 1e-12);
    }

    #[test]
    fn max_step_into_origin() {
        // From z = (2, 0, 0) along dz = (-1, 0, 0) — boundary hit at α = 2.
        let z  = [2.0, 0.0, 0.0];
        let dz = [-1.0, 0.0, 0.0];
        let a = soc_max_step(&z, &dz);
        assert!((a - 2.0).abs() < 1e-12, "got α = {}", a);
    }

    #[test]
    fn max_step_into_cone_is_unbounded() {
        // From z = (2, 0.5, 0.5) along dz = (1, 0, 0) — strictly into cone.
        let z  = [2.0, 0.5, 0.5];
        let dz = [1.0, 0.0, 0.0];
        let a = soc_max_step(&z, &dz);
        assert!(a.is_infinite() && a > 0.0, "expected +∞, got {}", a);
    }

    #[test]
    fn arrow_matrix_times_e_recovers_z() {
        // arrow(z) · e = z, by definition of the Jordan algebra identity.
        use nalgebra::SVector;
        let z = SVector::<f64, 4>::from_column_slice(&[3.0, 1.0, -2.0, 0.5]);
        let mut e = SVector::<f64, 4>::zeros();
        e[0] = 1.0;
        let m = soc_arrow_matrix(&z);
        let r = m * e;
        for i in 0..4 {
            assert!((r[i] - z[i]).abs() < 1e-12, "row {i}: {} vs {}", r[i], z[i]);
        }
    }

    #[test]
    fn arrow_matrix_is_symmetric() {
        use nalgebra::SVector;
        let z = SVector::<f64, 5>::from_column_slice(&[7.0, 1.0, 2.0, -1.0, 3.0]);
        let m = soc_arrow_matrix(&z);
        for i in 0..5 {
            for j in 0..5 {
                assert!((m[(i, j)] - m[(j, i)]).abs() < 1e-15);
            }
        }
    }

    #[test]
    fn arrow_matrix_jordan_product_matches_matvec() {
        // arrow(u) · v should equal u ∘ v.
        use nalgebra::SVector;
        let u = SVector::<f64, 4>::from_column_slice(&[5.0, 2.0, -1.0, 0.5]);
        let v = SVector::<f64, 4>::from_column_slice(&[3.0, 1.0, 2.0, -2.0]);
        let from_matrix = soc_arrow_matrix(&u) * v;
        let u_slice = [5.0, 2.0, -1.0, 0.5];
        let v_slice = [3.0, 1.0, 2.0, -2.0];
        let mut from_jordan = [0.0; 4];
        soc_jordan_product(&u_slice, &v_slice, &mut from_jordan);
        for i in 0..4 {
            assert!(
                (from_matrix[i] - from_jordan[i]).abs() < 1e-12,
                "row {i}"
            );
        }
    }

    /// Approximate-NT residual: M = W^{−2} should satisfy M·y ≈ s.
    /// Exact for `s_bar ∥ y_bar`; approximate otherwise (with residual
    /// bounded by misalignment angle).
    fn nt_matrix_residual<const D: usize>(s_arr: &[f64], y_arr: &[f64]) -> f64
    where
        nalgebra::Const<D>: nalgebra::Dim,
    {
        let s = nalgebra::SVector::<f64, D>::from_column_slice(s_arr);
        let y = nalgebra::SVector::<f64, D>::from_column_slice(y_arr);
        let m = soc_nt_scaling_matrix::<D>(&s, &y).expect("NT matrix");
        let got = m * y;
        let mut sq = 0.0;
        for i in 0..D {
            sq += (got[i] - s[i]) * (got[i] - s[i]);
        }
        sqrt(sq)
    }

    /// M is symmetric (key property for IPM conditioning).
    fn nt_matrix_symmetry<const D: usize>(s_arr: &[f64], y_arr: &[f64]) -> f64
    where
        nalgebra::Const<D>: nalgebra::Dim,
    {
        let s = nalgebra::SVector::<f64, D>::from_column_slice(s_arr);
        let y = nalgebra::SVector::<f64, D>::from_column_slice(y_arr);
        let m = soc_nt_scaling_matrix::<D>(&s, &y).expect("NT matrix");
        let diff = m - m.transpose();
        diff.iter().map(|v| v.abs()).fold(0.0_f64, f64::max)
    }

    #[test]
    fn nt_matrix_aligned_is_exact() {
        // Aligned cases (commuting arrow operators) should satisfy M·y = s exactly.
        assert!(nt_matrix_residual::<3>(&[2.0, 0.0, 0.0], &[1.0, 0.0, 0.0]) < 1e-12);
        assert!(nt_matrix_residual::<3>(&[5.0, 3.0, 0.0], &[3.0, 1.0, 0.0]) < 1e-10);
        // Pure boost along (1, 0) direction in both — aligned.
        assert!(nt_matrix_residual::<4>(&[5.0, 2.0, 0.0, 0.0], &[3.0, 1.0, 0.0, 0.0]) < 1e-10);
    }

    #[test]
    fn nt_matrix_misaligned_is_approximate() {
        // Misaligned bars give an approximate (not exact) NT scaling, but
        // the residual must be small. This is the "geometric-mean" property.
        let r1 = nt_matrix_residual::<4>(&[3.0, 1.0, -0.5, 0.7], &[2.0, -0.3, 0.4, 0.1]);
        assert!(r1 < 5.0e-2, "misaligned 4D residual = {}", r1);
        let r2 = nt_matrix_residual::<3>(&[3.0, 1.0, 0.0], &[3.0, 0.0, 1.0]);
        assert!(r2 < 5.0e-2, "perpendicular 3D residual = {}", r2);
    }

    /// `W · W⁻¹ ≈ I` to machine precision — basic invariant of the matrix
    /// square root + inverse pair.
    #[test]
    fn nt_w_and_inverse_compose_to_identity() {
        use nalgebra::SVector;
        let s = SVector::<f64, 4>::from_column_slice(&[3.0, 1.0, -0.5, 0.7]);
        let y = SVector::<f64, 4>::from_column_slice(&[2.0, -0.3, 0.4, 0.1]);
        let (w, w_inv) = soc_nt_w_and_inverse::<4>(&s, &y).expect("NT W");
        let prod = w * w_inv;
        for i in 0..4 {
            for j in 0..4 {
                let expect = if i == j { 1.0 } else { 0.0 };
                assert!((prod[(i, j)] - expect).abs() < 1e-12,
                        "(W·W⁻¹)[{i},{j}] = {} (expect {})", prod[(i, j)], expect);
            }
        }
    }

    /// `W · W = W²` to machine precision — confirms the matrix-sqrt
    /// is the right one for the operator geometric mean.
    #[test]
    fn nt_w_squared_equals_w_times_w() {
        use nalgebra::SVector;
        let s = SVector::<f64, 5>::from_column_slice(&[5.0, 1.0, -0.5, 0.7, -0.4]);
        let y = SVector::<f64, 5>::from_column_slice(&[3.0, -0.3, 0.4, 0.1, 0.2]);
        let (w, _w_inv) = soc_nt_w_and_inverse::<5>(&s, &y).expect("NT W");
        let w_sq_via_sqrt = w * w;
        let w_sq_direct   = soc_w_squared::<5>(&s, &y).expect("W²");
        for i in 0..5 {
            for j in 0..5 {
                assert!((w_sq_via_sqrt[(i, j)] - w_sq_direct[(i, j)]).abs() < 1e-11,
                        "[{i},{j}]: via_sqrt={} direct={}",
                        w_sq_via_sqrt[(i, j)], w_sq_direct[(i, j)]);
            }
        }
    }

    /// `W` is symmetric (it's a symmetric matrix sqrt).
    #[test]
    fn nt_w_is_symmetric() {
        use nalgebra::SVector;
        let s = SVector::<f64, 4>::from_column_slice(&[3.0, 1.0, -0.5, 0.7]);
        let y = SVector::<f64, 4>::from_column_slice(&[2.0, -0.3, 0.4, 0.1]);
        let (w, w_inv) = soc_nt_w_and_inverse::<4>(&s, &y).expect("NT W");
        for i in 0..4 {
            for j in 0..4 {
                assert!((w[(i, j)] - w[(j, i)]).abs() < 1e-13);
                assert!((w_inv[(i, j)] - w_inv[(j, i)]).abs() < 1e-13);
            }
        }
    }

    /// NT scaling at every cone dimension our SCvx subproblem uses
    /// (`D ∈ {1, 3, 4, 8, 11}`). Each must return a finite `(W, W⁻¹)` pair
    /// satisfying `W·W⁻¹ ≈ I` and `W = Wᵀ`.
    #[test]
    fn nt_w_works_at_scvx_cone_dims() {
        use nalgebra::SVector;

        // Helper closure: verify (W, W⁻¹) sanity for one set of (s, y).
        fn check<const D: usize>(s_arr: &[f64], y_arr: &[f64]) {
            let s = SVector::<f64, D>::from_column_slice(s_arr);
            let y = SVector::<f64, D>::from_column_slice(y_arr);
            let (w, w_inv) = soc_nt_w_and_inverse::<D>(&s, &y)
                .unwrap_or_else(|| panic!("D={D}: NT W failed"));
            // Symmetry.
            for i in 0..D {
                for j in 0..D {
                    assert!((w[(i, j)] - w[(j, i)]).abs() < 1e-12,
                            "D={D}: W not symmetric at ({i},{j})");
                    assert!((w_inv[(i, j)] - w_inv[(j, i)]).abs() < 1e-12,
                            "D={D}: W⁻¹ not symmetric at ({i},{j})");
                }
            }
            // W·W⁻¹ ≈ I.
            let prod = w * w_inv;
            for i in 0..D {
                for j in 0..D {
                    let expect = if i == j { 1.0 } else { 0.0 };
                    assert!((prod[(i, j)] - expect).abs() < 1e-10,
                            "D={D}: (W·W⁻¹)[{i},{j}] = {} (expect {})",
                            prod[(i, j)], expect);
                }
            }
            // No NaN/inf.
            for i in 0..D {
                for j in 0..D {
                    assert!(w[(i, j)].is_finite(),     "D={D}: W has NaN/inf");
                    assert!(w_inv[(i, j)].is_finite(), "D={D}: W⁻¹ has NaN/inf");
                }
            }
        }

        // D=1: trivial scalar case (SOC^1 = ℝ_+).
        check::<1>(&[3.0], &[2.0]);
        check::<1>(&[100.0], &[0.5]);

        // D=3: glide-slope cone.
        check::<3>(&[5.0, 1.0, -2.0], &[3.0, 0.5, 1.0]);

        // D=4: thrust-magnitude cone.
        check::<4>(&[10.0, 1.0, -2.0, 3.0], &[2.0, 0.5, 0.3, -0.4]);

        // D=8: virtual-control L2 cone.
        check::<8>(
            &[5.0, 0.5, -0.3, 0.1, -0.2, 0.4, 0.0, 0.1],
            &[3.0, 0.1, 0.2, -0.1, 0.3, -0.2, 0.4, 0.0],
        );

        // D=11: trust-region cone — strictly interior with mixed-scale
        // bar entries. `s_0` must exceed `‖s_bar‖₂` (cone interior); we
        // pick generously interior numbers to make the test robust.
        // ‖s_bar‖ here ≈ sqrt(1+1+4+0.04+0.01+50²) ≈ 50.06 < s_0 = 200.
        check::<11>(
            &[200.0,  1.0,  0.0,  1.0,  0.0,  0.0,  2.0,  0.2,  0.1,  0.0, 50.0],
            &[150.0,  0.5,  0.0,  0.8,  0.0,  0.0,  1.5,  0.1,  0.0,  0.0, 30.0],
        );
    }

    #[test]
    fn nt_matrix_is_always_symmetric() {
        // M MUST be symmetric for IPM Schur factorization to behave well.
        // This is non-negotiable — even when M·y ≠ s exactly, symmetry holds.
        assert!(nt_matrix_symmetry::<4>(&[3.0, 1.0, -0.5, 0.7], &[2.0, -0.3, 0.4, 0.1]) < 1e-12);
        assert!(nt_matrix_symmetry::<3>(&[3.0, 1.0, 0.0], &[3.0, 0.0, 1.0]) < 1e-12);
        assert!(nt_matrix_symmetry::<11>(
            &[5.0,  1.0, 0.5, -0.3, 0.7, -0.4, 0.2, 0.1, -0.2, 0.3, -0.1],
            &[3.0, -0.4, 0.3,  0.2, 0.1, -0.5, 0.4, 0.0, -0.3, 0.2, -0.2],
        ) < 1e-10);
    }

    #[test]
    fn arrow_inv_sqrt_squares_to_arrow_inverse() {
        use nalgebra::{SMatrix, SVector};
        let s = SVector::<f64, 4>::from_column_slice(&[3.0, 1.0, -0.5, 0.7]);
        let inv_half = soc_arrow_inv_sqrt::<4>(&s).unwrap();
        let arrow_s = soc_arrow_matrix(&s);
        let product = inv_half * inv_half * arrow_s;
        let identity = SMatrix::<f64, 4, 4>::identity();
        let diff = product - identity;
        let max_err = diff.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        assert!(max_err < 1e-12, "max err = {}", max_err);
    }

    #[test]
    fn nt_matrix_random_sweep_stays_bounded() {
        // For random pairs (s, y) ∈ int(K), the M matrix must be:
        // (1) symmetric to machine precision
        // (2) finite (no NaN/inf)
        // (3) approximately satisfy M·y ≈ s with residual bounded (< 0.1)
        let mut seed: u64 = 0x1357_2468_ACE0_BDF1;
        let next = |s: &mut u64| -> f64 {
            *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((*s >> 11) as f64) / ((1u64 << 53) as f64)
        };
        let mut max_res = 0.0_f64;
        let mut max_asym = 0.0_f64;
        for _trial in 0..50 {
            let s4 = [
                2.0 + next(&mut seed) * 3.0,
                (next(&mut seed) - 0.5) * 0.8,
                (next(&mut seed) - 0.5) * 0.8,
                (next(&mut seed) - 0.5) * 0.8,
            ];
            let y4 = [
                2.0 + next(&mut seed) * 3.0,
                (next(&mut seed) - 0.5) * 0.8,
                (next(&mut seed) - 0.5) * 0.8,
                (next(&mut seed) - 0.5) * 0.8,
            ];
            let r    = nt_matrix_residual::<4>(&s4, &y4);
            let asym = nt_matrix_symmetry::<4>(&s4, &y4);
            if r    > max_res  { max_res  = r;    }
            if asym > max_asym { max_asym = asym; }
            assert!(asym < 1e-12, "asymmetry {}", asym);
            assert!(r    < 0.1,    "residual {}", r);
            assert!(r.is_finite(), "non-finite residual");
        }
        extern crate std;
        std::eprintln!(
            "NT-matrix-form sweep (50 random 4D trials): max residual = {:.3e}, max asymmetry = {:.3e}",
            max_res, max_asym
        );
    }

    #[test]
    fn nt_matrix_rejects_boundary_input() {
        // s on the boundary (det = 0) should yield None from NT matrix builder.
        use nalgebra::SVector;
        let s_boundary = SVector::<f64, 3>::from_column_slice(&[1.0, 1.0, 0.0]); // det = 0
        let y_ok       = SVector::<f64, 3>::from_column_slice(&[2.0, 0.5, 0.0]);
        assert!(soc_nt_scaling_matrix::<3>(&s_boundary, &y_ok).is_none());
        assert!(soc_nt_scaling_matrix::<3>(&y_ok, &s_boundary).is_none());
    }

    #[test]
    fn max_step_tangential() {
        // From z = (2, 0, 0) ∈ int(K) along dz = (0, 1, 0).
        // det(z + αdz) = 4 − α² = 0 ⇒ α = 2.
        let z  = [2.0, 0.0, 0.0];
        let dz = [0.0, 1.0, 0.0];
        let a  = soc_max_step(&z, &dz);
        assert!((a - 2.0).abs() < 1e-12, "got α = {}", a);
    }

    /// Exact NT scaling must satisfy `W²·s = y` (the defining property) and
    /// `W·s = W⁻¹·y` to machine precision, at every SCvx cone dim, plus the
    /// basics (`W·W⁻¹ = I`, symmetry, finiteness).
    #[test]
    fn exact_nt_scaling_satisfies_w_squared_s_equals_y() {
        use nalgebra::SVector;
        fn check<const D: usize>(s_arr: &[f64], y_arr: &[f64], tol: f64, tag: &str) {
            let s = SVector::<f64, D>::from_column_slice(s_arr);
            let y = SVector::<f64, D>::from_column_slice(y_arr);
            let (w, w_inv) = soc_nt_scaling_exact::<D>(&s, &y)
                .unwrap_or_else(|| panic!("{tag}: exact NT returned None"));
            let w2s = w * (w * s);            // W²·s
            let ws  = w * s;                  // scaling point λ = W·s
            let wiy = w_inv * y;              // = W⁻¹·y, must equal λ
            for i in 0..D {
                assert!((w2s[i] - y[i]).abs() < tol,
                    "{tag}: (W²s−y)[{i}] = {:.3e}", (w2s[i] - y[i]).abs());
                assert!((ws[i] - wiy[i]).abs() < tol,
                    "{tag}: (Ws−W⁻¹y)[{i}] = {:.3e}", (ws[i] - wiy[i]).abs());
            }
            let prod = w * w_inv;
            for i in 0..D {
                for j in 0..D {
                    let e = if i == j { 1.0 } else { 0.0 };
                    assert!((prod[(i, j)] - e).abs() < 1e-9, "{tag}: W·W⁻¹ off ({i},{j})");
                    assert!((w[(i, j)] - w[(j, i)]).abs() < 1e-12, "{tag}: W asymmetric");
                    assert!(w[(i, j)].is_finite() && w_inv[(i, j)].is_finite(), "{tag}: non-finite");
                }
            }
        }
        check::<1>(&[3.0], &[2.0], 1e-12, "D1");
        check::<3>(&[5.0, 1.0, -2.0], &[3.0, 0.5, 1.0], 1e-10, "D3");
        check::<4>(&[10.0, 1.0, -2.0, 3.0], &[2.0, 0.5, 0.3, -0.4], 1e-10, "D4");
        check::<8>(
            &[5.0, 0.5, -0.3, 0.1, -0.2, 0.4, 0.0, 0.1],
            &[3.0, 0.1, 0.2, -0.1, 0.3, -0.2, 0.4, 0.0],
            1e-10, "D8",
        );
        check::<11>(
            &[200.0, 1.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.2, 0.1, 0.0, 50.0],
            &[150.0, 0.5, 0.0, 0.8, 0.0, 0.0, 1.5, 0.1, 0.0, 0.0, 30.0],
            1e-9, "D11",
        );
    }

    /// THE vanishing-cone fix, scoped honestly. As a cone vanishes (`det → 0`,
    /// the SCvx virtual-control relaxation at the optimum), the EXACT NT scaling
    /// stays bounded **on the central path** — where the centering condition
    /// `s ∘ y ≈ μe` keeps `s̄ ≈ ȳ` per cone, so `w̄ ≈ e` and `W ≈ η·I`. The
    /// geometric-mean form's `arrow(s)^{−1/2}` (eigenvalue `1/√(s₀−‖s̄‖)`) would
    /// blow up to `~1/eps` here REGARDLESS of `y`.
    ///
    /// HONEST SCOPE: the exact form is bounded because of near-complementarity,
    /// NOT because of unit determinant. For ANTI-aligned `s̄`, `ȳ` (opposite bar
    /// directions — a pairing real IPM iterates never produce, since
    /// complementarity forbids it) `W` stays finite (no NaN) but GROWS like
    /// `~1/√det`. Both regimes are asserted.
    #[test]
    fn exact_nt_scaling_bounded_on_central_path_vanishing_cone() {
        extern crate std;
        use nalgebra::SVector;
        fn near_boundary<const D: usize>(bar: &[f64], eps: f64) -> SVector<f64, D> {
            let mut v = SVector::<f64, D>::zeros();
            let mut nb = 0.0;
            for i in 1..D {
                v[i] = bar[i - 1];
                nb += bar[i - 1] * bar[i - 1];
            }
            v[0] = nb.sqrt() + eps; // det = v₀² − ‖bar‖² ≈ 2‖bar‖·eps → 0
            v
        }
        let sbar = [0.30, -0.20, 0.50, 0.10, -0.40, 0.20, 0.10];
        let ybar = [0.31, -0.19, 0.52, 0.08, -0.41, 0.18, 0.12]; // near-complementary

        // (a) CENTRAL PATH (s̄ ≈ ȳ): W bounded ~η, W²s = y tight, at every eps.
        for &eps in &[1e-3, 1e-5, 1e-7, 1e-9] {
            let s = near_boundary::<8>(&sbar, eps);
            let y = near_boundary::<8>(&ybar, eps);
            let (w, _wi) = soc_nt_scaling_exact::<8>(&s, &y)
                .unwrap_or_else(|| panic!("central eps={eps:.0e}: exact NT None"));
            let w2s = w * (w * s);
            let mut maxerr = 0.0_f64;
            let mut maxw = 0.0_f64;
            for i in 0..8 {
                maxerr = maxerr.max((w2s[i] - y[i]).abs());
                assert!(w2s[i].is_finite(), "central eps={eps:.0e}: W²s non-finite");
                for j in 0..8 {
                    maxw = maxw.max(w[(i, j)].abs());
                }
            }
            std::eprintln!("central eps={eps:.0e}: max|W²s−y|={maxerr:.2e} max|W|={maxw:.2e}");
            assert!(maxerr < 1e-4, "central eps={eps:.0e}: W²s−y = {maxerr:.2e} (blew up?)");
            assert!(maxw.is_finite() && maxw < 10.0, "central eps={eps:.0e}: W not ~η ({maxw:.2e})");
        }

        // (b) ANTI-ALIGNED (ȳ bar = −s̄ bar): W stays FINITE (the finiteness gate
        // holds — no NaN) but GROWS as det → 0, confirming the bound is
        // central-path-dependent, not universal. Not produced by a real IPM.
        let mut prev = 0.0_f64;
        for &eps in &[1e-3, 1e-5, 1e-7] {
            let s = near_boundary::<8>(&sbar, eps);
            let mut yneg = sbar;
            for v in yneg.iter_mut() {
                *v = -*v;
            }
            let y = near_boundary::<8>(&yneg, eps);
            let (w, _wi) = soc_nt_scaling_exact::<8>(&s, &y)
                .unwrap_or_else(|| panic!("anti eps={eps:.0e}: exact NT None"));
            let mut maxw = 0.0_f64;
            for i in 0..8 {
                for j in 0..8 {
                    assert!(w[(i, j)].is_finite(), "anti eps={eps:.0e}: W non-finite");
                    maxw = maxw.max(w[(i, j)].abs());
                }
            }
            std::eprintln!("anti-aligned eps={eps:.0e}: max|W|={maxw:.2e} (grows ~1/√det)");
            assert!(maxw > prev, "anti eps={eps:.0e}: |W| should grow as det→0");
            prev = maxw;
        }
    }

    /// The exact NT scaling (boost form) and the geometric-mean approximation
    /// both satisfy `W²·s = y` and AGREE on the subspace where `s, y` have
    /// support. They differ only in the transverse/null directions — there the
    /// boost form uses the canonical scalar `η² = √(det y/det s)` (it is a true
    /// cone automorphism) while the arrow-based geomean does not — but that
    /// difference cannot affect `W²·s`. Verify the shared, meaningful behaviour.
    #[test]
    fn exact_nt_and_geomean_agree_on_supported_action() {
        use nalgebra::SVector;
        let s = SVector::<f64, 4>::from_column_slice(&[5.0, 2.0, 0.0, 0.0]);
        let y = SVector::<f64, 4>::from_column_slice(&[3.0, 1.0, 0.0, 0.0]);
        let (w_exact, _) = soc_nt_scaling_exact::<4>(&s, &y).unwrap();
        let w2_exact = w_exact * w_exact;
        let w2_geo = soc_w_squared::<4>(&s, &y).unwrap();
        // Both map s → y.
        for (m, tag) in [(w2_exact, "exact"), (w2_geo, "geo")] {
            let ms = m * s;
            for i in 0..4 {
                assert!((ms[i] - y[i]).abs() < 1e-9, "{tag}: (W²s−y)[{i}] != 0");
            }
        }
        // They agree on the action on any vector supported in span{e0, e1}
        // (zero transverse component) — check on s and y themselves.
        for v in [s, y] {
            let a = w2_exact * v;
            let b = w2_geo * v;
            for i in 0..4 {
                assert!(
                    (a[i] - b[i]).abs() < 1e-9,
                    "exact vs geo differ on supported action at {i}: {} vs {}", a[i], b[i]
                );
            }
        }
    }
}
