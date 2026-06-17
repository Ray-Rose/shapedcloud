//! Three-way oracle diff test: Rust IPM vs CVXPY/Clarabel vs Julia/Clarabel.
//!
//! Reference values from running:
//! - `tools/py-oracle/Scripts/python.exe tools/py-oracle/solve_canonical.py`
//! - `julia --project=tools/jl-oracle tools/jl-oracle/solve_canonical.jl`
//!
//! Both oracles agreed to ~1e-9 on all three canonical problems. The Rust
//! IPM is expected to agree to ~1e-3 (AHO endgame ceiling — NT scaling
//! would tighten this; see P1b notes in cone.rs / socp.rs).
//!
//! This is the P8 and P9 deliverable in a single test file.

use nalgebra::{SMatrix, SVector};
use scvx_core::{IpmAlgoParams, IpmStatus};
use scvx_ipm::{
    solve_socp, solve_socp_hsd, solve_socp_nt, ConeDesc, SocpProblem, SocpResult, SocpWorkspace,
};

/// **Self-consistent optimality oracle** (no external reference needed):
/// verify the returned (x, λ, s, y) actually satisfy the SOCP KKT conditions —
/// primal feasibility `A·x=b`, `G·x+s=h`; dual feasibility `c+Aᵀλ+Gᵀy=0`;
/// complementarity `s·y≈0`; and `s, y ∈ K` (each cone interior). This validates
/// that the solver found a TRUE optimum, not just that it returned `Optimal`.
fn assert_kkt_optimal<
    const NP: usize, const NE: usize, const NCT: usize, const NCONES: usize,
>(
    label: &str,
    prob: &SocpProblem<NP, NE, NCT, NCONES>,
    res:  &SocpResult<NP, NE, NCT>,
    tol:  f64,
) {
    let r_a = (prob.a_mat * res.x - prob.b).norm();
    let r_g = (prob.g_mat * res.x + res.s - prob.h).norm();
    let r_d = (prob.c
        + prob.a_mat.transpose() * res.lambda
        + prob.g_mat.transpose() * res.y).norm();
    let compl = (res.s.dot(&res.y)).abs() / (NCT as f64);

    let s = res.s.as_slice();
    let y = res.y.as_slice();
    let mut min_s_marg = f64::INFINITY;
    let mut min_y_marg = f64::INFINITY;
    for cone in prob.cones.iter() {
        let (o, d) = (cone.offset, cone.dim);
        let sb = (1..d).map(|i| s[o + i] * s[o + i]).sum::<f64>().sqrt();
        let yb = (1..d).map(|i| y[o + i] * y[o + i]).sum::<f64>().sqrt();
        min_s_marg = min_s_marg.min(s[o] - sb);
        min_y_marg = min_y_marg.min(y[o] - yb);
    }
    eprintln!(
        "  KKT[{label}]: |Ax-b|={r_a:.2e} |Gx+s-h|={r_g:.2e} |c+Aᵀλ+Gᵀy|={r_d:.2e} \
         s·y/n={compl:.2e} s_marg={min_s_marg:.2e} y_marg={min_y_marg:.2e}"
    );
    assert!(r_a < tol, "{label}: primal-eq residual {r_a:.2e} >= {tol:.1e}");
    assert!(r_g < tol, "{label}: primal-cone residual {r_g:.2e} >= {tol:.1e}");
    assert!(r_d < tol, "{label}: dual-stationarity residual {r_d:.2e} >= {tol:.1e}");
    assert!(compl < tol, "{label}: complementarity {compl:.2e} >= {tol:.1e}");
    assert!(min_s_marg > -tol, "{label}: s left the cone (margin {min_s_marg:.2e})");
    assert!(min_y_marg > -tol, "{label}: y left the cone (margin {min_y_marg:.2e})");
}

/// Per-problem reference solution from CVXPY/Clarabel and Julia/Clarabel,
/// captured to 12 significant digits from oracle runs.
struct OracleReference {
    name:           &'static str,
    expected_cost:  f64,
    expected_x:     &'static [f64],
    /// Both oracles' max coordinate disagreement (the "oracle agreement floor").
    /// Rust IPM is held to this floor + AHO endgame margin (1e-3).
    oracle_floor:   f64,
}

const REF_TOY: OracleReference = OracleReference {
    name:          "toy_1cone",
    expected_cost: 0.707106776347,        // CVXPY value; Julia agrees to ~4e-10
    expected_x:    &[0.707106776347, 0.5, 0.5],
    oracle_floor:  1.0e-9,
};

const REF_TWO_CONE: OracleReference = OracleReference {
    name:          "two_cone_mixed",
    expected_cost: 2.0,                    // both oracles give 2.0 within 2e-9
    expected_x:    &[1.0, 1.0, 2.0],
    oracle_floor:  1.0e-8,
};

const REF_SOCP_4D: OracleReference = OracleReference {
    name:          "socp_4d",
    expected_cost: 1.464101615138,         // mean of CVXPY and Julia; agree to 1.2e-9
    expected_x:    &[1.464101615138, 0.845299461463, 0.845299461463, 0.845299461463],
    oracle_floor:  1.0e-8,
};

/// Tolerance for Rust vs oracle. Larger than oracle_floor because our AHO
/// IPM doesn't push as deep as Clarabel's NT-scaled IPM.
const RUST_VS_ORACLE_TOL: f64 = 1.0e-3;

// ===========================================================================

#[test]
fn rust_ipm_matches_oracles_toy_1cone() {
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

    diff_against_oracle("toy", &REF_TOY, result.status, result.x.as_slice(), prob.c.dot(&result.x));
    assert_kkt_optimal("toy/AHO", &prob, &result, 1.0e-3);

    // NT direction must reach the same external-oracle optimum on this toy
    // (NT is well-conditioned here — no vanishing cones).
    let mut ws_nt = SocpWorkspace::<NP, NE, NCT>::default();
    let res_nt = solve_socp_nt(&prob, &params, &mut ws_nt);
    diff_against_oracle("toy/NT", &REF_TOY, res_nt.status, res_nt.x.as_slice(), prob.c.dot(&res_nt.x));
    assert_kkt_optimal("toy/NT", &prob, &res_nt, 1.0e-3);

    // HSD direction must reach the same external-oracle optimum (the optimum
    // sits on the cone boundary — the vanishing-cone case the embedding handles).
    let mut ws_hsd = SocpWorkspace::<NP, NE, NCT>::default();
    let res_hsd = solve_socp_hsd(&prob, &params, &mut ws_hsd);
    diff_against_oracle("toy/HSD", &REF_TOY, res_hsd.status, res_hsd.x.as_slice(), prob.c.dot(&res_hsd.x));
    assert_kkt_optimal("toy/HSD", &prob, &res_hsd, 1.0e-3);
}

#[test]
fn rust_ipm_matches_oracles_two_cone() {
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
    let result = solve_socp(&prob, &params, &mut ws);

    diff_against_oracle("two_cone", &REF_TWO_CONE, result.status, result.x.as_slice(), prob.c.dot(&result.x));
    assert_kkt_optimal("two_cone/AHO", &prob, &result, 1.0e-3);

    let mut ws_nt = SocpWorkspace::<NP, NE, NCT>::default();
    let res_nt = solve_socp_nt(&prob, &params, &mut ws_nt);
    diff_against_oracle("two_cone/NT", &REF_TWO_CONE, res_nt.status, res_nt.x.as_slice(), prob.c.dot(&res_nt.x));
    assert_kkt_optimal("two_cone/NT", &prob, &res_nt, 1.0e-3);

    let mut ws_hsd = SocpWorkspace::<NP, NE, NCT>::default();
    let res_hsd = solve_socp_hsd(&prob, &params, &mut ws_hsd);
    diff_against_oracle("two_cone/HSD", &REF_TWO_CONE, res_hsd.status, res_hsd.x.as_slice(), prob.c.dot(&res_hsd.x));
    assert_kkt_optimal("two_cone/HSD", &prob, &res_hsd, 1.0e-3);
}

#[test]
fn rust_ipm_matches_oracles_socp_4d() {
    // P3: min x_0 s.t. (x_0, x_1, x_2, x_3) in SOC^4, sum(x) = 4.
    // s = -G·x, want s ∈ SOC^4, so G = -I_4, h = 0.
    const NP: usize     = 4;
    const NE: usize     = 1;
    const NCT: usize    = 4;
    const NCONES: usize = 1;

    let prob = SocpProblem::<NP, NE, NCT, NCONES> {
        c:     SVector::<f64, 4>::from_column_slice(&[1.0, 0.0, 0.0, 0.0]),
        a_mat: SMatrix::<f64, 1, 4>::from_row_slice(&[1.0, 1.0, 1.0, 1.0]),
        b:     SVector::<f64, 1>::from_element(4.0),
        g_mat: -SMatrix::<f64, 4, 4>::identity(),
        h:     SVector::<f64, 4>::zeros(),
        cones: [ConeDesc { offset: 0, dim: 4 }],
    };
    let mut ws = SocpWorkspace::<NP, NE, NCT>::default();
    let params = IpmAlgoParams::default();
    let result = solve_socp(&prob, &params, &mut ws);

    diff_against_oracle("socp_4d", &REF_SOCP_4D, result.status, result.x.as_slice(), prob.c.dot(&result.x));
    assert_kkt_optimal("socp_4d/AHO", &prob, &result, 1.0e-3);

    let mut ws_nt = SocpWorkspace::<NP, NE, NCT>::default();
    let res_nt = solve_socp_nt(&prob, &params, &mut ws_nt);
    diff_against_oracle("socp_4d/NT", &REF_SOCP_4D, res_nt.status, res_nt.x.as_slice(), prob.c.dot(&res_nt.x));
    assert_kkt_optimal("socp_4d/NT", &prob, &res_nt, 1.0e-3);

    let mut ws_hsd = SocpWorkspace::<NP, NE, NCT>::default();
    let res_hsd = solve_socp_hsd(&prob, &params, &mut ws_hsd);
    diff_against_oracle("socp_4d/HSD", &REF_SOCP_4D, res_hsd.status, res_hsd.x.as_slice(), prob.c.dot(&res_hsd.x));
    assert_kkt_optimal("socp_4d/HSD", &prob, &res_hsd, 1.0e-3);
}

// ===========================================================================
// Diff harness — fail loudly when Rust diverges from the oracle
// ===========================================================================

fn diff_against_oracle(
    label:  &str,
    oracle: &OracleReference,
    status: IpmStatus,
    x:      &[f64],
    cost:   f64,
) {
    eprintln!("--- {} ---", oracle.name);
    eprintln!("  Rust IPM status: {} ({})",
              status_str(status), status.as_u32());
    eprintln!("  Rust x:    {:?}", x);
    eprintln!("  Oracle x:  {:?}", oracle.expected_x);
    eprintln!("  Oracle floor (CVXPY vs Julia agreement): {:.1e}",
              oracle.oracle_floor);

    // Status must be Optimal or BestFeasible.
    assert!(
        matches!(status, IpmStatus::Optimal | IpmStatus::BestFeasible),
        "{label}: Rust IPM did not converge cleanly (status = {})",
        status.as_u32()
    );

    // Coordinate-wise match against oracle, with AHO endgame margin.
    let mut max_err = 0.0_f64;
    let mut worst_i = 0;
    for (i, (&got, &want)) in x.iter().zip(oracle.expected_x).enumerate() {
        let err = (got - want).abs();
        if err > max_err { max_err = err; worst_i = i; }
    }
    eprintln!("  max Rust↔oracle coordinate error: {:.3e}  (i={}, tol={:.1e})",
              max_err, worst_i, RUST_VS_ORACLE_TOL);
    assert!(
        max_err < RUST_VS_ORACLE_TOL,
        "{label}: Rust diverged from oracle at coord {worst_i}: \
         got {} vs {} (err {:.3e}, tol {:.1e})",
        x[worst_i], oracle.expected_x[worst_i], max_err, RUST_VS_ORACLE_TOL
    );

    // Cost agreement: the objective the solver actually minimized (`cᵀx`,
    // passed in by the caller) must match the oracle's optimal cost.
    let cost_err = (cost - oracle.expected_cost).abs();
    eprintln!("  Rust cost {cost:.12} vs oracle {:.12} (err {cost_err:.3e})",
              oracle.expected_cost);
    assert!(
        cost_err < RUST_VS_ORACLE_TOL,
        "{label}: cost {cost} vs oracle {} (err {cost_err:.3e}, tol {RUST_VS_ORACLE_TOL:.1e})",
        oracle.expected_cost
    );
}

fn status_str(s: IpmStatus) -> &'static str {
    match s {
        IpmStatus::Optimal        => "Optimal",
        IpmStatus::BestFeasible   => "BestFeasible",
        IpmStatus::Infeasible     => "Infeasible",
        IpmStatus::NumericalError => "NumericalError",
        IpmStatus::IterCap        => "IterCap",
    }
}
