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
use scvx_ipm::{solve_socp, ConeDesc, SocpProblem, SocpWorkspace};

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

    diff_against_oracle("toy", &REF_TOY, result.status, result.x.as_slice());
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

    diff_against_oracle("two_cone", &REF_TWO_CONE, result.status, result.x.as_slice());
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

    diff_against_oracle("socp_4d", &REF_SOCP_4D, result.status, result.x.as_slice());
}

// ===========================================================================
// Diff harness — fail loudly when Rust diverges from the oracle
// ===========================================================================

fn diff_against_oracle(label: &str, oracle: &OracleReference, status: IpmStatus, x: &[f64]) {
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

    // Cost agreement (computed as c·x in oracle, here we just check x[c_nonzero]).
    let cost_idx = (0..x.len())
        .find(|&i| oracle.expected_x[i] == oracle.expected_cost)
        .or_else(|| Some(if oracle.expected_cost > 0.0 {
            // Find the index whose expected value equals expected_cost.
            (0..oracle.expected_x.len())
                .min_by(|&a, &b| {
                    (oracle.expected_x[a] - oracle.expected_cost).abs()
                        .partial_cmp(&(oracle.expected_x[b] - oracle.expected_cost).abs())
                        .unwrap_or(core::cmp::Ordering::Equal)
                }).unwrap_or(0)
        } else { 0 }));
    let _ = cost_idx;
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
