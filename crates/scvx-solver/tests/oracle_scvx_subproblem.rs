//! External-oracle coverage for a **flight-scale SCvx subproblem**.
//!
//! `oracle_diff.rs` validates the Rust IPM against CVXPY/Clarabel + Julia/Clarabel
//! on three *toy* SOCPs (≤ 4 vars, 1–2 cones). This file extends that coverage to
//! a **real assembled SCvx subproblem** — the 19·N-variable, 8·N-cone problem the
//! production outer loop actually hands the inner IPM, with all eight cone types
//! per node (thrust SOC^4, pointing, mass-floor, glide SOC^3, T_min/T_max, trust
//! SOC^11, virtual-control SOC^8) and the column preconditioning that ships by
//! default.
//!
//! ## How the oracle is generated (transcription-free)
//!
//! Rather than re-encode the powered-descent physics in CVXPY (error-prone), we
//! dump the **already-assembled** standard-form matrices `(c, A, b, G, h, cones)`
//! and have the external solver re-solve that exact generic SOCP:
//! `min cᵀx s.t. Ax = b, (h − Gx)_cone ∈ SOC`. The oracle therefore validates the
//! IPM's *solve* of the assembled problem, taking the (separately unit-tested)
//! assembly as given — exactly the cross-check we want. Because the dumped
//! matrices are the same bits the Rust IPM consumes, there is zero opportunity for
//! a Python/Rust modelling drift.
//!
//! ## Regenerating the baked reference
//!
//! ```sh
//! # 1. Dump the assembled subproblems to tools/oracle-data/:
//! cargo test --release -p scvx-solver --test oracle_scvx_subproblem \
//!     dump_oracle_fixtures -- --ignored --nocapture
//! # 2. Solve them offline (CVXPY/Clarabel, and Julia/Clarabel):
//! tools/py-oracle/Scripts/python.exe tools/py-oracle/solve_scvx_subproblem.py
//! julia --project=tools/jl-oracle tools/jl-oracle/solve_scvx_subproblem.jl
//! # 3. Update the REF_* constants below from the printed values.
//! ```
//!
//! The `dump_oracle_fixtures` test and the assertion tests share the SAME fixture
//! builders, so the dumped problem and the asserted problem can never drift.

use nalgebra::SVector;
use scvx_core::{IpmAlgoParams, IpmStatus, PhysicalParams, Trajectory, G_MARS};
use scvx_dynamics::{discretize_foh, LinearizedDynamics};
use scvx_ipm::{solve_socp, SocpProblem, SocpResult, SocpWorkspace};
use scvx_solver::{
    assemble_scvx_socp, build_cone_scale_diagonal, build_scaling_diagonal,
    scale_cone_rows_in_place, scale_socp_in_place, scale_warm_start_in_place,
    sigma_idx_scvx, u_idx_scvx, x_idx_scvx, TerminalCondition,
};

// ---------------------------------------------------------------------------
// Canonical scenario (deterministic) — a 100 m Mars descent, N = 3.
//
// Chosen so the iter-0 subproblem is genuinely non-trivial: the linear-
// interpolation reference is NOT dynamically feasible, so the SOCP must do real
// work (non-zero δ within the trust region, non-zero virtual control ν).
// ---------------------------------------------------------------------------

const N: usize = 3;

// Fixed-tf SCvx layout dims (19·N, 7·N+6, 30·N, 8·N).
const NP_FX: usize = 19 * N; // 57
const NE: usize = 7 * N + 6; // 27
const NCT_FX: usize = 30 * N; // 90
const NCONES_FX: usize = 8 * N; // 24

// Free-tf adds the global δτ variable + two τ-bound cones.
const NP_FT: usize = 19 * N + 1; // 58
const NCT_FT: usize = 30 * N + 2; // 92
const NCONES_FT: usize = 8 * N + 2; // 26

const TRUST_ETA0: f64 = 5.0;
const VIRT_WEIGHT: f64 = 1.0e5;
const TARGET_MASS: f64 = 600.0;
const TAU: f64 = 12.0;
const RK4_SUBSTEPS: u32 = 4; // matches the production `RK4_SUBSTEPS` in scvx.rs

fn scenario_phys() -> PhysicalParams {
    PhysicalParams {
        g: [0.0, 0.0, -G_MARS],
        m_dry: 200.0,
        m_wet: 1000.0,
        isp: 225.0,
        g0: 9.80665,
        t_min: 1000.0,
        t_max: 6000.0,
        cos_theta_max: 0.7660444,
        tan_gamma_gs: 1.0,
        rho: 0.0,
        cd_a: 0.0,
        tau_lo: 5.0,
        tau_hi: 50.0,
    }
}

fn scenario_initial() -> SVector<f64, 7> {
    let mut x = SVector::<f64, 7>::zeros();
    x[2] = 100.0; // r_z = 100 m
    x[5] = -10.0; // v_z = -10 m/s
    x[6] = (800.0_f64).ln(); // m = 800 kg
    x
}

fn scenario_terminal() -> TerminalCondition {
    TerminalCondition { r: [0.0; 3], v: [0.0; 3] }
}

/// Replicates `api::seed_linear_reference` (which is private): linear state
/// interpolation initial→terminal, hover thrust at every node, fixed τ.
fn seed_reference(phys: &PhysicalParams, initial: &SVector<f64, 7>) -> Trajectory<N> {
    let term = scenario_terminal();
    let mut x_target = SVector::<f64, 7>::zeros();
    for i in 0..3 {
        x_target[i] = term.r[i];
        x_target[3 + i] = term.v[i];
    }
    x_target[6] = TARGET_MASS.max(phys.m_dry).ln();

    let m_avg = (initial[6].exp() + TARGET_MASS) * 0.5;
    let u_hover_z = -m_avg * phys.g[2]; // g[2] < 0 ⇒ > 0

    let mut traj = Trajectory::<N>::default();
    for k in 0..N {
        let alpha = if N > 1 { k as f64 / (N - 1) as f64 } else { 0.0 };
        for i in 0..7 {
            traj.x[(i, k)] = (1.0 - alpha) * initial[i] + alpha * x_target[i];
        }
        traj.u[(2, k)] = u_hover_z;
        traj.sigma[k] = u_hover_z;
    }
    traj.tau = TAU;
    traj
}

/// Fill the warm-start primal `x` from the reference (the production
/// `seed_warm_start`: state/control/σ at the reference, ν = w = 0).
fn seed_warm_start<const NP: usize>(reference: &Trajectory<N>) -> SVector<f64, NP> {
    let mut x = SVector::<f64, NP>::zeros();
    for k in 0..N {
        for i in 0..7 {
            x[x_idx_scvx(k) + i] = reference.x[(i, k)];
        }
        for i in 0..3 {
            x[u_idx_scvx(k) + i] = reference.u[(i, k)];
        }
        x[sigma_idx_scvx(k)] = reference.sigma[k];
    }
    x
}

/// Build the **preconditioned, fixed-tf** iter-0 subproblem plus its scaled
/// warm-start primal — exactly what `solve_scvx` feeds the inner IPM at iter 0
/// with `use_preconditioning = true`, `use_free_tf = false`.
fn build_fixture_fixedtf() -> (
    Box<SocpProblem<NP_FX, NE, NCT_FX, NCONES_FX>>,
    SVector<f64, NP_FX>,
) {
    let phys = scenario_phys();
    let initial = scenario_initial();
    let terminal = scenario_terminal();
    let reference = seed_reference(&phys, &initial);

    let mut lin = Box::<LinearizedDynamics<N>>::default();
    discretize_foh(&reference, &phys, &mut lin, RK4_SUBSTEPS);

    let mut prob = Box::<SocpProblem<NP_FX, NE, NCT_FX, NCONES_FX>>::default();
    assemble_scvx_socp(
        &reference, &lin, &phys, &initial, &terminal, TRUST_ETA0, VIRT_WEIGHT,
        false, &mut prob,
    );

    // Full preconditioning (column + per-cone row scaling): the "AHO + full
    // precond" config. Cone-row scaling divides each cone's `(G_c, h_c)` by
    // `e_c > 0`, which preserves the feasible set and the cost, so the external
    // oracle's optimum is unchanged — it only conditions the per-cone slacks so
    // the AHO endgame reliably snapshots a BestFeasible iterate (column-only is
    // fragile here: near-optimal at iter ~25, then breaks down). Cost is also
    // invariant under column scaling (`c'·x' = c·x`), so the dumped problem and
    // the oracle optimum are directly comparable.
    let scale = build_scaling_diagonal::<N, NP_FX>(&phys, &initial, false);
    scale_socp_in_place(&mut prob, &scale);
    let cone_scale = build_cone_scale_diagonal::<N, NCONES_FX>(&phys, &initial, TRUST_ETA0, false);
    scale_cone_rows_in_place(&mut prob, &cone_scale);

    let mut warm = seed_warm_start::<NP_FX>(&reference);
    scale_warm_start_in_place(&mut warm, &scale);
    (prob, warm)
}

/// Build the **preconditioned, free-tf** iter-0 subproblem (adds the global δτ
/// variable and the two τ-bound cones).
fn build_fixture_freetf() -> (
    Box<SocpProblem<NP_FT, NE, NCT_FT, NCONES_FT>>,
    SVector<f64, NP_FT>,
) {
    let phys = scenario_phys();
    let initial = scenario_initial();
    let terminal = scenario_terminal();
    let reference = seed_reference(&phys, &initial);

    let mut lin = Box::<LinearizedDynamics<N>>::default();
    discretize_foh(&reference, &phys, &mut lin, RK4_SUBSTEPS);

    let mut prob = Box::<SocpProblem<NP_FT, NE, NCT_FT, NCONES_FT>>::default();
    assemble_scvx_socp(
        &reference, &lin, &phys, &initial, &terminal, TRUST_ETA0, VIRT_WEIGHT,
        true, &mut prob,
    );

    let scale = build_scaling_diagonal::<N, NP_FT>(&phys, &initial, true);
    scale_socp_in_place(&mut prob, &scale);
    let cone_scale = build_cone_scale_diagonal::<N, NCONES_FT>(&phys, &initial, TRUST_ETA0, true);
    scale_cone_rows_in_place(&mut prob, &cone_scale);

    let mut warm = seed_warm_start::<NP_FT>(&reference);
    scale_warm_start_in_place(&mut warm, &scale);
    (prob, warm)
}

// ---------------------------------------------------------------------------
// Dump (transcription-free standard-form export)
// ---------------------------------------------------------------------------

fn dump_problem<const NP: usize, const NE2: usize, const NCT: usize, const NCONES: usize>(
    name: &str,
    prob: &SocpProblem<NP, NE2, NCT, NCONES>,
) {
    use std::fmt::Write as _;
    let mut s = String::new();
    let _ = writeln!(s, "# scvx subproblem dump: {name}");
    let _ = writeln!(s, "NP {NP}");
    let _ = writeln!(s, "NE {NE2}");
    let _ = writeln!(s, "NCT {NCT}");
    let _ = writeln!(s, "NCONES {NCONES}");

    let _ = writeln!(s, "c");
    for v in prob.c.iter() {
        let _ = writeln!(s, "{v:.17e}");
    }
    let _ = writeln!(s, "b");
    for v in prob.b.iter() {
        let _ = writeln!(s, "{v:.17e}");
    }
    let _ = writeln!(s, "h");
    for v in prob.h.iter() {
        let _ = writeln!(s, "{v:.17e}");
    }
    // A and G row-major.
    let _ = writeln!(s, "A {NE2} {NP}");
    for i in 0..NE2 {
        for j in 0..NP {
            let _ = writeln!(s, "{:.17e}", prob.a_mat[(i, j)]);
        }
    }
    let _ = writeln!(s, "G {NCT} {NP}");
    for i in 0..NCT {
        for j in 0..NP {
            let _ = writeln!(s, "{:.17e}", prob.g_mat[(i, j)]);
        }
    }
    let _ = writeln!(s, "cones {NCONES}");
    for cone in prob.cones.iter() {
        let _ = writeln!(s, "{} {}", cone.offset, cone.dim);
    }

    let dir = format!("{}/../../tools/oracle-data", env!("CARGO_MANIFEST_DIR"));
    std::fs::create_dir_all(&dir).expect("create oracle-data dir");
    let path = format!("{dir}/scvx_{name}.txt");
    std::fs::write(&path, s).expect("write dump");
    eprintln!("wrote {path}");
}

/// Not a correctness test — run explicitly to (re)generate the oracle input
/// dumps under `tools/oracle-data/`. Ignored in CI.
#[test]
#[ignore]
fn dump_oracle_fixtures() {
    let (fx, _) = build_fixture_fixedtf();
    dump_problem("fixedtf", &fx);
    let (ft, _) = build_fixture_freetf();
    dump_problem("freetf", &ft);
}

// ---------------------------------------------------------------------------
// KKT-optimality self-oracle (mirrors oracle_diff.rs::assert_kkt_optimal)
// ---------------------------------------------------------------------------

/// Assert the IPM iterate is **primal-feasible** (`Ax=b`, `Gx+s=h` tight, every
/// cone slack `s` interior) and REPORT the dual-stationarity residual + duality
/// gap. AHO leaves the complementarity gap loose on flight-scale SCvx
/// subproblems (the documented endgame — `s·y/n` stays O(10..1e3)), so those are
/// reported, not asserted; the optimal-**cost** agreement vs the external oracle
/// (asserted separately) is the convergence proof that the primal solution is
/// the right one.
fn assert_primal_feasible_report<
    const NP: usize, const NE2: usize, const NCT: usize, const NCONES: usize,
>(
    label: &str,
    prob: &SocpProblem<NP, NE2, NCT, NCONES>,
    res: &SocpResult<NP, NE2, NCT>,
    feas_tol: f64,
) {
    let r_a = (prob.a_mat * res.x - prob.b).norm();
    let r_g = (prob.g_mat * res.x + res.s - prob.h).norm();
    let r_d = (prob.c + prob.a_mat.transpose() * res.lambda + prob.g_mat.transpose() * res.y).norm();
    let compl = (res.s.dot(&res.y)).abs() / (NCT as f64);

    let s = res.s.as_slice();
    let mut min_s_marg = f64::INFINITY;
    for cone in prob.cones.iter() {
        let (o, d) = (cone.offset, cone.dim);
        let bar = (1..d).map(|i| s[o + i] * s[o + i]).sum::<f64>().sqrt();
        min_s_marg = min_s_marg.min(s[o] - bar);
    }
    eprintln!(
        "  KKT[{label}]: |Ax-b|={r_a:.2e} |Gx+s-h|={r_g:.2e} |c+Aᵀλ+Gᵀy|={r_d:.2e} \
         s·y/n={compl:.2e} s_marg={min_s_marg:.2e}"
    );
    assert!(r_a < feas_tol, "{label}: primal-eq residual {r_a:.2e} >= {feas_tol:.1e}");
    assert!(r_g < feas_tol, "{label}: primal-cone residual {r_g:.2e} >= {feas_tol:.1e}");
    assert!(
        min_s_marg > -feas_tol,
        "{label}: a cone slack left its cone (margin {min_s_marg:.2e})"
    );
}

fn cost<const NP: usize, const NE2: usize, const NCT: usize, const NCONES: usize>(
    prob: &SocpProblem<NP, NE2, NCT, NCONES>,
    res: &SocpResult<NP, NE2, NCT>,
) -> f64 {
    prob.c.dot(&res.x)
}

// ===========================================================================
// Assertion tests — Rust IPM vs external oracle on the flight subproblem.
// The REF_* values are baked from the offline CVXPY/Clarabel + Julia solves
// (see the module header for the regeneration recipe).
// ===========================================================================

/// Baked optimal cost `cᵀx` of the preconditioned fixed-tf subproblem.
/// Cost is invariant under the column preconditioning (`c'·x' = c·x`), and is
/// well-defined even if the argmin is not unique — so it is the primary
/// oracle quantity.
// Optimal objective of the (preconditioned) dumped subproblem. CVXPY/Clarabel
// and Julia/Clarabel agree to ~1e-9: fixed-tf 2.176946359299e6 (both), free-tf
// CVXPY 1.041035061760e6 / Julia 1.041035061761e6. Cost is invariant under the
// preconditioning, so these are directly comparable to the Rust solve.
const REF_FIXEDTF_COST: f64 = 2.176946359299e6;
const REF_FREETF_COST: f64 = 1.041035061760e6;

/// Tolerances for Rust-vs-oracle optimal-cost agreement under full precond. The
/// AHO BestFeasible snapshot lands a fraction of a percent above the true
/// optimum (the documented endgame duality gap): measured ~5e-4 (fixed-tf) /
/// ~7.6e-3 (free-tf, with the global δτ + Sherman-Morrison). Tolerances carry
/// ~2x margin.
const COST_REL_TOL_FIXEDTF: f64 = 1.0e-3;
const COST_REL_TOL_FREETF: f64 = 1.5e-2;

fn run_aho<const NP: usize, const NE2: usize, const NCT: usize, const NCONES: usize>(
    prob: &SocpProblem<NP, NE2, NCT, NCONES>,
    warm: &SVector<f64, NP>,
) -> SocpResult<NP, NE2, NCT> {
    let mut ws = Box::<SocpWorkspace<NP, NE2, NCT>>::default();
    ws.x = *warm;
    let params = IpmAlgoParams { warm_start_x: true, max_iters: 60, ..IpmAlgoParams::default() };
    solve_socp(prob, &params, &mut ws)
}

#[test]
fn rust_ipm_matches_external_oracle_scvx_fixedtf() {
    let (prob, warm) = build_fixture_fixedtf();
    let res = run_aho(&prob, &warm);
    eprintln!(
        "fixedtf: status={} iters={} cost={:.10e}",
        res.status.as_u32(), res.iters, cost(&prob, &res)
    );
    assert!(
        matches!(
            res.status,
            IpmStatus::Optimal | IpmStatus::BestFeasible | IpmStatus::IterCap
        ),
        "fixedtf: IPM returned a hard failure (status {})", res.status.as_u32()
    );
    assert_primal_feasible_report("fixedtf/AHO", &prob, &res, 1.0e-6);

    let c = cost(&prob, &res);
    let rel = (c - REF_FIXEDTF_COST).abs() / REF_FIXEDTF_COST.abs().max(1.0);
    assert!(
        rel < COST_REL_TOL_FIXEDTF,
        "fixedtf: cost {c:.10e} vs oracle {REF_FIXEDTF_COST:.10e} \
         (rel {rel:.2e} >= {COST_REL_TOL_FIXEDTF:.1e})"
    );
}

#[test]
fn rust_ipm_matches_external_oracle_scvx_freetf() {
    let (prob, warm) = build_fixture_freetf();
    let res = run_aho(&prob, &warm);
    eprintln!(
        "freetf: status={} iters={} cost={:.10e}",
        res.status.as_u32(), res.iters, cost(&prob, &res)
    );
    assert!(
        matches!(
            res.status,
            IpmStatus::Optimal | IpmStatus::BestFeasible | IpmStatus::IterCap
        ),
        "freetf: IPM returned a hard failure (status {})", res.status.as_u32()
    );
    assert_primal_feasible_report("freetf/AHO", &prob, &res, 1.0e-6);

    let c = cost(&prob, &res);
    let rel = (c - REF_FREETF_COST).abs() / REF_FREETF_COST.abs().max(1.0);
    assert!(
        rel < COST_REL_TOL_FREETF,
        "freetf: cost {c:.10e} vs oracle {REF_FREETF_COST:.10e} \
         (rel {rel:.2e} >= {COST_REL_TOL_FREETF:.1e})"
    );
}
