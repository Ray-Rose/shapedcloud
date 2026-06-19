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
use scvx_ipm::{solve_socp, solve_socp_hsd, solve_socp_nt, SocpProblem, SocpResult, SocpWorkspace};
use scvx_solver::{
    assemble_scvx_socp, build_cone_scale_diagonal, build_scaling_diagonal,
    scale_cone_rows_in_place, scale_socp_in_place, scale_warm_start_in_place,
    sigma_idx_scvx, solve_socp_structured_hsd, solve_socp_structured_hsd_free_tf,
    u_idx_scvx, x_idx_scvx, TerminalCondition,
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

fn scenario_initial(alt: f64) -> SVector<f64, 7> {
    let mut x = SVector::<f64, 7>::zeros();
    x[2] = alt; // r_z (m)
    x[5] = -0.1 * alt; // v_z (m/s) — descent rate proportional to altitude
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
fn build_fixture_fixedtf(
    alt: f64,
) -> (Box<SocpProblem<NP_FX, NE, NCT_FX, NCONES_FX>>, SVector<f64, NP_FX>) {
    let phys = scenario_phys();
    let initial = scenario_initial(alt);
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
fn build_fixture_freetf(
    alt: f64,
) -> (Box<SocpProblem<NP_FT, NE, NCT_FT, NCONES_FT>>, SVector<f64, NP_FT>) {
    let phys = scenario_phys();
    let initial = scenario_initial(alt);
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
    let (fx, _) = build_fixture_fixedtf(100.0);
    dump_problem("fixedtf", &fx);
    let (ft, _) = build_fixture_freetf(100.0);
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

fn run_nt<const NP: usize, const NE2: usize, const NCT: usize, const NCONES: usize>(
    prob: &SocpProblem<NP, NE2, NCT, NCONES>,
    warm: &SVector<f64, NP>,
) -> SocpResult<NP, NE2, NCT> {
    let mut ws = Box::<SocpWorkspace<NP, NE2, NCT>>::default();
    ws.x = *warm;
    let params = IpmAlgoParams { warm_start_x: true, max_iters: 60, ..IpmAlgoParams::default() };
    solve_socp_nt(prob, &params, &mut ws)
}

/// HSD ignores the warm-start (it cold-starts from the central point of the
/// self-dual embedding), so — unlike `run_aho`/`run_nt` — no `ws.x` seed.
fn run_hsd<const NP: usize, const NE2: usize, const NCT: usize, const NCONES: usize>(
    prob: &SocpProblem<NP, NE2, NCT, NCONES>,
) -> SocpResult<NP, NE2, NCT> {
    let mut ws = Box::<SocpWorkspace<NP, NE2, NCT>>::default();
    let params = IpmAlgoParams { max_iters: 60, ..IpmAlgoParams::default() };
    solve_socp_hsd(prob, &params, &mut ws)
}

/// Diagnostic (ignored): NT vs AHO across subproblem difficulty (altitude).
///
/// **Measured finding (the NT-frontier result).** With the exact closed-form
/// NT scaling (`soc_nt_scaling_exact`, numerically stable for vanishing cones —
/// it eliminates the geometric-mean `arrow(s)^{−1/2}` overflow), NT STILL
/// diverges at every altitude: primal infeasibility `|Ax-b|` *grows* from the
/// warm start to ~9-29 and the duality gap explodes (`s·y/n` ~ 1e12-1e14),
/// while AHO reaches BestFeasible. This independently confirms and refines the
/// Phase-15 conclusion: the per-cone scaling is NOT the bottleneck (the exact
/// canonical scaling does not help) — the barrier is the global NT Newton
/// direction / centering on the imbalanced vanishing-cone structure (the
/// virtual-control SOC^8 cones carry `μ_cone ~ 1e-4` while thrust/trust carry
/// ~1e7, so `H = GᵀW²G` is catastrophically ill-conditioned). A real fix needs
/// a wide-neighborhood / per-cone-balanced centering scheme (IPM research), not
/// a better scaling. AHO stays the robust reference direction and HSD (Phase 26)
/// is the production default that actually cracks this; NT stays opt-in.
#[test]
#[ignore]
fn diag_nt_on_flight_subproblem() {
    // status: 0=Optimal 1=BestFeasible 2=Infeasible 3=NumericalError 4=IterCap
    eprintln!("alt(m)|dir| st it |       cost       | |Ax-b|  |Gx+s-h| s·y/n");
    for &alt in &[2.0, 10.0, 50.0, 100.0] {
        let (p, w) = build_fixture_fixedtf(alt);
        for (dir, r) in [("AHO", run_aho(&p, &w)), ("NT ", run_nt(&p, &w)), ("HSD", run_hsd(&p))] {
            let r_a = (p.a_mat * r.x - p.b).norm();
            let r_g = (p.g_mat * r.x + r.s - p.h).norm();
            let compl = (r.s.dot(&r.y)).abs() / (NCT_FX as f64);
            eprintln!(
                "{alt:>5} |{dir}| {:>2} {:>2} | {:>15.6e} | {r_a:.1e} {r_g:.1e} {compl:.1e}",
                r.status.as_u32(), r.iters, cost(&p, &r)
            );
        }
    }
}

/// Diagnostic (ignored): HSD on the hardest subproblem (alt=100) across an
/// iteration budget, to characterize the endgame (does τ drift / does it stall?).
#[test]
#[ignore]
fn diag_hsd_iter_sweep_alt100() {
    let (p, _w) = build_fixture_fixedtf(100.0);
    eprintln!("HSD alt=100 iter sweep (oracle cost = {REF_FIXEDTF_COST:.6e}):");
    for &mi in &[10u32, 15, 20, 25, 28, 30, 32, 35, 40, 50, 64] {
        let mut ws = Box::<SocpWorkspace<NP_FX, NE, NCT_FX>>::default();
        let params = IpmAlgoParams { max_iters: mi, ..IpmAlgoParams::default() };
        let r = solve_socp_hsd(&p, &params, &mut ws);
        let r_a = (p.a_mat * r.x - p.b).norm();
        let r_g = (p.g_mat * r.x + r.s - p.h).norm();
        let compl = (r.s.dot(&r.y)).abs() / (NCT_FX as f64);
        let rel = (cost(&p, &r) - REF_FIXEDTF_COST).abs() / REF_FIXEDTF_COST.abs();
        eprintln!(
            "  mi={mi:>2} st={} it={:>2} cost={:>13.5e} relCost={rel:.2e} |Ax-b|={r_a:.2e} |Gx+s-h|={r_g:.2e} gap={compl:.2e}",
            r.status.as_u32(), r.iters, cost(&p, &r)
        );
    }
}

#[test]
fn rust_ipm_matches_external_oracle_scvx_fixedtf() {
    let (prob, warm) = build_fixture_fixedtf(100.0);
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
    let (prob, warm) = build_fixture_freetf(100.0);
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

// ===========================================================================
// HSD-direction external-oracle gates (the NT/O(N) frontier crack).
//
// These are the gates the plain-NT direction could NEVER pass: on the SAME
// flight-scale subproblem, `solve_socp_nt` DIVERGES (primal infeasibility
// `|Ax-b|` grows to ~9-29, duality gap `s·y/n` explodes to ~1e12-1e14 — see
// `diag_nt_on_flight_subproblem`). The homogeneous self-dual embedding
// (`solve_socp_hsd`) instead converges to the external CVXPY/Clarabel + Julia
// optimum, MORE tightly and faster than even the AHO reference direction
// (measured fixed-tf: HSD rel-cost ~8.5e-8 in ~15 iters vs AHO ~5e-4 in 60).
// HSD cold-starts from the self-dual central point (no warm-start).
// ===========================================================================

/// Tolerances for the HSD gates. The cost tolerance is ~100x tighter than the
/// AHO gate's (HSD measured ~8.5e-8 fixed-tf); the gap bound is the load-bearing
/// assertion — it is what plain NT cannot satisfy (NT diverges to ~1e13 here).
const HSD_COST_REL_TOL_FIXEDTF: f64 = 1.0e-5;
const HSD_COST_REL_TOL_FREETF:  f64 = 1.0e-2;
// "Did not diverge" guard. Plain NT explodes to `s·y/n ~ 1e13` on this
// subproblem; HSD stays bounded (measured fixed-tf 5.4e-4, free-tf 1.06). A
// bound of 10 still proves the crack by ~12 orders of magnitude while giving the
// free-tf endgame (extra δτ variable + 2 τ-bound cones) headroom.
const HSD_GAP_BOUND:            f64 = 10.0;

#[test]
fn rust_hsd_matches_external_oracle_scvx_fixedtf() {
    let (prob, _warm) = build_fixture_fixedtf(100.0);
    let res = run_hsd(&prob);
    let gap = res.s.dot(&res.y).abs() / (NCT_FX as f64);
    eprintln!(
        "HSD fixedtf: status={} iters={} cost={:.10e} gap={gap:.2e}",
        res.status.as_u32(), res.iters, cost(&prob, &res)
    );
    assert!(
        matches!(res.status, IpmStatus::Optimal | IpmStatus::BestFeasible),
        "HSD fixedtf: did not converge (status {})", res.status.as_u32()
    );
    assert_primal_feasible_report("fixedtf/HSD", &prob, &res, 1.0e-3);

    // THE crack: HSD stays near-complementary (bounded gap) where plain NT
    // diverges to `s·y/n ~ 1e13`. This is the assertion NT structurally fails.
    assert!(
        gap < HSD_GAP_BOUND,
        "HSD fixedtf: duality gap {gap:.2e} exploded (plain NT hits ~1e13 here)"
    );

    // And it nails the external oracle far tighter than AHO.
    let c = cost(&prob, &res);
    let rel = (c - REF_FIXEDTF_COST).abs() / REF_FIXEDTF_COST.abs().max(1.0);
    assert!(
        rel < HSD_COST_REL_TOL_FIXEDTF,
        "HSD fixedtf: cost {c:.10e} vs oracle {REF_FIXEDTF_COST:.10e} \
         (rel {rel:.2e} >= {HSD_COST_REL_TOL_FIXEDTF:.1e})"
    );
}

/// **O(N) structured HSD** (Phase 28) — the block-tridiagonal-Schur twin of the
/// dense HSD. This is the gate that proves NT and O(N) finally close TOGETHER:
/// the SAME structured factorization that diverges under plain NT (the
/// vanishing-cone `W²` blow-up) is numerically SOUND under HSD (the self-dual
/// embedding keeps `W` bounded). It must (a) reach the external oracle — a Schur
/// bug would diverge — and (b) match the dense HSD (the drop-in claim).
#[test]
fn rust_structured_hsd_matches_external_oracle_scvx_fixedtf() {
    let (prob, _warm) = build_fixture_fixedtf(100.0);

    let mut ws = Box::<SocpWorkspace<NP_FX, NE, NCT_FX>>::default();
    let params = IpmAlgoParams { max_iters: 60, ..IpmAlgoParams::default() };
    let res = solve_socp_structured_hsd::<N, NP_FX, NE, NCT_FX, NCONES_FX>(&prob, &params, &mut ws);

    let gap = res.s.dot(&res.y).abs() / (NCT_FX as f64);
    eprintln!(
        "structured HSD fixedtf: status={} iters={} cost={:.10e} gap={gap:.2e}",
        res.status.as_u32(), res.iters, cost(&prob, &res)
    );
    assert!(
        matches!(res.status, IpmStatus::Optimal | IpmStatus::BestFeasible),
        "structured HSD fixedtf: did not converge (status {})", res.status.as_u32()
    );
    assert_primal_feasible_report("fixedtf/structured-HSD", &prob, &res, 1.0e-3);
    assert!(gap < HSD_GAP_BOUND, "structured HSD fixedtf: gap {gap:.2e} exploded");

    // (a) Reaches the external CVXPY/Clarabel + Julia oracle (correctness).
    let c = cost(&prob, &res);
    let rel = (c - REF_FIXEDTF_COST).abs() / REF_FIXEDTF_COST.abs().max(1.0);
    assert!(
        rel < HSD_COST_REL_TOL_FIXEDTF,
        "structured HSD fixedtf: cost {c:.10e} vs oracle {REF_FIXEDTF_COST:.10e} (rel {rel:.2e})"
    );

    // (b) Matches the DENSE HSD (the structured Schur is a drop-in for the dense
    // H⁻¹/S⁻¹ — both cold-start central and take the same Newton step each iter).
    let dense = run_hsd(&prob);
    let c_dense = cost(&prob, &dense);
    let cost_rel = (c - c_dense).abs() / c_dense.abs().max(1.0);
    let dx_rel = (res.x - dense.x).norm() / dense.x.norm().max(1.0);
    eprintln!("  structured-vs-dense HSD: cost rel={cost_rel:.2e}  |Δx|/|x|={dx_rel:.2e}");
    assert!(
        cost_rel < 1.0e-4,
        "structured HSD cost {c:.10e} diverged from dense {c_dense:.10e} (rel {cost_rel:.2e})"
    );
}

/// **O(N) structured free-tf HSD** (Phase 29) — completes the structured HSD
/// matrix. Composes the HSD τ-coupling with the free-tf Sherman-Morrison δτ
/// machinery, so the O(N) block-tridiagonal Schur handles the global δτ column.
/// Must reach the external oracle AND match the dense HSD on the free-tf
/// subproblem.
#[test]
fn rust_structured_hsd_freetf_matches_external_oracle() {
    let (prob, _warm) = build_fixture_freetf(100.0);

    let mut ws = Box::<SocpWorkspace<NP_FT, NE, NCT_FT>>::default();
    let params = IpmAlgoParams { max_iters: 60, ..IpmAlgoParams::default() };
    let res = solve_socp_structured_hsd_free_tf::<N, NP_FT, NE, NCT_FT, NCONES_FT>(&prob, &params, &mut ws);

    let gap = res.s.dot(&res.y).abs() / (NCT_FT as f64);
    eprintln!(
        "structured HSD freetf: status={} iters={} cost={:.10e} gap={gap:.2e}",
        res.status.as_u32(), res.iters, cost(&prob, &res)
    );
    assert!(
        matches!(res.status, IpmStatus::Optimal | IpmStatus::BestFeasible),
        "structured HSD freetf: did not converge (status {})", res.status.as_u32()
    );
    assert_primal_feasible_report("freetf/structured-HSD", &prob, &res, 1.0e-3);
    assert!(gap < HSD_GAP_BOUND, "structured HSD freetf: gap {gap:.2e} exploded");

    let c = cost(&prob, &res);
    let rel = (c - REF_FREETF_COST).abs() / REF_FREETF_COST.abs().max(1.0);
    assert!(
        rel < HSD_COST_REL_TOL_FREETF,
        "structured HSD freetf: cost {c:.10e} vs oracle {REF_FREETF_COST:.10e} (rel {rel:.2e})"
    );

    // Matches the dense free-tf HSD (the SMW-on-HSD composition is a drop-in).
    let dense = run_hsd(&prob);
    let c_dense = cost(&prob, &dense);
    let cost_rel = (c - c_dense).abs() / c_dense.abs().max(1.0);
    let dx_rel = (res.x - dense.x).norm() / dense.x.norm().max(1.0);
    eprintln!("  structured-vs-dense free-tf HSD: cost rel={cost_rel:.2e}  |Δx|/|x|={dx_rel:.2e}");
    assert!(
        cost_rel < 1.0e-3,
        "structured free-tf HSD cost {c:.10e} diverged from dense {c_dense:.10e} (rel {cost_rel:.2e})"
    );
}

#[test]
fn rust_hsd_matches_external_oracle_scvx_freetf() {
    let (prob, _warm) = build_fixture_freetf(100.0);
    let res = run_hsd(&prob);
    let gap = res.s.dot(&res.y).abs() / (NCT_FT as f64);
    eprintln!(
        "HSD freetf: status={} iters={} cost={:.10e} gap={gap:.2e}",
        res.status.as_u32(), res.iters, cost(&prob, &res)
    );
    assert!(
        matches!(res.status, IpmStatus::Optimal | IpmStatus::BestFeasible),
        "HSD freetf: did not converge (status {})", res.status.as_u32()
    );
    assert_primal_feasible_report("freetf/HSD", &prob, &res, 1.0e-3);
    assert!(
        gap < HSD_GAP_BOUND,
        "HSD freetf: duality gap {gap:.2e} exploded (plain NT diverges here)"
    );
    let c = cost(&prob, &res);
    let rel = (c - REF_FREETF_COST).abs() / REF_FREETF_COST.abs().max(1.0);
    assert!(
        rel < HSD_COST_REL_TOL_FREETF,
        "HSD freetf: cost {c:.10e} vs oracle {REF_FREETF_COST:.10e} \
         (rel {rel:.2e} >= {HSD_COST_REL_TOL_FREETF:.1e})"
    );
}
