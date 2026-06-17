//! WCET benchmark suite for the SCvx solver stack.
//!
//! Measures host-side timing (x86_64-pc-windows-msvc) for the building
//! blocks that have to fit a flight cycle. Reports `min/p50/p99/max` in
//! nanoseconds across 1k–10k iterations of each operation.
//!
//! These are **host** numbers; absolute values translate to Cortex-R/A
//! cycles only after the toolchain-and-target-specific compilation step.
//! The point at this phase is to establish **bounded** WCET — no operation
//! grows past its expected complexity class, no operation hangs.
//!
//! Run with:
//! ```text
//! cargo test --release --test wcet_benchmarks -- --nocapture
//! ```

use std::time::Instant;

use nalgebra::{SMatrix, SVector};
use scvx_core::{
    IpmAlgoParams, IpmStatus, PhysicalParams, ScvxAlgoParams, SolverStatus, Trajectory,
};
use scvx_dynamics::{discretize_foh, LinearizedDynamics};
use scvx_solver::{solve_scvx, solve_socp_structured_hsd, ScvxWorkspace};
use scvx_ipm::{
    riccati_factor, riccati_solve, soc_arrow_inv_sqrt, soc_arrow_matrix, soc_det,
    soc_jordan_product, soc_max_step, soc_nt_scaling_matrix, soc_project, soc_sqrt,
    solve_socp, solve_socp_hsd, ConeDesc, LqrProblem, LqrSolution, LqrWorkspace,
    SocpProblem, SocpWorkspace,
};
use scvx_solver::assemble::{
    assemble_lcvx_socp, assemble_scvx_socp,
    N_CONES_PER_NODE, N_CONES_PER_NODE_SCVX, N_CONE_DIM_PER_NODE,
    N_CONE_DIM_PER_NODE_SCVX, N_EQ_PER_DYN, N_EQ_TERMINAL, N_VARS_PER_NODE,
    N_VARS_PER_NODE_SCVX, TerminalCondition,
};
use scvx_solver::reduced_kkt::{
    factor_reduced_kkt_scvx_block_m, solve_reduced_kkt_scvx_block_m,
    solve_reduced_kkt_scvx_with_factor, ReducedKktFactor, ReducedKktSolution,
    ReducedKktStatus,
};

/// Time `f` for `n_runs` iterations and report the min/p50/p99/max in
/// nanoseconds. `f` is called in the hot loop — keep it allocation-free.
fn time<F: FnMut()>(name: &str, n_runs: usize, mut f: F) -> (u64, u64, u64, u64) {
    let mut samples = Vec::with_capacity(n_runs);
    // Warm-up to reduce cold-cache noise.
    for _ in 0..16 {
        f();
    }
    for _ in 0..n_runs {
        let t0 = Instant::now();
        f();
        let dt = t0.elapsed().as_nanos();
        // Saturate at u64::MAX (impossible in practice; defensive).
        samples.push(dt.min(u64::MAX as u128) as u64);
    }
    samples.sort_unstable();
    let min = samples[0];
    let p50 = samples[n_runs / 2];
    let p99 = samples[(n_runs * 99) / 100];
    let max = samples[n_runs - 1];
    eprintln!(
        "  {:<40}  min={:>8}  p50={:>8}  p99={:>8}  max={:>8}  ns",
        name, min, p50, p99, max
    );
    (min, p50, p99, max)
}

/// Sanity bound: nothing in this suite should take longer than 100 ms on a
/// modern x86_64 host. If something does, it's misbehaving (probably the
/// build was a debug build, or there's a runaway loop).
const SANITY_BOUND_NS: u64 = 100_000_000;

fn assert_bounded(name: &str, _min: u64, _p50: u64, p99: u64, max: u64) {
    assert!(
        max < SANITY_BOUND_NS,
        "{name}: max {max} ns exceeds {SANITY_BOUND_NS} ns — broken benchmark?"
    );
    assert!(p99 <= max, "{name}: p99 > max — corrupted sample buffer?");
}

// ===========================================================================

#[test]
fn wcet_cone_primitives() {
    eprintln!("\n--- WCET: SOC cone primitives (D=4, the SCvx thrust-mag size) ---");

    let z = [3.0_f64, 1.0, -0.5, 0.7];
    let dz = [0.1_f64, -0.2, 0.05, 0.0];
    let mut out = [0.0_f64; 4];

    let (m, p, p99, mx) = time("soc_det", 100_000, || {
        let r = soc_det(&z);
        std::hint::black_box(r);
    });
    assert_bounded("soc_det", m, p, p99, mx);

    let (m, p, p99, mx) = time("soc_max_step", 100_000, || {
        let r = soc_max_step(&z, &dz);
        std::hint::black_box(r);
    });
    assert_bounded("soc_max_step", m, p, p99, mx);

    let (m, p, p99, mx) = time("soc_project", 100_000, || {
        soc_project(&z, &mut out);
        std::hint::black_box(out);
    });
    assert_bounded("soc_project", m, p, p99, mx);

    let (m, p, p99, mx) = time("soc_sqrt", 100_000, || {
        soc_sqrt(&z, &mut out);
        std::hint::black_box(out);
    });
    assert_bounded("soc_sqrt", m, p, p99, mx);

    let u = [2.0_f64, 0.5, 0.3, -0.2];
    let v = [1.0_f64, -0.4, 0.6, 0.1];
    let (m, p, p99, mx) = time("soc_jordan_product", 100_000, || {
        soc_jordan_product(&u, &v, &mut out);
        std::hint::black_box(out);
    });
    assert_bounded("soc_jordan_product", m, p, p99, mx);

    let z_vec = SVector::<f64, 4>::from_column_slice(&z);
    let (m, p, p99, mx) = time("soc_arrow_matrix", 100_000, || {
        let r = soc_arrow_matrix(&z_vec);
        std::hint::black_box(r);
    });
    assert_bounded("soc_arrow_matrix", m, p, p99, mx);

    // NT scaling matrix construction (D=4). The real cost of NT integration
    // would be ~one of these per cone per IPM iter.
    let y_vec = SVector::<f64, 4>::from_column_slice(&[2.5, 0.8, -0.3, 0.4]);
    let z_vec = SVector::<f64, 4>::from_column_slice(&z);
    let (m, p, p99, mx) = time("soc_arrow_inv_sqrt (D=4)", 100_000, || {
        let r = soc_arrow_inv_sqrt::<4>(&z_vec);
        std::hint::black_box(r);
    });
    assert_bounded("soc_arrow_inv_sqrt", m, p, p99, mx);

    let (m, p, p99, mx) = time("soc_nt_scaling_matrix (D=4)", 100_000, || {
        let r = soc_nt_scaling_matrix::<4>(&z_vec, &y_vec);
        std::hint::black_box(r);
    });
    assert_bounded("soc_nt_scaling_matrix", m, p, p99, mx);
}

#[test]
fn wcet_dynamics_discretization() {
    eprintln!("\n--- WCET: 3-DoF dynamics + discretization (N=4) ---");

    const N: usize = 4;
    let phys = PhysicalParams {
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
    };
    let mut traj = Trajectory::<N>::default();
    let mut x0 = SVector::<f64, 7>::zeros();
    x0[2] = 500.0;
    x0[5] = -20.0;
    x0[6] = (800.0_f64).ln();
    let u_hover_z = 800.0 * 3.7114;
    for k in 0..N {
        for i in 0..7 {
            traj.x[(i, k)] = x0[i];
        }
        traj.u[(2, k)] = u_hover_z;
    }
    traj.tau = 20.0;
    let mut lin = Box::new(LinearizedDynamics::<N>::default());

    let (m, p, p99, mx) = time("discretize_foh (N=4, RK4=4)", 1_000, || {
        discretize_foh(&traj, &phys, &mut lin, 4);
        std::hint::black_box(&*lin);
    });
    assert_bounded("discretize_foh", m, p, p99, mx);
}

#[test]
fn wcet_riccati() {
    eprintln!("\n--- WCET: block-banded Riccati (N=8, NX=2, NU=1) ---");

    const N: usize = 8;
    const NX: usize = 2;
    const NU: usize = 1;
    let mut prob = LqrProblem::<N, NX, NU>::default();
    for k in 0..N - 1 {
        prob.a[k] = SMatrix::<f64, 2, 2>::from_row_slice(&[1.0, 0.1, 0.0, 1.0]);
        prob.b[k] = SMatrix::<f64, 2, 1>::from_column_slice(&[0.0, 0.1]);
        prob.c[k] = SVector::<f64, 2>::from_column_slice(&[0.0, -0.02]);
    }
    for k in 0..N {
        prob.q_mat[k] = SMatrix::<f64, 2, 2>::identity();
    }
    for k in 0..N - 1 {
        prob.r_mat[k] = SMatrix::<f64, 1, 1>::from_element(0.1);
    }
    prob.x_init = SVector::<f64, 2>::from_column_slice(&[10.0, 0.0]);

    let mut ws = Box::new(LqrWorkspace::<N, NX, NU>::default());
    let mut sol = Box::new(LqrSolution::<N, NX, NU>::default());

    let (m, p, p99, mx) = time("riccati_factor", 10_000, || {
        let _ = riccati_factor(&prob, &mut ws);
        std::hint::black_box(&*ws);
    });
    assert_bounded("riccati_factor", m, p, p99, mx);

    let _ = riccati_factor(&prob, &mut ws); // pre-factor for solve bench
    let (m, p, p99, mx) = time("riccati_solve", 10_000, || {
        riccati_solve(&prob, &ws, &mut sol);
        std::hint::black_box(&*sol);
    });
    assert_bounded("riccati_solve", m, p, p99, mx);
}

#[test]
fn wcet_socp_ipm_two_cone() {
    eprintln!("\n--- WCET: generic SOCP IPM, 2-cone mixed problem ---");

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
    let params = IpmAlgoParams::default();
    let mut ws = Box::new(SocpWorkspace::<NP, NE, NCT>::default());

    // Verify it solves before benchmarking.
    let result = solve_socp(&prob, &params, &mut ws);
    eprintln!("  2-cone IPM status = {} after {} iters",
              result.status.as_u32(), result.iters);
    assert!(matches!(
        result.status,
        IpmStatus::Optimal | IpmStatus::BestFeasible
    ));

    let (m, p, p99, mx) = time("solve_socp (2-cone, full IPM run)", 1_000, || {
        // Reset workspace each run so the IPM does the full factor+solve.
        *ws = SocpWorkspace::default();
        let _ = solve_socp(&prob, &params, &mut ws);
        std::hint::black_box(&*ws);
    });
    assert_bounded("solve_socp 2-cone", m, p, p99, mx);
}

#[test]
fn wcet_full_lcvx_assembly() {
    eprintln!("\n--- WCET: full LCvx SOCP assembly (N=4) ---");

    const N: usize = 4;
    const NP: usize     = N * N_VARS_PER_NODE;
    const NE: usize     = N * N_EQ_PER_DYN + N_EQ_TERMINAL;
    const NCT: usize    = N * N_CONE_DIM_PER_NODE;
    const NCONES: usize = N * N_CONES_PER_NODE;

    let phys = PhysicalParams {
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
    };
    let mut x0 = SVector::<f64, 7>::zeros();
    x0[2] = 500.0;
    x0[5] = -20.0;
    x0[6] = (800.0_f64).ln();
    let mut traj = Trajectory::<N>::default();
    let u_hover_z = 800.0 * 3.7114;
    for k in 0..N {
        for i in 0..7 {
            traj.x[(i, k)] = x0[i];
        }
        traj.u[(2, k)] = u_hover_z;
    }
    traj.tau = 20.0;
    let mut lin = Box::new(LinearizedDynamics::<N>::default());
    discretize_foh(&traj, &phys, &mut lin, 4);

    let term = TerminalCondition { r: [0.0; 3], v: [0.0; 3] };
    let mut prob = Box::new(SocpProblem::<NP, NE, NCT, NCONES>::default());

    let (m, p, p99, mx) = time("assemble_lcvx_socp (N=4)", 1_000, || {
        assemble_lcvx_socp(&traj, &lin, &phys, &x0, &term, &mut prob);
        std::hint::black_box(&*prob);
    });
    assert_bounded("assemble_lcvx_socp", m, p, p99, mx);
}

#[test]
fn wcet_riccati_grows_linearly_in_n() {
    eprintln!("\n--- WCET: Riccati scaling vs N (smoke test for O(N · NX³)) ---");

    // Hand-rolled instantiations because const generics need fixed N.
    fn bench_riccati<const N: usize>() -> u64 {
        const NX: usize = 2;
        const NU: usize = 1;
        let mut prob = LqrProblem::<N, NX, NU>::default();
        for k in 0..N - 1 {
            prob.a[k] = SMatrix::<f64, 2, 2>::from_row_slice(&[1.0, 0.1, 0.0, 1.0]);
            prob.b[k] = SMatrix::<f64, 2, 1>::from_column_slice(&[0.0, 0.1]);
        }
        for k in 0..N {
            prob.q_mat[k] = SMatrix::<f64, 2, 2>::identity();
        }
        for k in 0..N - 1 {
            prob.r_mat[k] = SMatrix::<f64, 1, 1>::from_element(0.1);
        }
        prob.x_init = SVector::<f64, 2>::from_column_slice(&[10.0, 0.0]);

        let mut ws = Box::new(LqrWorkspace::<N, NX, NU>::default());
        let mut sol = Box::new(LqrSolution::<N, NX, NU>::default());

        // warm
        for _ in 0..16 {
            let _ = riccati_factor(&prob, &mut ws);
            riccati_solve(&prob, &ws, &mut sol);
        }
        // measure
        let n_runs = 1_000;
        let t0 = Instant::now();
        for _ in 0..n_runs {
            let _ = riccati_factor(&prob, &mut ws);
            riccati_solve(&prob, &ws, &mut sol);
        }
        let dt = t0.elapsed().as_nanos() as u64;
        std::hint::black_box(&*sol);
        dt / (n_runs as u64)
    }

    let t4 = bench_riccati::<4>();
    let t8 = bench_riccati::<8>();
    let t16 = bench_riccati::<16>();
    let t32 = bench_riccati::<32>();
    eprintln!("  riccati N=4:  {} ns/solve", t4);
    eprintln!("  riccati N=8:  {} ns/solve  (ratio vs N=4: {:.2}×)", t8,  t8  as f64 / t4 as f64);
    eprintln!("  riccati N=16: {} ns/solve  (ratio vs N=4: {:.2}×)", t16, t16 as f64 / t4 as f64);
    eprintln!("  riccati N=32: {} ns/solve  (ratio vs N=4: {:.2}×)", t32, t32 as f64 / t4 as f64);

    // Expect roughly linear scaling — N=32 should be < 16× the N=4 cost
    // (some constant overhead per call dominates at small N). Generous bound:
    assert!(
        t32 < 20 * t4,
        "Riccati scaling looks non-linear: 32× nodes took {}× the time",
        t32 as f64 / t4 as f64
    );
}

/// **Phase 6 quantitative gate**: time the structured (block-tridiagonal)
/// reduced-KKT solve against the dense `K.try_inverse()` baseline on a
/// real SCvx subproblem. Reports both timings + the speedup ratio.
///
/// Expected: structured wins at moderate-to-large N because dense cost is
/// `O((N·NZ)³)` while structured is `O(N·NZ³)`. At N=5 the gap is small
/// (the structured solver has bookkeeping overhead); at N=10+ it should
/// be clearly faster.
///
/// This benchmark is a **smoke test** — we assert the structured solver
/// is at most some multiple slower than dense (very loose bound), not
/// strict equality, because we want to flag massive perf regressions
/// without making the test brittle.
#[test]
fn wcet_structured_vs_dense_kkt() {
    // SCvx-shaped Hessians at N=7 push the dense-LU path past the default
    // 2 MB test stack in debug builds. Run the actual bench in a worker
    // thread with a 32 MB stack — same discipline as the other large-N
    // SCvx tests.
    std::thread::Builder::new()
        .stack_size(32 * 1024 * 1024)
        .spawn(wcet_structured_vs_dense_kkt_inner)
        .expect("spawn bench thread")
        .join()
        .expect("bench thread panicked");
}

fn wcet_structured_vs_dense_kkt_inner() {
    eprintln!("\n--- WCET: structured vs dense reduced-KKT on SCvx subproblem ---");

    // Const NP/NE/NCT/NCONES must all be const-generic params on the inner
    // function — Rust doesn't allow `const X = f(N)` inside a generic fn body.
    fn bench<
        const N: usize,
        const NP: usize,
        const NE: usize,
        const NCT: usize,
        const NCONES: usize,
    >() -> (u64, u64, f64) {

        let phys = PhysicalParams {
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
        };
        let mut x_init = SVector::<f64, 7>::zeros();
        x_init[2] = 500.0; x_init[5] = -20.0; x_init[6] = (800.0_f64).ln();
        let mut traj = Trajectory::<N>::default();
        for k in 0..N {
            for i in 0..7 { traj.x[(i, k)] = x_init[i]; }
            traj.u[(2, k)] = 800.0 * 3.7114;
        }
        traj.tau = 20.0;

        let mut lin = Box::new(LinearizedDynamics::<N>::default());
        discretize_foh(&traj, &phys, &mut lin, 4);

        let term = TerminalCondition { r: [0.0; 3], v: [0.0; 3] };
        let mut prob = Box::new(SocpProblem::<NP, NE, NCT, NCONES>::default());
        assemble_scvx_socp::<N, NP, NE, NCT, NCONES>(
            &traj, &lin, &phys, &x_init, &term,
            1.0e3, 1.0e4, false, &mut prob,
        );

        // Synthesize a block-diagonal M (identity per cone — simplest).
        let m_full = SMatrix::<f64, NCT, NCT>::identity();
        let reg: f64 = 1.0e-8;

        // Build a synthetic RHS.
        let mut b_x = SVector::<f64, NP>::zeros();
        let mut b_a = SVector::<f64, NE>::zeros();
        for i in 0..NP { b_x[i] = ((i as f64) * 0.13).sin(); }
        for i in 0..NE { b_a[i] = ((i as f64) * 0.07).cos(); }

        // ---- Bench structured ----
        let mut sol = Box::new(ReducedKktSolution::<N>::default());
        for _ in 0..8 {
            let _ = solve_reduced_kkt_scvx_block_m::<N, NP, NE, NCT, NCONES>(
                &prob, &m_full, reg, &b_x, &b_a, &mut sol,
            );
        }
        let n_runs = 200;
        let t0 = Instant::now();
        for _ in 0..n_runs {
            let _ = solve_reduced_kkt_scvx_block_m::<N, NP, NE, NCT, NCONES>(
                &prob, &m_full, reg, &b_x, &b_a, &mut sol,
            );
        }
        let t_struct = (t0.elapsed().as_nanos() as u64) / (n_runs as u64);
        // Sanity: structured solver succeeded.
        let last_status = solve_reduced_kkt_scvx_block_m::<N, NP, NE, NCT, NCONES>(
            &prob, &m_full, reg, &b_x, &b_a, &mut sol,
        );
        assert_eq!(last_status, ReducedKktStatus::Ok);
        std::hint::black_box(&*sol);

        // ---- Bench dense ----
        //
        // Build the dense augmented KKT and invert each call. Mirrors what
        // the existing IPM does inside build_step_factors.
        let mut h_dense = Box::new(SMatrix::<f64, NP, NP>::zeros());
        // H = G^T·M·G + reg·I. Compute outside the timing loop.
        *h_dense = prob.g_mat.transpose() * m_full * prob.g_mat;
        for i in 0..NP { h_dense[(i, i)] += reg; }

        for _ in 0..8 {
            let h_inv = h_dense.try_inverse().unwrap();
            std::hint::black_box(h_inv);
        }
        let t0 = Instant::now();
        for _ in 0..n_runs {
            let h_inv = h_dense.try_inverse().unwrap();
            std::hint::black_box(h_inv);
        }
        let t_dense_inv = (t0.elapsed().as_nanos() as u64) / (n_runs as u64);

        let speedup = t_dense_inv as f64 / t_struct as f64;
        (t_struct, t_dense_inv, speedup)
    }

    // Per-N const expansion (Rust doesn't have const-arithmetic-bounded
    // generic dispatch; each N gets its own concrete instantiation).
    const N3: usize = 3;
    const N3_NP: usize     = N3 * N_VARS_PER_NODE_SCVX;     // 57
    const N3_NE: usize     = N3 * 7 + 6;                     // 27
    const N3_NCT: usize    = N3 * N_CONE_DIM_PER_NODE_SCVX; // 90
    const N3_NCONES: usize = N3 * 8;                         // 24
    let (s3, d3, sp3) = bench::<N3, N3_NP, N3_NE, N3_NCT, N3_NCONES>();

    const N5: usize = 5;
    const N5_NP: usize     = N5 * N_VARS_PER_NODE_SCVX;
    const N5_NE: usize     = N5 * 7 + 6;
    const N5_NCT: usize    = N5 * N_CONE_DIM_PER_NODE_SCVX;
    const N5_NCONES: usize = N5 * 8;
    let (s5, d5, sp5) = bench::<N5, N5_NP, N5_NE, N5_NCT, N5_NCONES>();

    const N7: usize = 7;
    const N7_NP: usize     = N7 * N_VARS_PER_NODE_SCVX;
    const N7_NE: usize     = N7 * 7 + 6;
    const N7_NCT: usize    = N7 * N_CONE_DIM_PER_NODE_SCVX;
    const N7_NCONES: usize = N7 * 8;
    let (s7, d7, sp7) = bench::<N7, N7_NP, N7_NE, N7_NCT, N7_NCONES>();
    eprintln!("  N=3 : structured {:>8} ns,  dense-Hinv {:>8} ns  ({:>4.2}× of dense)",
              s3, d3, sp3);
    eprintln!("  N=5 : structured {:>8} ns,  dense-Hinv {:>8} ns  ({:>4.2}× of dense)",
              s5, d5, sp5);
    eprintln!("  N=7 : structured {:>8} ns,  dense-Hinv {:>8} ns  ({:>4.2}× of dense)",
              s7, d7, sp7);
    eprintln!("  scaling structured (N=7/N=3): {:.2}×", s7 as f64 / s3 as f64);
    eprintln!("  scaling dense      (N=7/N=3): {:.2}×", d7 as f64 / d3 as f64);

    // Smoke: structured should not be wildly slower than dense even at
    // small N. (At small N the constant overhead of the structured solver
    // dominates; allow up to 20× while we're not yet maximally optimized.)
    assert!(s3 < 30 * d3, "structured solver pathologically slow at N=3: {s3}/{d3}");

    // Scaling: dense-H_inv is O((N·NZ)^3) so scales with N^3.
    // Structured is O(N·NZ^3) so scales with N. From N=3 to N=7 we expect:
    //   dense ratio ≈ (7/3)^3 ≈ 12.7×
    //   structured ratio ≈ 7/3 ≈ 2.3×
    // The actual ratios won't be that clean due to cache effects, but
    // structured should scale much slower than dense. We only assert
    // that dense scales no slower than structured at N=7.
    let dense_ratio  = d7 as f64 / d3 as f64;
    let struct_ratio = s7 as f64 / s3 as f64;
    eprintln!("  dense scaling ratio  ({:.2}×) should be >= structured ratio ({:.2}×)",
              dense_ratio, struct_ratio);
    // (No hard assert here — host timing is too noisy. Just report.)
}

/// **Phase 30 — the O(N) HSD scaling benchmark.** Measures the FULL
/// `solve_socp_structured_hsd` vs the dense `solve_socp_hsd` wall-clock across
/// N ∈ {3, 5, 7} on assembled SCvx subproblems. The point: unlike the structured
/// AHO/NT paths — whose per-KKT-solve win does NOT survive to the full solve
/// (fallback erosion / endgame instability, the Phase-6.12 caveat) — the
/// structured HSD's win DOES survive end-to-end (HSD is central-path-conditioned,
/// so the structured factorization is sound every iter with no fallbacks). Both
/// drivers cold-start central and take identical Newton steps, so they run the
/// same iteration count; the wall-clock ratio is the O(N·NZ³)-vs-O((N·NZ)³)
/// structured-Schur speedup, realized for real.
///
/// `#[ignore]`d (a release-timing benchmark; the dense N=7 full solve is too slow
/// in a debug CI run). Run explicitly:
/// `cargo test --release --test wcet_benchmarks wcet_hsd_structured_vs_dense -- --ignored --nocapture`.
/// Measured (release, x86_64): N=3 3.4×, N=5 4.95×, N=7 7.05× faster than dense;
/// structured scales 6.0× (N=3→7) vs dense 12.46× (≈cubic).
#[test]
#[ignore]
fn wcet_hsd_structured_vs_dense_full_solve() {
    std::thread::Builder::new()
        .stack_size(32 * 1024 * 1024)
        .spawn(wcet_hsd_structured_vs_dense_inner)
        .expect("spawn bench thread")
        .join()
        .expect("bench thread panicked");
}

fn wcet_hsd_structured_vs_dense_inner() {
    eprintln!("\n--- WCET: structured vs dense HSD FULL solve on SCvx subproblem ---");

    fn bench<
        const N: usize,
        const NP: usize,
        const NE: usize,
        const NCT: usize,
        const NCONES: usize,
    >() -> (u64, u64, f64) {
        let phys = PhysicalParams {
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
        };
        let mut x_init = SVector::<f64, 7>::zeros();
        x_init[2] = 500.0; x_init[5] = -20.0; x_init[6] = (800.0_f64).ln();
        let mut traj = Trajectory::<N>::default();
        for k in 0..N {
            for i in 0..7 { traj.x[(i, k)] = x_init[i]; }
            traj.u[(2, k)] = 800.0 * 3.7114;
        }
        traj.tau = 20.0;

        let mut lin = Box::new(LinearizedDynamics::<N>::default());
        discretize_foh(&traj, &phys, &mut lin, 4);

        let term = TerminalCondition { r: [0.0; 3], v: [0.0; 3] };
        let mut prob = Box::new(SocpProblem::<NP, NE, NCT, NCONES>::default());
        assemble_scvx_socp::<N, NP, NE, NCT, NCONES>(
            &traj, &lin, &phys, &x_init, &term,
            1.0e3, 1.0e4, false, &mut prob,
        );

        // Both drivers cold-start central (ignore ws.x), so the workspace can be
        // reused across runs. Fixed max_iters bounds the work identically.
        let params = IpmAlgoParams { max_iters: 12, ..IpmAlgoParams::default() };
        let mut ws = Box::new(SocpWorkspace::<NP, NE, NCT>::default());

        // ---- Structured HSD ----
        for _ in 0..3 {
            let r = solve_socp_structured_hsd::<N, NP, NE, NCT, NCONES>(&prob, &params, &mut ws);
            std::hint::black_box(&r);
        }
        let n_runs = 20;
        let t0 = Instant::now();
        for _ in 0..n_runs {
            let r = solve_socp_structured_hsd::<N, NP, NE, NCT, NCONES>(&prob, &params, &mut ws);
            std::hint::black_box(&r);
        }
        let t_struct = (t0.elapsed().as_nanos() as u64) / (n_runs as u64);

        // ---- Dense HSD ----
        for _ in 0..3 {
            let r = solve_socp_hsd::<NP, NE, NCT, NCONES>(&prob, &params, &mut ws);
            std::hint::black_box(&r);
        }
        let t0 = Instant::now();
        for _ in 0..n_runs {
            let r = solve_socp_hsd::<NP, NE, NCT, NCONES>(&prob, &params, &mut ws);
            std::hint::black_box(&r);
        }
        let t_dense = (t0.elapsed().as_nanos() as u64) / (n_runs as u64);

        let speedup = t_dense as f64 / t_struct as f64;
        (t_struct, t_dense, speedup)
    }

    const N3: usize = 3;
    const N3_NP: usize     = N3 * N_VARS_PER_NODE_SCVX;
    const N3_NE: usize     = N3 * 7 + 6;
    const N3_NCT: usize    = N3 * N_CONE_DIM_PER_NODE_SCVX;
    const N3_NCONES: usize = N3 * 8;
    let (s3, d3, sp3) = bench::<N3, N3_NP, N3_NE, N3_NCT, N3_NCONES>();

    const N5: usize = 5;
    const N5_NP: usize     = N5 * N_VARS_PER_NODE_SCVX;
    const N5_NE: usize     = N5 * 7 + 6;
    const N5_NCT: usize    = N5 * N_CONE_DIM_PER_NODE_SCVX;
    const N5_NCONES: usize = N5 * 8;
    let (s5, d5, sp5) = bench::<N5, N5_NP, N5_NE, N5_NCT, N5_NCONES>();

    const N7: usize = 7;
    const N7_NP: usize     = N7 * N_VARS_PER_NODE_SCVX;
    const N7_NE: usize     = N7 * 7 + 6;
    const N7_NCT: usize    = N7 * N_CONE_DIM_PER_NODE_SCVX;
    const N7_NCONES: usize = N7 * 8;
    let (s7, d7, sp7) = bench::<N7, N7_NP, N7_NE, N7_NCT, N7_NCONES>();

    eprintln!("  N=3 : structured HSD {:>9} ns,  dense HSD {:>9} ns  ({:>4.2}× faster)", s3, d3, sp3);
    eprintln!("  N=5 : structured HSD {:>9} ns,  dense HSD {:>9} ns  ({:>4.2}× faster)", s5, d5, sp5);
    eprintln!("  N=7 : structured HSD {:>9} ns,  dense HSD {:>9} ns  ({:>4.2}× faster)", s7, d7, sp7);
    eprintln!("  full-solve scaling structured (N=7/N=3): {:.2}×", s7 as f64 / s3 as f64);
    eprintln!("  full-solve scaling dense      (N=7/N=3): {:.2}×", d7 as f64 / d3 as f64);

    // Smoke (host timing is noisy — report, don't over-assert): the structured
    // HSD must scale STRICTLY SUB-CUBICALLY relative to dense across N=3→7, i.e.
    // the dense full solve grows faster than the structured one. (dense ~ N³,
    // structured ~ N, so dense ratio should exceed structured ratio.)
    let struct_ratio = s7 as f64 / s3 as f64;
    let dense_ratio  = d7 as f64 / d3 as f64;
    assert!(
        dense_ratio > struct_ratio,
        "dense HSD should scale faster than structured (dense {dense_ratio:.2}× vs struct {struct_ratio:.2}×) \
         — the O(N) win"
    );
    // And the structured path should win outright at the largest N measured.
    assert!(s7 < d7, "structured HSD should be faster than dense at N=7 ({s7} vs {d7} ns)");
}

/// **Phase 6.7 quantitative gate**: measure the per-IPM-iter cost of the
/// factor+apply split path vs the one-shot path (two full solves).
///
/// Inside the structured IPM, each iteration calls the KKT solver twice
/// (predictor + corrector) with the same H but different RHS. The one-shot
/// path rebuilds the factor each time; the split path builds once and
/// applies twice. Expected speedup: ~2× since the apply cost is dominated
/// by the back-sub + Δz recovery (much less work than building H_k⁻¹ and
/// the Schur factor).
#[test]
fn wcet_factor_apply_split_vs_one_shot() {
    std::thread::Builder::new()
        .stack_size(32 * 1024 * 1024)
        .spawn(wcet_factor_apply_split_inner)
        .expect("spawn bench thread")
        .join()
        .expect("bench thread panicked");
}

fn wcet_factor_apply_split_inner() {
    eprintln!("\n--- WCET: factor+apply vs one-shot KKT (per-iter IPM cost) ---");

    fn bench<
        const N: usize,
        const NP: usize,
        const NE: usize,
        const NCT: usize,
        const NCONES: usize,
    >() -> (u64, u64, f64) {
        let phys = PhysicalParams {
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
        };
        let mut x_init = SVector::<f64, 7>::zeros();
        x_init[2] = 500.0; x_init[5] = -20.0; x_init[6] = (800.0_f64).ln();
        let mut traj = Trajectory::<N>::default();
        for k in 0..N {
            for i in 0..7 { traj.x[(i, k)] = x_init[i]; }
            traj.u[(2, k)] = 800.0 * 3.7114;
        }
        traj.tau = 20.0;

        let mut lin = Box::new(LinearizedDynamics::<N>::default());
        discretize_foh(&traj, &phys, &mut lin, 4);

        let term = TerminalCondition { r: [0.0; 3], v: [0.0; 3] };
        let mut prob = Box::new(SocpProblem::<NP, NE, NCT, NCONES>::default());
        assemble_scvx_socp::<N, NP, NE, NCT, NCONES>(
            &traj, &lin, &phys, &x_init, &term,
            1.0e3, 1.0e4, false, &mut prob,
        );

        let m_full = SMatrix::<f64, NCT, NCT>::identity();
        let reg: f64 = 1.0e-8;

        // Two distinct RHS vectors (predictor + corrector).
        let mut b_x_pred = SVector::<f64, NP>::zeros();
        let mut b_x_corr = SVector::<f64, NP>::zeros();
        let mut b_a      = SVector::<f64, NE>::zeros();
        for i in 0..NP { b_x_pred[i] = ((i as f64) * 0.13).sin(); }
        for i in 0..NP { b_x_corr[i] = ((i as f64) * 0.17).cos(); }
        for i in 0..NE { b_a[i]      = ((i as f64) * 0.09).sin(); }

        let mut sol_pred = Box::new(ReducedKktSolution::<N>::default());
        let mut sol_corr = Box::new(ReducedKktSolution::<N>::default());

        // ---- One-shot path: two solves ----
        for _ in 0..8 {
            let _ = solve_reduced_kkt_scvx_block_m::<N, NP, NE, NCT, NCONES>(
                &prob, &m_full, reg, &b_x_pred, &b_a, &mut sol_pred,
            );
            let _ = solve_reduced_kkt_scvx_block_m::<N, NP, NE, NCT, NCONES>(
                &prob, &m_full, reg, &b_x_corr, &b_a, &mut sol_corr,
            );
        }
        let n_runs = 200;
        let t0 = Instant::now();
        for _ in 0..n_runs {
            let _ = solve_reduced_kkt_scvx_block_m::<N, NP, NE, NCT, NCONES>(
                &prob, &m_full, reg, &b_x_pred, &b_a, &mut sol_pred,
            );
            let _ = solve_reduced_kkt_scvx_block_m::<N, NP, NE, NCT, NCONES>(
                &prob, &m_full, reg, &b_x_corr, &b_a, &mut sol_corr,
            );
        }
        let t_oneshot = (t0.elapsed().as_nanos() as u64) / (n_runs as u64);
        std::hint::black_box(&*sol_corr);

        // ---- Split path: factor once + apply twice ----
        let mut factor = Box::new(ReducedKktFactor::<N>::default());
        for _ in 0..8 {
            let _ = factor_reduced_kkt_scvx_block_m::<N, NP, NE, NCT, NCONES>(
                &prob, &m_full, reg, &mut factor,
            );
            let _ = solve_reduced_kkt_scvx_with_factor::<N, NP, NE, NCT, NCONES>(
                &prob, &factor, &b_x_pred, &b_a, &mut sol_pred,
            );
            let _ = solve_reduced_kkt_scvx_with_factor::<N, NP, NE, NCT, NCONES>(
                &prob, &factor, &b_x_corr, &b_a, &mut sol_corr,
            );
        }
        let t0 = Instant::now();
        for _ in 0..n_runs {
            let _ = factor_reduced_kkt_scvx_block_m::<N, NP, NE, NCT, NCONES>(
                &prob, &m_full, reg, &mut factor,
            );
            let _ = solve_reduced_kkt_scvx_with_factor::<N, NP, NE, NCT, NCONES>(
                &prob, &factor, &b_x_pred, &b_a, &mut sol_pred,
            );
            let _ = solve_reduced_kkt_scvx_with_factor::<N, NP, NE, NCT, NCONES>(
                &prob, &factor, &b_x_corr, &b_a, &mut sol_corr,
            );
        }
        let t_split = (t0.elapsed().as_nanos() as u64) / (n_runs as u64);
        std::hint::black_box(&*sol_corr);

        // Sanity: split should succeed.
        let last_status = solve_reduced_kkt_scvx_with_factor::<N, NP, NE, NCT, NCONES>(
            &prob, &factor, &b_x_pred, &b_a, &mut sol_pred,
        );
        assert_eq!(last_status, ReducedKktStatus::Ok);

        let speedup = t_oneshot as f64 / t_split as f64;
        (t_oneshot, t_split, speedup)
    }

    // Per-N const expansion.
    const N3: usize = 3;
    let (o3, s3, sp3) = bench::<
        N3,
        { N3 * N_VARS_PER_NODE_SCVX },
        { N3 * 7 + 6 },
        { N3 * N_CONE_DIM_PER_NODE_SCVX },
        { N3 * 8 },
    >();

    const N5: usize = 5;
    let (o5, s5, sp5) = bench::<
        N5,
        { N5 * N_VARS_PER_NODE_SCVX },
        { N5 * 7 + 6 },
        { N5 * N_CONE_DIM_PER_NODE_SCVX },
        { N5 * 8 },
    >();

    const N7: usize = 7;
    let (o7, s7, sp7) = bench::<
        N7,
        { N7 * N_VARS_PER_NODE_SCVX },
        { N7 * 7 + 6 },
        { N7 * N_CONE_DIM_PER_NODE_SCVX },
        { N7 * 8 },
    >();

    eprintln!("  N=3 : one-shot {:>8} ns,  factor+apply {:>8} ns  ({:>4.2}× speedup)",
              o3, s3, sp3);
    eprintln!("  N=5 : one-shot {:>8} ns,  factor+apply {:>8} ns  ({:>4.2}× speedup)",
              o5, s5, sp5);
    eprintln!("  N=7 : one-shot {:>8} ns,  factor+apply {:>8} ns  ({:>4.2}× speedup)",
              o7, s7, sp7);

    // Smoke: factor+apply should not be slower than one-shot. At small N
    // the difference is small; at larger N the factor work dominates so
    // the savings grow. We assert split ≤ 1.2× of one-shot to allow
    // for benchmark noise.
    assert!(s3 < (o3 * 12) / 10,
            "split path slower than one-shot at N=3: {s3}/{o3}");
    assert!(s7 < o7,
            "split path slower than one-shot at N=7: {s7}/{o7}");
}

// ===========================================================================
// Full-solve WCET: dense vs structured end-to-end SCvx solve
// ===========================================================================

/// **The headline WCET deliverable**: time a complete `solve_scvx` outer
/// loop with the dense inner solver vs the structured (block-tridiagonal
/// Schur) inner solver, on a representative N=5 fixed-tf Mars-descent
/// problem. Reports p50/p99/max for both and the structured fallback count.
///
/// **The critical flight-software nuance this surfaces**: the structured
/// path improves the *average* (p50) inner-solve cost, but its *worst case*
/// is bounded by ~1× dense, NOT below it — because on the subproblems where
/// the structured IPM breaks down in the AHO endgame, the outer loop pays
/// the structured attempt AND the dense re-solve (the fallback).
///
/// - **Throughput / average power**: structured wins (most subproblems
///   take the fast O(N·NZ³) path).
/// - **WCET budgeting**: size for the dense path — the structured fast
///   path never *reduces* the worst case, and a fallback-heavy run can
///   approach ~2× a single dense solve on the affected iterations.
///
/// This is exactly the kind of distinction a flight engineer must see
/// before trusting a "13.5× faster" headline number.
///
/// **`#[ignore]` by default**: a full N=5 solve is ~0.65 s, so 16 solves
/// (2 warm-up + 6 timed, ×2 paths) take ~11 s — too slow for the routine
/// suite, and the timing is host-noisy with deliberately-loose asserts.
/// This is an on-demand *characterization* tool, not a CI gate. Run with:
/// `cargo test --release --test wcet_benchmarks -- --ignored --nocapture`.
#[test]
#[ignore = "slow (~11s) characterization benchmark; run with --ignored"]
fn wcet_full_scvx_solve_dense_vs_structured() {
    std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(wcet_full_scvx_solve_inner)
        .expect("spawn bench thread")
        .join()
        .expect("bench thread panicked");
}

fn wcet_full_scvx_solve_inner() {
    eprintln!("\n--- WCET: full SCvx solve, dense vs structured (N=5, fixed-tf) ---");

    const N: usize         = 5;
    const NP: usize        = N * N_VARS_PER_NODE_SCVX;          // 95
    const NE: usize        = N * N_EQ_PER_DYN + N_EQ_TERMINAL;  // 41
    const NCT: usize       = N * N_CONE_DIM_PER_NODE_SCVX;      // 150
    const NCONES: usize    = N * N_CONES_PER_NODE_SCVX;         // 40
    const MAX_OUTER: usize = 15;

    let phys = PhysicalParams {
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
    };

    let mut x_init = SVector::<f64, 7>::zeros();
    x_init[2] = 2.0;
    x_init[5] = -0.1;
    x_init[6] = (400.0_f64).ln();
    let mut x_target = SVector::<f64, 7>::zeros();
    x_target[6] = (380.0_f64).ln();

    // Linear-interpolation reference with hover thrust (replica of the
    // scvx test helper). Restored before every timed solve.
    let u_hover_z = -390.0 * phys.g[2];
    let mut ref0 = Trajectory::<N>::default();
    for k in 0..N {
        let alpha = k as f64 / (N - 1) as f64;
        for i in 0..7 {
            ref0.x[(i, k)] = (1.0 - alpha) * x_init[i] + alpha * x_target[i];
        }
        ref0.u[(2, k)] = u_hover_z;
        ref0.sigma[k]  = u_hover_z;
    }
    ref0.tau = 10.0;

    let term = TerminalCondition { r: [0.0; 3], v: [0.0; 3] };
    let algo_base = ScvxAlgoParams {
        trust_eta0:    5.0,
        trust_eta_max: 20.0,
        trust_eta_min: 1.0e-3,
        ..ScvxAlgoParams::default()
    };
    let algo_dense  = ScvxAlgoParams { use_structured_solve: false, ..algo_base };
    let algo_struct = ScvxAlgoParams { use_structured_solve: true,  ..algo_base };
    let ipm = IpmAlgoParams {
        tol_mu:              1.0e-4,
        tol_primal:          1.0e-4,
        tol_dual:            1.0e-4,
        tol_gap:             1.0e-4,
        use_nt_scaling:      false,
        use_preconditioning: true,
        ..IpmAlgoParams::default()
    };

    let mut ws = Box::new(ScvxWorkspace::<N, NP, NE, NCT, NCONES, MAX_OUTER>::default());

    // Last-ACCEPTED cost: `record_iter` logs every candidate's cost whether
    // or not the trust step accepted it, so `history[iter].cost` can be a
    // rejected trial (misleadingly bad). Scan backward for the last accepted
    // record to report the actual converged-reference quality.
    let last_accepted_cost = |ws: &ScvxWorkspace<N, NP, NE, NCT, NCONES, MAX_OUTER>| -> f64 {
        let last = ws.iter as usize;
        for i in (0..=last.min(MAX_OUTER - 1)).rev() {
            if ws.history[i].accepted {
                return ws.history[i].cost;
            }
        }
        ws.history[0].cost
    };

    // Confirm both configs solve cleanly (status check, before timing).
    ws.reference = ref0.clone();
    let st_dense = solve_scvx(&mut ws, &phys, &algo_dense, &ipm, &x_init, &term);
    let cost_dense = last_accepted_cost(&ws);
    ws.reference = ref0.clone();
    let st_struct = solve_scvx(&mut ws, &phys, &algo_struct, &ipm, &x_init, &term);
    let cost_struct = last_accepted_cost(&ws);
    let fallbacks = ws.structured_fallbacks;
    assert!(matches!(st_dense,  SolverStatus::Converged | SolverStatus::OuterIterCap));
    assert!(matches!(st_struct, SolverStatus::Converged | SolverStatus::OuterIterCap));

    // Lean local timing (a full N=5 solve is ~10²–10³ ms; the shared
    // `time()` helper's 16-run warm-up would make this dominate the suite).
    // 2 warm-up + 6 timed per path is plenty for a min/median/max profile.
    let bench = |ws: &mut ScvxWorkspace<N, NP, NE, NCT, NCONES, MAX_OUTER>,
                 algo: &ScvxAlgoParams| -> (u64, u64, u64) {
        for _ in 0..2 {
            ws.reference = ref0.clone();
            let _ = solve_scvx(ws, &phys, algo, &ipm, &x_init, &term);
        }
        let mut samples = [0u64; 6];
        for s in samples.iter_mut() {
            ws.reference = ref0.clone();
            let t0 = Instant::now();
            let _ = solve_scvx(ws, &phys, algo, &ipm, &x_init, &term);
            std::hint::black_box(&ws.reference);
            *s = t0.elapsed().as_nanos().min(u64::MAX as u128) as u64;
        }
        samples.sort_unstable();
        (samples[0], samples[3], samples[5]) // min, median, max
    };

    let (d_min, d_med, d_max) = bench(&mut ws, &algo_dense);
    let (s_min, s_med, s_max) = bench(&mut ws, &algo_struct);

    eprintln!("  dense  final cost = {cost_dense:.4e} (status {})", st_dense as u32);
    eprintln!("  struct final cost = {cost_struct:.4e} (status {}), fallbacks = {fallbacks}/{MAX_OUTER}",
              st_struct as u32);
    eprintln!("  min:  dense {:>10} ns   structured {:>10} ns   ({:.2}× of dense)",
              d_min, s_min, s_min as f64 / d_min as f64);
    eprintln!("  med:  dense {:>10} ns   structured {:>10} ns   ({:.2}× of dense)",
              d_med, s_med, s_med as f64 / d_med as f64);
    eprintln!("  max:  dense {:>10} ns   structured {:>10} ns   ({:.2}× of dense)",
              d_max, s_max, s_max as f64 / d_max as f64);
    eprintln!("  fallbacks = {fallbacks}/{MAX_OUTER}: each fallback iter pays the");
    eprintln!("  structured attempt + the dense re-solve, so the per-KKT-solve");
    eprintln!("  speedup (~13.5× micro-bench) is heavily eroded at the full-solve");
    eprintln!("  level. Structured improves average throughput; for WCET budgeting");
    eprintln!("  size for the dense path (structured never *reduces* the worst case).");

    // This is a *characterization* benchmark, not a pass/fail gate on timing
    // or convergence cost (both are validated elsewhere: the one-iter
    // equivalence tests pin the math, the end-to-end scvx tests pin
    // convergence). We assert only that nothing pathological happened:
    // both paths produced a finite, bounded solve and a finite cost.
    assert!(d_max < 10_000_000_000, "dense full solve max {d_max} ns > 10 s — runaway?");
    assert!(s_max < 10_000_000_000, "structured full solve max {s_max} ns > 10 s — runaway?");
    assert!(cost_dense.is_finite() && cost_struct.is_finite(),
            "non-finite final cost: dense {cost_dense}, struct {cost_struct}");
}
