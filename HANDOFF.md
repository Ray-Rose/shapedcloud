# HANDOFF — SCvx Powered-Descent Solver (Rust, no_std)

**Read top to bottom before touching code.** The first ~15 minutes here
save hours of re-derivation. This document was written immediately before
a context-window roll-over so the next session starts coherent.

---

## TL;DR

- **What**: Rust workspace implementing a Successive-Convexification (SCvx)
  trajectory-optimization solver for **3-DoF powered descent** with drag and
  free-final-time. Static-memory, no-`std`, no-`alloc`, no-`panic`,
  bounded-WCET — "research-grade flight-shaped" per the original plan.
- **Where**: the repository workspace root (all paths below are repo-relative)
- **State**: **132 tests pass** across **5 crates**,
  `cargo clippy --all-targets -- -D warnings` clean,
  `cargo build --release --target thumbv7em-none-eabihf -p scvx-solver`
  clean (the no_std flight crates cross-compile to ARM Cortex-M), and the
  **C-FFI layer now cross-compiles to bare-metal too** (feature-gated no_std).
  **Phase 10 (v1 audit & packaging) landed**: a 4-agent red-team audit found
  and fixed a **Critical** FFI bug (a `use_free_tf` flag / entrypoint-dim
  mismatch — reachable from the most natural C usage — that wrote out of bounds
  → panic/abort; now returns `BadInput` at three layers with a regression
  test), plus High/Medium input-validation hardening (`Isp·g0>0`, full
  warm-start finiteness, cone-descriptor bounds) and doc-accuracy fixes. The
  **FFI surface expanded to N ∈ {3,5,8,10,12,15,20}** (fixed + free-tf, DRY via
  Rust+C macros; header verified to compile as C99 and C++11; a pure-C example
  links and runs). A **flight-integration guide** lives at
  `crates/scvx-ffi/INTEGRATION.md`.
  Architecture is **complete end-to-end** with **six SCvx phases
  landed**: column preconditioning (P11), cone-row scaling (Phase 2),
  free-final-time τ (Phase 3), application API + example (Phase 4),
  C-FFI + NT robustness + N=10 scale-up (Phase 5), and now the
  **Riccati structured-KKT primitive** (Phase 6, P3b). With AHO +
  column preconditioning + free-tf, the application demo produces a
  real Mars-descent trajectory: 2m altitude → soft landing,
  τ adjusted 8.0s → 14.029s, ‖ν‖₁ = 3.02e-9 (near machine precision),
  ~9 kg propellant burned. Run `cargo run --release --example
  mars_descent` for the full trace. The C-FFI crate (`scvx-ffi`) builds
  a `.lib`/`.dll`/`.so` linkable from flight C/C++ code, with a
  hand-written header in `crates/scvx-ffi/include/scvx_ffi.h`.
- **Recommended production pairing**:
  - **Fixed-tf**: AHO + column + row preconditioning. Best convergence
    (machine precision `‖ν‖`).
  - **Free-tf**: AHO + column preconditioning, `use_free_tf = true`.
    Cone-row scaling with free-tf works but adds extra dynamics that
    can stress the IPM at scale; column-only is the safe default.
- **Phase 6 landed (Riccati / block-tridiagonal Schur primitives)**:
  - `scvx-ipm/src/block_tridiag.rs` — generic block-tridiagonal LU
    factor+solve (block-Thomas), `O(N·B³)` cost. **5 tests**, including
    dense-LU oracle agreement on a 15×15 system at ≤ 1e-10.
  - `scvx-solver/src/reduced_kkt.rs` — SCvx-aware reduced-KKT solver.
    Exploits the property that **all SCvx cones are stage-local**, so
    `H = GᵀMG` is block-diagonal per stage and `S = A·H⁻¹·Aᵀ` is
    block-tridiagonal. Two public entry points:
    - `solve_reduced_kkt_scvx(...)` — diagonal-only cone scaling
      (cheap, useful when cone-row preconditioning has reduced M to
      near-diagonal).
    - `solve_reduced_kkt_scvx_block_m(...)` — full per-cone block-dense
      scaling (the form AHO and NT actually produce).
    **4 tests**, including dense-LU oracle agreement at ≤ 1e-9 (N=3),
    ≤ 1e-8 (N=5), and block-M ≤ 1e-9 (N=3 with synthetic dense per-cone
    M). Cost `O(N·NZ³)` where `NZ = 19` for fixed-tf SCvx.

- **Phases 6.5–6.10 landed (structured IPM drivers — the wiring is DONE)**:
  the structured Schur solver is now wired end-to-end through the public
  SCvx API via `ScvxAlgoParams::use_structured_solve`. The dispatch matrix
  in `solve_scvx` is **complete** — all four cells have a machine-precision-
  verified structured driver with a direction-matched dense fallback:
  - `solve_socp_structured`          (AHO, fixed-tf)
  - `solve_socp_structured_free_tf`  (AHO, free-tf — Sherman-Morrison δτ)
  - `solve_socp_structured_nt`       (NT,  fixed-tf — `W²` scaling)
  - `solve_socp_structured_nt_free_tf` (NT, free-tf — `W²` + SMW δτ)
  Factor/apply split (`factor_*` + `solve_*_with_factor`) gives ~2× per-iter
  reuse; the **per-KKT-solve** micro-benchmark is ~13.5× at N=7. Each driver
  has a one-iter equivalence test vs its dense reference at ≤ 1e-12 (AHO) /
  ≤ 1e-12 (NT, modulo the eigendecomp-vs-DB matrix-sqrt difference).

  ⚠️ **CRITICAL CAVEAT — the 13.5× does NOT survive to the full solve**
  (Phase 6.12 full-solve WCET benchmark, `wcet_full_scvx_solve_dense_vs_
  structured`, `#[ignore]`d): on a real N=5 fixed-tf solve the structured
  path is **a wash on time (median 0.97× dense)** and converges to a
  **worse** cost (1.16e4 vs dense 7.29e3). Two compounding reasons: (1)
  fallback erosion — 9 of 15 outer iters break down in the AHO endgame and
  re-solve with dense, paying *both*; (2) on the ~6 non-fallback iters the
  structured fp trajectory diverges enough to steer the trust-region/accept
  logic to a worse local fixed point. **Net: the structured fast path does
  not yet deliver an end-to-end win.** It is a *verified-correct prototype*
  (machine-precision per step) whose payoff is blocked by the same AHO-
  endgame instability as the fallback problem — realizing it needs the
  research-grade stable factorization (regularized-Schur / QR), not more
  wiring. **Dense remains the correct production default** (`use_structured_
  solve = false`).

- **Genuinely-open future work** (extensions, not gaps — nothing blocks
  flight-readiness):
  1. **NT convergence on flight-scale subproblems** (Phase 8 — bottleneck
     re-diagnosed; the matrix-sqrt is NOT it). **The per-D Higham-scaled
     matrix-sqrt is already implemented** — `socp.rs::emit_nt_w_specialized!`
     does eigendecomposition-first (`SymmetricEigen`) with a Higham-scaled
     Denman-Beavers fallback, per fixed `D ∈ {1,3,4,8,11}`. So the dense NT
     driver already builds `W` robustly. **It does not fix NT convergence.**
     Measured (`nt_full_precond_fails_gracefully` trace): NT diverges at
     **outer iteration 0** — the inner IPM runs ~14–20 inner iters then
     `NumericalError` (NaN) on the *very first* flight-scale SCvx subproblem.
     This is **independent of cone-row scaling** (fails the same with
     column-precond only, ~20 inner iters) and independent of the W (robust).
     Yet the NT inner solver passes the toy-SOCP unit tests
     (`nt_solves_two_cone_problem`, `nt_toy_socp_recovers_closed_form`), so
     it is not fundamentally broken — it diverges *specifically* on the
     ill-conditioned, many-cone SCvx subproblem (30 cone-dims/node: thrust
     SOC^4 + trust SOC^11 + virtual-control SOC^8 + glide SOC^3 + bounds,
     spanning ~6 orders of magnitude). **The real open problem is the NT
     Mehrotra direction/centering on that cone structure** — a genuine
     IPM-theory effort (e.g., a different centering/neighborhood strategy,
     or a cone-by-cone NT scaling that respects the trust + virtual-control
     blocks), NOT more matrix-sqrt work. AHO remains the robust production
     default; NT stays an opt-in that fails gracefully where it can't
     converge.

     **Phase 9 — research sprint outcome (deep-research → implement → measure).**
     A focused `/deep-research` sprint (105 agents, 23 sources) ranked the
     **weighted (Colombo–Gondzio) corrector** as the leading citable remedy,
     plus the SDPT3 adaptive centering exponent and adaptive fraction-to-
     boundary. All three landed in `solve_socp_nt` (and are mirrored into both
     structured NT drivers — see below). **Measured effect, honestly:**
     - The corrector + adaptive σ/γ **pushed the inner-IPM breakdown from
       ~iter 14 out to ~iter 35** and eliminated the mid-path NaN — a real,
       reproducible stability gain. The 19 NT unit tests stay green (ω=1
       recovers textbook Mehrotra on the well-conditioned toys).
     - It did **not** achieve convergence on the flight subproblem; the
       endgame still breaks down (NumericalError) as μ→0.
     - **Iterative refinement of the reduced-KKT solve was then implemented,
       measured, and reverted.** Even *monitored* IR (accept a correction only
       if it strictly reduces the KKT residual) moved the breakdown *earlier*
       (iter 35→30). The decisive finding: the accepted corrections genuinely
       reduce the linear-solve residual, yet the IPM breaks sooner — so a
       *more-accurate* NT direction breaks down faster. **The breakdown is
       degeneracy of the Newton *linearization* as μ→0, NOT linear-solve
       accuracy.** This also rules out the heavier accurate-solve remedies
       (Krylov/PSQMR) the research surfaced — they would not help either. A
       `NOTE` in `solve_newton_step_nt` records this so no one re-treads it.
     - **Net**: the genuine corrector win is kept; the dead-end (IR) is gone.
       The remaining NT frontier is an algorithm change (neighborhood/
       centering theory or per-cone scaling), not a numerics tweak.
  2. **Reduce structured-driver fallback frequency** (Phase 6.11 —
     **root cause now diagnosed, not yet eliminated**). On the Mars demo the
     structured AHO driver falls back to dense on ~5 of 15 outer iters
     (measured via the new `ScvxWorkspace::structured_fallbacks` telemetry
     counter). Two findings landed:
     - **Snapshot-preferring bail** (DONE): all four structured drivers now
       return their captured best-feasible snapshot as `BestFeasible` on a
       mid-loop numerical breakdown, instead of scrubbing to `NumericalError`
       — matching the dense driver's `numerical_exit` semantics. Correct
       improvement (helps whenever a snapshot exists at breakdown), but it
       did **not** move the Mars-demo count because `best_valid` is false at
       every fallback there.
     - **Proven root cause** (NOT slow convergence): raising `max_inner_iters`
       from 25→40 leaves the count at 5 (and pushes one case from IterCap
       into an earlier NumericalError). The structured block-tridiagonal
       factorization amplifies the **AHO endgame ill-conditioning**
       (`arrow(s)·arrow(y) → singular` at the cone boundary) differently from
       dense LU; on those subproblems μ never drops below the loose-feasible
       threshold (1e-2) before breakdown. The dense fallback keeps
       correctness (outer loop converges to within ~1% of dense).
     **Three cheap/standard levers tried and ruled out** (Phase 6.11):
     - *Snapshot-preferring bail*: landed, but didn't move the count
       (`best_valid` is false at every breakdown — no snapshot to keep).
     - *More inner iters* (25→40): no help; one IterCap case became an
       earlier NumericalError. Confirms breakdown, not slow convergence.
     - *Higher regularization* (1e-8→1e-6): **worse** (5→8 fallbacks). Too
       much reg biases the Newton step and slows convergence, so the loose-
       feasible snapshot is reached even less often. The structured reg
       matching the dense reg (1e-8) is optimal; deviating diverges the two
       fp trajectories further.
     **Conclusion**: the fallback is *intrinsic* — block-tridiagonal Schur
     and dense LU are two distinct backward-stable factorizations that
     diverge chaotically in the ill-conditioned AHO endgame (`arrow(s)·
     arrow(y) → singular`), and on ~1/3 of subproblems the structured path
     is the unlucky one. Per-step accuracy is NOT the bottleneck (the
     equivalence tests show 1e-13 per-step match), so iterative refinement
     would not help either. Reducing it further needs a fundamentally
     more stable structured factorization (regularized-Schur or QR-based)
     — research-grade, well beyond tuning. The dense fallback guarantees
     correctness; the structured fast path delivers its speedup on the
     ~2/3 of subproblems where it succeeds.
  3. **Widen the convergence envelope beyond the Mars no-drag sweet spot**
     (Phase 7 finding) — **ADDRESSED for drag + non-Mars gravity by the
     Phase 17 trust-shrink retry** (see the Phase 17 section). The old framing:
     the shipping default (AHO + column preconditioning) converged cleanly on
     Mars no-drag, but **active drag** (`rho=0.02, cd_a=50`) or a **gravity
     change** (Mars→lunar) tripped the AHO endgame after a few outer iters and
     the outer loop *aborted* (`InnerFailure`). The root cause was NOT the
     drag/gravity code paths (correct + exercised) nor preconditioning (scale
     table is identical) — it was the outer loop **giving up on the first
     unsolvable subproblem** instead of shrinking the trust and re-solving.
     Phase 17 makes that the standard trust-region retry: **active drag now
     reaches min ‖ν‖ = 1.58e-10 (base AHO + column precond); lunar reaches
     3.45e-10 (+ cone-row scaling)** — machine-precision dynamics feasibility,
     vs the old `InnerFailure` at ‖ν‖ ≈ 0.4–0.7. Asserted in
     `scvx_active_drag_path_exercised_and_handled` (upgraded to a convergence
     gate) and the new `scvx_converges_lunar_gravity`. **Remaining**: the
     solver reaches ‖ν‖→0 but `OuterIterCap`s (it oscillates around the optimum
     at large `eta_max` rather than formally `Converged`), and problems the
     retry can't tame still bottom out on the item-1 vanishing-cone endgame —
     the deeper frontier.

---

## Quick verification (run this first to confirm baseline)

```sh
cd <repo-root>   # the workspace root of this repository
cargo test 2>&1 | grep -E "test result"
cargo clippy --all-targets -- -D warnings
cargo build --release --target thumbv7em-none-eabihf
```

Expected: 132 tests pass (split as `0+17+46+48+3+2+8+8`
= core 0, dynamics 17, ipm 46, solver-lib 48, oracle_diff 3,
oracle_scvx_subproblem 2, wcet 8, ffi 8;
**note**: a Windows Application Control / Defender policy sometimes
transiently blocks a freshly-recompiled debug test binary (`os error
4551`); if `cargo test` aborts mid-run with "An Application Control
policy has blocked this file", re-run with `--release` (different binary
path, reliably clears the scan) — it is an environment quirk, not a code
failure,
where the +5 in scvx-ipm is `block_tridiag`, the +7 in scvx-solver
includes `reduced_kkt` (5 tests including the full-Newton-step
dense-vs-structured equivalence gate), `structured_socp` (1 test:
single-iter live-driver equivalence to machine precision), and
`scvx_converges_with_structured_solve` (1 test: end-to-end SCvx outer
loop with `use_structured_solve = true`), the +2 in scvx-ffi are
misalignment + pathological-input defense tests, and the +1 in
wcet_benchmarks is the structured-vs-dense KKT timing benchmark),
clippy silent, thumb release silent. **If any of these fail, stop and
diagnose before doing anything else.**

Plus the example binary:

```sh
cargo run --release --example mars_descent
```

Should print a Mars-descent trajectory and exit cleanly.

### Docker clean-room verification (recommended — bypasses the WDAC block)

The Windows host's Application Control / Defender policy intermittently blocks
freshly-compiled **debug** test binaries AND the example binary from executing
(`os error 4551`). The reliable workaround is a clean-room Linux build in
Docker (Docker Desktop with the linux/amd64 engine is present on this host):

```sh
# From the repo root. CARGO_TARGET_DIR is set to a container-local path so the
# Linux build never collides with the Windows MSVC target/ cache (and vice versa).
MSYS_NO_PATHCONV=1 docker run --rm \
  -v "<ABSOLUTE_PATH_TO_REPO_ROOT>":/work \
  -w /work -e CARGO_TARGET_DIR=/build-target rust:1.94-bookworm \
  bash -c "cargo test --workspace 2>&1 | grep 'test result' && \
           cargo run --release --example mars_descent 2>&1 | grep -E 'Status|Final cost|τ'"
```

For the full matrix (clippy needs its component added in the image; thumb needs
the target):

```sh
... rust:1.94-bookworm bash -c "\
  rustup component add clippy >/dev/null 2>&1 && \
  rustup target add thumbv7em-none-eabihf >/dev/null 2>&1 && \
  cargo clippy --all-targets -- -D warnings && \
  cargo build --release --target thumbv7em-none-eabihf -p scvx-solver && \
  cargo build --release -p scvx-ffi && \
  cargo build --release -p scvx-ffi --no-default-features --features panic-handler \
    --target thumbv7em-none-eabihf"
```

The last line is the **flight cross-compile of the C-ABI layer**: it produces
`libscvx_ffi.a` (staticlib) + rlib for `thumbv7em-none-eabihf` (cdylib is
auto-dropped for bare-metal — a benign warning). The default host `-p scvx-ffi`
build keeps `std`; the no_std build is what links into MCU firmware.

**Cross-platform determinism result**: the no_std flight crates use `libm`
transcendentals (not platform libm), so the same bits run on Linux x86_64, Windows
MSVC, and the flight target. The pre-audit cross-platform run matched to all
printed digits, which established the property. The current (post free-tf ρ-fix)
Linux-measured figures are `OuterIterCap, τ = 14.029 s, cost = 4.3699e3,
‖ν‖₁ = 3.02e-9`; Windows re-execution is still blocked by the WDAC policy, but the
fix does not touch the `libm`-based determinism mechanism, so the cross-platform
bit-match property is preserved. The Linux FFI
build emits `libscvx_ffi.a` (staticlib), `libscvx_ffi.so` (cdylib), and
`libscvx_ffi.rlib` — the Linux counterparts to the Windows `.lib`/`.dll`.

Plus the C-FFI build:

```sh
cargo build --release -p scvx-ffi
```

Should produce `target/release/scvx_ffi.lib` (static), `scvx_ffi.dll`
(dynamic), and `libscvx_ffi.rlib` (Rust rlib).

---

## Architecture (one screen)

```
PLAYGROUND_SHAPEDCLOUD/
├── Cargo.toml                # workspace, panic="abort", LTO, codegen-units=1
└── crates/
    ├── scvx-core/            # tiny: shared types, params, status enums
    │   └── src/{lib,types,params,status}.rs
    ├── scvx-dynamics/        # 3-DoF model + linearization
    │   └── src/{continuous,jacobian,discretize,forward}.rs
    ├── scvx-ipm/             # SOCP interior-point machinery
    │   └── src/{cone,kkt,mehrotra,socp}.rs
    ├── scvx-solver/          # SCvx outer loop + SOCP assembly + high-level API
    │   ├── src/{api,assemble,precondition,scvx,trust}.rs
    │   ├── tests/{oracle_diff,wcet_benchmarks}.rs
    │   └── examples/mars_descent.rs   # standalone application demo
    └── scvx-ffi/             # C-ABI wrapper for embedding in flight C/C++
        ├── Cargo.toml        # crate-type = [staticlib, cdylib, rlib]
        ├── include/scvx_ffi.h # hand-written C header
        └── src/lib.rs
```

### Crate purposes (read in this order if new)

1. **scvx-core** — type definitions only. `Trajectory<N>`, `PhysicalParams`,
   `ScvxAlgoParams`, `IpmAlgoParams`, `IpmStatus`, `SolverStatus`. No logic.
2. **scvx-dynamics** — `f_continuous`, `df_dx`, `df_du`, `discretize_foh`
   (FOH+RK4 with state ⊕ Φ ⊕ B⁻ ⊕ B⁺ ⊕ s augmented integration),
   `nonlinear_propagate` (ground-truth trajectory for the LM ρ defect).
3. **scvx-ipm** — `cone.rs` (SOC primitives + NT scaling via SymmetricEigen
   per-D), `mehrotra.rs` (toy AHO IPM, regression bench),
   `socp.rs` (generic SOCP solver with both AHO and NT directions),
   `kkt.rs` (standalone Riccati LQR, P2 — not currently wired into IPM).
4. **scvx-solver** — `api.rs` (high-level `solve_powered_descent`
   entrypoint + `PoweredDescentOptions` + workspace dim helpers),
   `assemble.rs` (LCvx + SCvx SOCP assembly, fixed-tf and free-tf),
   `precondition.rs` (per-variable column scaling + per-cone row scaling),
   `scvx.rs` (outer loop with real LM ρ-ratio, free-tf δτ extraction),
   `trust.rs` (placeholder). `examples/mars_descent.rs` is a standalone
   binary demo: `cargo run --release --example mars_descent`.

### Variable layout in the SCvx subproblem (CRITICAL — memorize this)

Per node `k ∈ 0..N-1`, **19 variables**:
```
z_k = [ x_k(7) ⊕ u_k(3) ⊕ σ_k(1) ⊕ ν_k(7) ⊕ w_k(1) ]
```
- `x_k = [r(3), v(3), z=ln(m)]` — state (log-mass parameterization)
- `u_k` — thrust vector (N)
- `σ_k` — thrust-magnitude slack (`‖u‖ ≤ σ`, also `T_min ≤ σ ≤ T_max`)
- `ν_k` — virtual-control slack in the dynamics row block
- `w_k` — L2-epigraph aux: `w_k ≥ ‖ν_k‖₂`

Total dims: `NP = 19N`, `NE = 7N+6`, `NCT = 30N`, `NCONES = 8N`.

**8 cones per node** (total 30 cone-slot dims per node):
1. Thrust mag `(σ_k, u_k) ∈ SOC^4`
2. Pointing `u_z − cos(θ)·σ ∈ ℝ_+` (`SOC^1`)
3. Mass floor `z_k − ln(m_dry) ∈ ℝ_+` (`SOC^1`)
4. Glide slope `(tan(γ)·r_z, r_x, r_y) ∈ SOC^3`
5. `σ_k − T_min ∈ ℝ_+` (`SOC^1`)
6. `T_max − σ_k ∈ ℝ_+` (`SOC^1`)
7. Trust region `(η, x_k − x̄_k, u_k − ū_k) ∈ SOC^{11}`
8. Virtual ctrl L2 `(w_k, ν_k) ∈ SOC^8`

**Helpers in `assemble.rs`**: `x_idx_scvx(k)`, `u_idx_scvx(k)`,
`sigma_idx_scvx(k)`, `nu_idx_scvx(k)`, `w_idx_scvx(k)` give the column
offsets. Use them; never hard-code offsets.

---

## What works (verified by tests)

- **SOC primitives** (det, proj, max-step, sqrt, Jordan product, arrow matrix) — `cone.rs`, 14 tests
- **NT scaling matrix** `W²` and `(W, W⁻¹)` — `cone.rs`, 7 tests including all SCvx cone dims `D ∈ {1,3,4,8,11}`
- **Riccati LQR factor+solve** — `kkt.rs`, 5 tests, dense-LU oracle agreement at 3.5e-15
- **Block-tridiagonal LU (block-Thomas)** — `block_tridiag.rs`, 5 tests, dense-LU oracle agreement at 1e-10 (N=5, B=3) plus linearity + N=1/N=2 hand-computed + clean singular failure
- **SCvx structured reduced-KKT solver** — `reduced_kkt.rs`, 5 tests, dense-LU oracle agreement at 1e-9 (N=3 diagonal-M and block-M) and 1e-8 (N=5 stress) plus N=1 degenerate-boundary clean handling, **plus a full Mehrotra Newton-step equivalence gate (Δx, Δλ, Δs, Δy all match dense path at ≤ 1e-7)** — proves the structured solver is a drop-in replacement for the dense KKT in the IPM
- **Empirical structured-vs-dense speedup** — `wcet_benchmarks.rs::wcet_structured_vs_dense_kkt`: at N=3, structured is 1.89× faster than dense `H.try_inverse()`; at N=5, 4.0×; at N=7, **6.82× faster**. Structured scales 2.49× from N=3 to N=7; dense scales 8.97× (close to the theoretical (7/3)³ ≈ 12.7×). Confirms the O(N·NZ³) vs O((N·NZ)³) prediction empirically
- **Factor/apply split speedup** (Phase 6.7) — `wcet_benchmarks.rs::wcet_factor_apply_split_vs_one_shot`: the structured IPM now factors the reduced KKT once per iter and applies it to both the predictor and corrector RHS, instead of re-factoring twice. Measured speedup: **2.12× at N=3, 2.74× at N=5, 2.01× at N=7**. Combined with the structured-vs-dense gain, the **per-KKT-solve** speedup vs dense at N=7 is **~13.5×**. Equivalence test (`factor_apply_split_matches_one_shot`) confirms factor+apply produces the same `(Δz, Δλ_*)` as the one-shot path to ≤ 1e-12 (effectively machine precision modulo compiler reordering). **NB**: this per-solve number does NOT translate to a full-solve win — see the Phase 6.12 caveat in the TL;DR (fallback erosion + outer-loop divergence make the structured path a wash-on-time / worse-on-cost at N=5; dense stays the default)
- **Free-tf Sherman-Morrison** (Phase 6.8) — `reduced_kkt.rs::factor_reduced_kkt_scvx_block_m_free_tf` + `solve_reduced_kkt_scvx_with_factor_free_tf`. Extends the structured KKT solver to handle the global δτ column via rank-1 update on the block-tridiag Schur: `S_full = S_tridiag + α·u·uᵀ` where `u = a_δτ` (the δτ column of A) and `α = 1/H_δτ`. Standard Sherman-Morrison-Woodbury formula gives `Δλ = S_tridiag⁻¹·rhs − γ·(uᵀ·S_tridiag⁻¹·rhs)·v` where `v = S_tridiag⁻¹·u` (cached in the factor) and `γ = α/(1 + α·uᵀ·v)`. Equivalence test (`free_tf_structured_matches_dense_lu`) confirms `Δz/Δδτ/Δλ` match dense LU to **machine precision** (1.2e-13, 2.4e-14, 1.8e-12). End-to-end test (`scvx_converges_with_structured_solve_free_tf`) confirms the SCvx outer loop converges with **both** `use_structured_solve = true` AND `use_free_tf = true` — making the structured driver usable on the canonical Mars descent problem (which is free-tf by default). Lifts the `!use_free_tf` gate in `solve_scvx`'s dispatch.
- **Live-driver structured IPM** — `structured_socp.rs::solve_socp_structured`, 1 test. A complete AHO Mehrotra IPM driver that uses `solve_reduced_kkt_scvx_block_m` internally instead of dense `H.try_inverse()`. Single-iter dense-vs-structured iterate match at **machine precision** (≤ 1.4e-13). Confirms the structured solver is a drop-in replacement at the per-step level; multi-iter trajectories diverge solely from floating-point roundoff amplification, which is normal for IPM iteration and not a defect
- **Structured driver wired into the SCvx outer loop** — `ScvxAlgoParams::use_structured_solve` flag dispatches `solve_scvx` to `solve_socp_structured` when fixed-tf + AHO. Includes a **graceful fallback to the dense driver** if the structured path's accumulated fp drift causes it to miss the best-feasible snapshot on a given warm-started call. The fallback re-seeds the warm-start from the reference trajectory (with preconditioning re-applied) and re-solves with `solve_socp`; net cost is at most 2× the dense path even when fallback fires, vs the 6-7× theoretical speedup when structured succeeds. Test `scvx_converges_with_structured_solve` confirms the full outer loop converges to within ~1% of the dense path's final cost (4393 vs 4345 on the small-scale Mars demo)
- **NT-direction structured driver** (Phase 6.9) — `structured_socp.rs::solve_socp_structured_nt` + `build_per_cone_nt_blocks`, 1 test. The NT twin of `solve_socp_structured`: passes `W²` as the cone scaling `M` to the **same** scaling-agnostic block-tridiag solver (no `reduced_kkt` changes), with NT RHS (`b_x = −r_x − GᵀW²r_g + GᵀW·r_c_arg`) and NT Δs/Δy recovery (`Δy = −W·r_c_arg + W²·(r_g + G·Δx)`). NT blocks built via the public `soc_nt_w_and_inverse` (Denman-Beavers). Wired into `solve_scvx` dispatch (structured + NT + fixed-tf → structured NT, with **dense-NT fallback**). Equivalence test `structured_nt_matches_dense_nt_one_iter` confirms a one-iter NT step matches dense `solve_socp_nt` to **machine precision** (Δx 3.4e-13, Δs 3.4e-13, Δy 9.1e-13).
- **NT free-tf structured driver** (Phase 6.10 — **completes the dispatch matrix**) — `structured_socp.rs::solve_socp_structured_nt_free_tf`, 1 test. Composes the NT `W²` scaling (Phase 6.9) with the Sherman-Morrison δτ correction (Phase 6.8): the free-tf SMW factor/apply pair is scaling-agnostic, so it takes `W²` as `m_full` and augments the block-tridiag Schur with the rank-1 δτ term transparently. Wired into dispatch (structured + NT + free-tf → structured NT free-tf, dense-NT fallback). Equivalence test `structured_nt_free_tf_matches_dense_nt_one_iter` confirms a one-iter step matches dense NT to **machine precision** (Δx 3.1e-13, Δs 3.1e-13, Δy 3.3e-12, **δτ 3.1e-13**). The structured dispatch matrix is now **complete**: all four cells (AHO/NT × fixed-tf/free-tf) have a machine-precision-verified structured path with a direction-matched dense fallback
- **Toy Mehrotra AHO IPM** — `mehrotra.rs`, 4 tests, closed-form regression
- **Generic SOCP IPM (AHO direction)** — `socp.rs` `solve_socp`, 3 tests
- **Generic SOCP IPM (NT direction)** — `socp.rs` `solve_socp_nt`, 2 tests
- **3-DoF dynamics + analytic Jacobians** — `continuous.rs` + `jacobian.rs`, FD-validated to 2e-9 over 200 random points
- **FOH+RK4 discretization with sensitivities** — `discretize.rs`, 5 tests including linearization identity at reference
- **Nonlinear forward integration** — `forward.rs`, 3 tests including free-fall closed-form match
- **LCvx SOCP assembly** — `assemble.rs::assemble_lcvx_socp`, 2 tests
- **SCvx SOCP assembly** — `assemble.rs::assemble_scvx_socp`
- **SCvx outer loop with real LM ρ-ratio** — `scvx.rs::solve_scvx`, 5 tests including τ-preservation regression, 6-attack red-team, and preconditioning convergence demo
- **Active-drag flight-envelope coverage** — `scvx.rs::scvx_active_drag_path_exercised_and_handled`, 1 test. Runs a full `solve_scvx` with drag ON (`rho=0.02, cd_a=50`) — the FIRST end-to-end exercise of the aerodynamic-drag path (`continuous.rs`/`jacobian.rs`/`discretize.rs` with `rho>0`); every other test uses `rho=0`. The solver makes real progress (cost 1.25e5→5.5e4, drag-induced ‖ν‖ 0.70→0.28 over 5 accepted iters) before the AHO endgame breakdown. Verified to the *graceful-handling* bar: iter-0 drag SOCP solvable, no BadInput/panic, no NaN in the reference. **Finding (documented below): convergence is currently tuned to the Mars no-drag sweet spot — drag/gravity perturbations trigger the documented AHO-endgame fragility.**
- **Per-variable preconditioning** — `precondition.rs`, 7 tests covering: diagonal positivity, scale-table conformance, large-x_init override, column-scaling consistency, warm-start scale/unscale round-trip, scaled vs unscaled optimum agreement, feasibility preservation
- **Oracle agreement** — `oracle_diff.rs`, 3 tests against pre-computed CVXPY/Clarabel + Julia/Clarabel values on 3 canonical SOCPs (toy 1-cone, 2-cone, 4D), all at 3e-5 or better
- **WCET benchmarks** — `wcet_benchmarks.rs`, 6 tests confirming O(N·NX³) Riccati scaling and µs-scale primitive times on x86_64

---

## What works now with full preconditioning (Phase 2 landed)

Both column scaling (P11) and per-cone slack rescaling (Phase 2) are
implemented. On the small-scale Mars descent demo (N=3, r_z=2m,
v_z=-0.1 m/s, m=400 kg, τ=10 s):

| Configuration                       | Outer iters | Final cost | Final ‖ν‖ | IPM behavior              |
|-------------------------------------|-------------|------------|-----------|---------------------------|
| AHO, no precond                     | 4           | NaN        | NaN       | `InnerFailure` at iter 3  |
| AHO + column precond                | 15          | ~7042      | 0.027     | `OuterIterCap`, all clean |
| **AHO + full precond (col + row)**  | **15**      | **~4345**  | **6.7e-13** | **`OuterIterCap`, all clean** |
| NT + column precond                 | 1           | NaN        | NaN       | `InnerFailure` (NT iter ~20) |
| NT + full precond                   | 1           | NaN        | NaN       | `InnerFailure` (NT iter ~14) |

**Headline**: AHO + full preconditioning drops the cost by **~6500×**
from initial (4.88e6 → 4345) and reaches **machine-precision virtual
control** (‖ν‖ = 6.7e-13) at outer iter 4. The cone-row scaling is the
final ingredient that makes the SCvx outer loop's defect-tracking
converge near-exactly.

### Preconditioning math

**Column scaling** (`use_preconditioning`, `precondition::scale_socp_in_place`):
- `x_orig = D · x_scaled` for diagonal `D ≻ 0`
- `c' = D·c`, `A' = A·D`, `G' = G·D`; `b`, `h`, cone descriptors unchanged
- Cone slack `s = h − G·x` is invariant under this transformation
- Dual variables `λ`, `y` invariant

**Cone-row scaling** (`use_cone_row_scaling`, `precondition::scale_cone_rows_in_place`):
- For each cone `c` with positive scale `e_c`:
  - `G_c' = G_c / e_c`, `h_c' = h_c / e_c`
- Cone slack scales: `s_c_normalized = s_c_orig / e_c`
- Cone constraint preserved (SOC is positive-homogeneous degree 1)
- The IPM operates on normalized slacks, so `arrow(s)^{-1}` magnitudes
  are bounded across all cones uniformly

The two scalings commute (independent row/column operations); applied
together they normalize both primal magnitudes AND cone slack
magnitudes.

### Scale choices (in `build_cone_scale_diagonal`)

| #   | Cone                                             | Scale                  |
|-----|--------------------------------------------------|------------------------|
| 1   | Thrust magnitude `(σ, u) ∈ SOC^4`                | `t_max`                |
| 2   | Pointing `u_z − cos(θ)·σ ∈ ℝ_+`                  | `t_max`                |
| 3   | Mass floor `z − ln(m_dry) ∈ ℝ_+`                 | `1`                    |
| 4   | Glide slope `(tan(γ)·r_z, r_x, r_y) ∈ SOC^3`     | `pos_scale`            |
| 5   | T_min `σ − T_min ∈ ℝ_+`                          | `t_max`                |
| 6   | T_max `T_max − σ ∈ ℝ_+`                          | `t_max`                |
| 7   | Trust region `(η, x − x̄, u − ū) ∈ SOC^{11}`      | `trust_eta` (per iter) |
| 8   | Virtual control L2 `(w, ν) ∈ SOC^8`              | `1`                    |

Trust scale **must** be `trust_eta` (NOT `max(trust_eta, pos, thrust)`).
The larger choice balances the dual warm-start better, but empirically
breaks AHO convergence — the cost blows up because the trust-region
constraint gets effectively tightened in scaled coords. Documented in
the function's docstring with a "tradeoff favors small `e_trust`" note.

## What does NOT work yet (the remaining gap)

**NT + full preconditioning** still fails. NT runs ~14 inner iterations
before bailing with `NumericalError`. This is **independent of
preconditioning** (NT + column-only fails at ~20 iters; NT + full
preconditioning at ~14 iters — both are inside the IPM, not the
preconditioner).

**Diagnoses for NT-specific issue**:
1. **Matrix-sqrt of W² accumulates error** over many iterations. The
   current implementation uses eigendecomp-first (for `D ∈ {1,3,4,8,11}`)
   with Denman-Beavers fallback. A closed-form Sturm-1999 SOC `W²`
   computation that's exact for misaligned bars would eliminate this.
2. **Mehrotra centering parameter `σ`** for NT may need tuning. The
   AHO formula carries through unchanged into NT; small differences
   in the relevant residual sizes could mean a different σ choice is
   optimal in NT coords.

**For production use today**: AHO + full preconditioning is the
recommended pairing. It converts the small-scale Mars demo from
`InnerFailure` (no precond) to `OuterIterCap` with 100% inner-IPM
success AND brings ‖ν‖ to machine precision. NT is held back pending
the IPM-side robustness fixes above.

---

## Previously planned task (P11, DONE) — per-variable rescaling

Documented here for future reference. Implementation lives at
`crates/scvx-solver/src/precondition.rs`.

### Scale choices implemented

Derived from `PhysicalParams` and initial state with a `MIN_SCALE = 1.0`
floor (so no diagonal entry degenerates to zero):

| Variable     | Index            | Scale source                                |
|--------------|------------------|---------------------------------------------|
| `r` (3)      | `x_idx_scvx(k)+0..3` | `max(|x_init[0..3]|, 100.0)` — altitude |
| `v` (3)      | `x_idx_scvx(k)+3..6` | `max(|x_init[3..6]|, 10.0)` — velocity  |
| `z = ln(m)`  | `x_idx_scvx(k)+6`    | `1.0` — log-mass naturally O(1)         |
| `u` (3)      | `u_idx_scvx(k)+0..3` | `phys.t_max` — thrust magnitude         |
| `σ`          | `sigma_idx_scvx(k)`  | `phys.t_max` — matches `u` (cone coupling) |
| `ν` (7)      | `nu_idx_scvx(k)+0..7`| per-component match to `x` scale         |
| `w`          | `w_idx_scvx(k)`      | `1.0` — L2 norm of small ν is small      |

**Why u and σ must match**: the thrust-magnitude cone enforces `‖u‖ ≤ σ`.
With per-variable scaling, the cone in scaled coords is preserved iff
`D_u = D_σ`. Same logic for ν per-component matching x.

### Integration point in `solve_scvx`

```rust
// Once per call (right after input validation):
workspace.scale_diag = build_scaling_diagonal::<N, NP>(phys, initial_state);

// Per outer iteration:
// 1. assemble_scvx_socp(...)
// 2. if ipm.use_preconditioning { scale_socp_in_place(&mut prob, &scale_diag); }
// 3. seed_warm_start(reference, &mut ws.x);    // original units
// 4. if ipm.use_preconditioning { scale_warm_start_in_place(&mut ws.x, &scale_diag); }
// 5. solve_socp[_nt](&prob, &params, &mut ws_ipm);
// 6. if ipm.use_preconditioning { unscale_solution(&result.x, &scale_diag, &mut x_unscaled); }
//    else { x_unscaled = result.x; }
// 7. All consumers (virt_l1, step_norm, fuel_cost, extract_candidate, u_cand)
//    read from x_unscaled — never from result.x directly.
```

`cost = workspace.prob.c.dot(&result.x)` stays correct in either coord
system thanks to the dot-product invariance. **Do NOT** unscale before
the cost computation; that would double-scale.

---

## Critical files — recommended reading order

For someone picking up cold, read in this order (rough times noted):

1. **This file (HANDOFF.md)** — 15 min, you're here
2. **`Cargo.toml`** — 1 min, workspace layout
3. **`crates/scvx-core/src/params.rs`** — 3 min, key knobs the IPM cares about (`IpmAlgoParams`, `ScvxAlgoParams`)
4. **`crates/scvx-solver/src/assemble.rs`** — 15 min, **THE** layout reference; understand variable indexing and the 8 per-node cones
5. **`crates/scvx-solver/src/scvx.rs`** — 20 min, the outer loop. Note: real LM-ρ, force-accept iter 0, τ preservation, NT dispatch flag
6. **`crates/scvx-ipm/src/socp.rs`** — 30 min, the inner IPM (both AHO `solve_socp` and NT `solve_socp_nt`)
7. **`crates/scvx-ipm/src/cone.rs`** — 15 min, SOC primitives + matrix-sqrt strategies
8. **`crates/scvx-dynamics/src/discretize.rs`** + **`forward.rs`** — 10 min, the augmented-state RK4 and nonlinear propagator
9. **`crates/scvx-solver/tests/oracle_diff.rs`** — 5 min, what the oracle agreement looks like

---

## Red-team / safety invariants (verified by tests + grep)

| Property                                            | Mechanism                                |
|-----------------------------------------------------|------------------------------------------|
| No `panic!`/`unwrap`/`expect` in production code    | Verified by repo-wide grep              |
| No `alloc`/`Vec`/`Box`/`Rc`/`Arc`/`String`          | `#![no_std]` on all flight crates       |
| No `unsafe` anywhere                                | `#![forbid(unsafe_code)]` on 3/4, scvx-ipm has zero unsafe in source |
| No NaN propagation from IPM to caller               | `numerical_exit` scrubs NaN to 0.0      |
| No clamp-panic on bad params                        | Entry validation in `solve_scvx` (6-attack regression test) |
| Bounded WCET (per IPM iter)                         | Compile-time `IPM_HARD_MAX_ITERS = 64` cap; loop runs `min(max_iters, 64)` |
| Bounded outer loop iterations                       | `min(algo.max_outer_iters, MAX_OUTER)`  |
| τ preserved across SCvx iterations                  | Explicit `preserved_tau` capture; regression test |
| Iterate magnitude bounded                           | `>1e50` check in IPM → `numerical_exit` |
| Cone interior maintained                            | Post-step `all_cones_interior` check    |
| NaN in Newton step rejected                         | `step_finite` check both predictor + corrector |
| Determinism                                         | f64 IEEE 754 + libm transcendentals, no threading, no RNG |
| Thumb cross-compile clean                           | `cargo build --release --target thumbv7em-none-eabihf` |
| Zero panic-message strings in release rlibs        | Verified by `llvm-objdump --string-dump=.rdata`/`rodata` in earlier P10 audit |

### Authorized `#[allow(...)]` (11 total, all justified)

Enumerate with `grep -rn '#!\?\[allow(' crates/*/src` (line numbers drift, so
they're omitted here — the categories are what matter):

- **5× `clippy::too_many_arguments`** — the wide solver entrypoints that
  genuinely need every argument: `assemble_scvx_socp`, `solve_scvx`,
  `solve_powered_descent`, `record_iter`, and the free-tf reduced-KKT factor.
- **3× `clippy::manual_clamp`** — `clip01` in `socp.rs`/`mehrotra.rs` and the
  σ-clamp in `structured_socp.rs`. Deliberate: `f64::clamp` *propagates* NaN,
  and these paths explicitly map NaN→0 for IPM safety, so the manual form is
  the correct (not lint-pacifying) choice.
- **1× `clippy::type_complexity`** — a structured-driver 4-tuple step type.
- **1× `clippy::identity_op`** (module-level, `assemble.rs`) — keeps the `+ 0`
  cone-block offsets column-aligned for readability.
- **1× `dead_code`** — a `dbg_trajectory` printf-debug helper in a test module.

No `#[allow]` suppresses a correctness or safety lint; all are readability /
arg-count / intentional-NaN-handling.

### What an attacker / bad-input source CANNOT do

Verified by `scvx::tests::red_team_input_validation` (6 attack scenarios):
- Bypass clamp-panic via `trust_eta_min > trust_eta_max`
- Inject NaN via `trust_eta0`, `virt_weight`, `initial_state`, `terminal.r/v`
- Run forever via `max_outer_iters = u32::MAX`
- Cause divide-by-zero (predictively guarded everywhere via `> 1e-12` checks)
- Smuggle `alloc` into a flight crate
- Smuggle `unsafe` into a flight crate (forbidden by attr in 3/4, zero in 4th)

---

## Common pitfalls (lessons learned this project)

### 1. Stack overflow during `Box::default()` of `ScvxWorkspace`

The workspace contains many large const-generic matrices. Default-
constructing on the stack then moving to heap overflows the default
2 MB test thread stack at `N ≥ 3`. **Solution**: tests use
`run_in_big_stack` which spawns a 32 MB thread. In production, the
workspace lives in a static or pre-allocated arena.

### 2. nalgebra const-generic bounds

`SymmetricEigen::try_new` requires `Const<D>: DimSub<U1>`, and
`SMatrix::determinant` requires `Const<D>: ToTypenum`. Both are
implemented per-fixed-D in nalgebra but NOT for generic `Const<D>`.
**Workaround**: macros that emit per-D specialized helpers (see
`emit_nt_w_specialized!` in `socp.rs`). The match dispatcher uses fixed
literals.

### 3. `f64::clamp` propagates NaN

`a.clamp(0.0, 1.0)` returns NaN if `a` is NaN. For step lengths, we
explicitly want NaN → 0 (reject step). Our `clip01` checks `is_nan` first.
**Don't let clippy convince you to use `f64::clamp` for step lengths.**

### 4. `f64::powi` is `std`-only

The `.powi(3)` method is in `std`, not `core`. In no_std, use `r * r * r`
or `libm::pow(r, 3.0)`. We use the multiplication form throughout.

### 5. `.ln()` on float literal needs explicit type

`let m = 1000.0; m.ln()` fails to compile in `#[cfg(test)]` modules
because the type is ambiguous. Use `let m: f64 = 1000.0; m.ln()` or
`(1000.0_f64).ln()`.

### 6. `extern crate std` in `#[cfg(test)]` modules

Required to use `std::println!` etc. inside test modules of `#![no_std]`
crates. Put `extern crate std;` as the first item inside the test mod.

### 7. The toy IPM uses `solve_toy_socp` (hardcoded NP=3, NE=1)

It's the regression bench from P1c. **Don't change it.** It's verified
against closed-form. The generic IPM (`solve_socp`) is what real callers use.

### 8. `Trajectory::default().tau = 1.0`

`extract_candidate` doesn't touch `tau`. Without explicit preservation,
`reference = candidate.clone()` would set `reference.tau = 1.0` and
silently change the time scale. Already fixed; **don't accidentally undo**
the `preserved_tau` capture in `solve_scvx`.

### 9. NT inner IPM and SCvx demo

`use_nt_scaling: true` in `IpmAlgoParams` dispatches to `solve_socp_nt`,
but as of this handoff it still fails on the SCvx subproblem even WITH
preconditioning (~20 iters then NumericalError, vs ~16 without
preconditioning). Diagnosis: the fixed `H_REGULARIZATION = 1e-8` is
negligible relative to the preconditioned scaled-Hessian magnitudes.
Production demo uses **AHO + preconditioning** which now converges
cleanly. NT is blocked on adaptive regularization (next planned task).
The eigendecomp-based matrix sqrt is the default for `D ∈ {1,3,4,8,11}`
with plain Denman-Beavers as fallback.

### 10. Cone match dispatcher limited to SCvx dims

`build_nt_block_for_cone` only handles `D ∈ {1,3,4,8,11}`. If you add a
new cone with a different dim, add a case there or NT will silently
return `false` (caller bails with NumericalError). The macro
`emit_nt_w_specialized!` makes adding new dims a one-liner.

---

## Tooling already set up

### Rust toolchain (Windows MSVC)

- Rust 1.94 stable (rustc 1.94 at last verification)
- `thumbv7em-none-eabihf` target installed (for cross-compile gate)
- `llvm-tools` component for `llvm-nm`, `llvm-objdump` (P10 symbol audit)

### External oracles (for P8/P9 validation tests)

Pre-existing — DO NOT re-create from scratch:

- `tools/py-oracle/` — Python venv with CVXPY+Clarabel. Run: `tools/py-oracle/Scripts/python.exe tools/py-oracle/solve_canonical.py`
- `tools/jl-oracle/` — Julia env with Clarabel.jl. Run: `julia --project=tools/jl-oracle tools/jl-oracle/solve_canonical.jl`
- `crates/scvx-solver/tests/oracle_diff.rs` — Reference values from these
  oracles baked in as constants; 3 tests verify Rust IPM agreement.

If you regenerate oracle reference values, update the constants in
`oracle_diff.rs` (not the oracles).

### Cargo profiles

- `[profile.release]`: `panic = "abort"`, `lto = "fat"`, `codegen-units = 1`,
  `opt-level = 3`, `overflow-checks = false`
- `[profile.dev]`: `panic = "abort"`, `opt-level = 1`,
  `overflow-checks = true`

`panic = "abort"` is intentional and **must not be changed** without
discussion (it's part of the bounded-WCET story).

---

## Code conventions (followed throughout)

- **Module-level docstring** (`//!`) at the top of every file explains the
  module's purpose, key types, and any relevant references (e.g., paper
  citations for algorithms).
- **Function docstrings** (`///`) describe the contract, including
  preconditions, postconditions, failure modes, and the math derivation
  for non-obvious computations.
- **Comments inside functions** explain *why*, not *what*. The *what*
  should be readable from the code.
- **Visual alignment** with spaces between identifiers and colons:
  ```
  let arrow_s_inv: SMatrix<f64, NCT, NCT> = ...;
  let m_scale:     SMatrix<f64, NCT, NCT> = ...;
  ```
  This is intentional; clippy lint `clippy::identity_op` was suppressed in
  `assemble.rs` to allow `+ 0` for alignment in cone-block construction.
- **No emojis in source files** (the request was to avoid them unless
  explicitly asked). Doc tables and the like are fine.
- **`#![forbid(unsafe_code)]` on three of four crates**; `scvx-ipm` uses
  `#![deny(unsafe_op_in_unsafe_fn)]` to preserve the future option of
  adding audited `unsafe` to the Riccati hot path (currently zero unsafe).
- **Test names** describe the property being verified, not the function
  being tested. E.g. `nt_w_squared_equals_w_times_w`,
  `tau_sensitivity_matches_central_difference`,
  `red_team_input_validation`.

---

## Open todo (in priority order)

1. **Phase 6 Riccati substitution — FULLY LANDED, end-to-end.** All five layers in place:
   - **Layer 1**: `block_tridiag.rs` — generic block-tridiagonal LU (5 tests, dense-LU agreement at 1e-10).
   - **Layer 2**: `reduced_kkt.rs` — SCvx-aware reduced-KKT solver, both diagonal-M and block-M variants (5 tests, dense-LU agreement at 1e-9, plus full Newton-step Δx/Δλ/Δs/Δy equivalence at 1e-7).
   - **Layer 3**: `structured_socp.rs::solve_socp_structured` — complete AHO Mehrotra IPM driver using the structured solver internally (1 test, single-iter live-driver iterate match at **machine precision ≤ 1.4e-13**).
   - **Layer 4**: empirical speedup measured at **6.82× at N=7** with near-linear structured scaling vs cubic dense.
   - **Layer 5**: **Wired into the SCvx outer loop** via `ScvxAlgoParams::use_structured_solve`. When the flag is set (and `!use_free_tf && !use_nt_scaling`), the outer loop dispatches to `solve_socp_structured` per iteration. Includes a **graceful fallback to dense** if structured drift causes a miss: the fallback re-seeds the warm-start from the reference (with preconditioning re-applied) and retries with `solve_socp`. End-to-end test (`scvx_converges_with_structured_solve`) confirms the full outer loop converges with the new flag enabled.

   **Open follow-ups** (none blocking; all extensions, not gaps):
   - **~~NT direction~~ LANDED (Phase 6.9)**: `solve_socp_structured_nt` mirrors the AHO structured driver with NT scaling — passes `W²` as `m_full` to the same scaling-agnostic block-tridiag solver, builds NT blocks via `build_per_cone_nt_blocks` (Denman-Beavers W), uses NT RHS/recovery formulas. Wired into `solve_scvx` dispatch (structured + NT + fixed-tf → `solve_socp_structured_nt`, dense-NT fallback). Equivalence test `structured_nt_matches_dense_nt_one_iter` confirms a one-iter NT step matches dense `solve_socp_nt` to machine precision (Δx 3.4e-13, Δy 9.1e-13).
   - **~~NT free-tf structured~~ LANDED (Phase 6.10)**: `solve_socp_structured_nt_free_tf` composes the NT W² scaling with the Sherman-Morrison δτ correction (the SMW factor/apply is scaling-agnostic, so it takes W² as `m_full`). Wired into dispatch with dense-NT fallback. Equivalence test confirms machine-precision match including δτ. **The structured dispatch matrix is now complete** — all four AHO/NT × fixed-tf/free-tf cells have verified structured paths.
   - **~~Free-tf via Sherman-Morrison~~ LANDED (Phase 6.8)**: `factor_reduced_kkt_scvx_block_m_free_tf` + `solve_reduced_kkt_scvx_with_factor_free_tf` handle the global δτ column via rank-1 update on the block-tridiag Schur. End-to-end test confirms the SCvx outer loop converges with `use_free_tf = true` AND `use_structured_solve = true`.
   - **~~Single-factor reuse~~ LANDED (Phase 6.7)**: `factor_reduced_kkt_scvx_block_m` + `solve_reduced_kkt_scvx_with_factor` split, structured IPM factors once + applies twice per iter, measured 2-2.7× speedup. Total speedup vs dense at N=7 is now ~13.5× (6.82× structured-vs-dense × 2.01× factor-reuse).
   - **Reduce fallback frequency** (Phase 6.11 diagnosed): root cause is
     AHO-endgame ill-conditioning in the structured factorization, NOT
     snapshot-discard or slow convergence (both ruled out experimentally).
     Snapshot-preferring bail landed; `structured_fallbacks` telemetry
     counter added. Candidate fix: iterative refinement on the structured
     KKT solve. See TL;DR open-work item 2 for the full analysis.
   - **`mars_descent` example switching to structured**: currently the example uses `use_structured_solve = false` (default). With Phase 6.8 the free-tf structured path works end-to-end, so the example could be updated to opt in and showcase the speedup. Pure cosmetic — no algorithm change needed.

2. **NT inner-IPM further robustness**. Higham-scaled DB landed (per-D
   determinant scaling, in `socp.rs::emit_nt_w_specialized!`). NT
   still doesn't fully converge on the small Mars demo — the failure
   is NOT in the matrix sqrt (test `nt_full_precond_fails_gracefully`
   pins the current behavior) but somewhere in the NT-specific code:
   scaling formula `μ_k = (|det Z_k|/|det Y_k|)^(1/(2D))` needs
   `SMatrix::determinant`, which requires `Const<D>: ToTypenum`.
   nalgebra implements this per fixed D (works for our SCvx cone dims
   `D ∈ {1,3,4,8,11}`). The implementation goes in
   `socp.rs::emit_nt_w_specialized!` as a per-D specialization — the
   generic `soc_nt_w_and_inverse` in `cone.rs` would stay unchanged as
   the fallback. ~150 LOC, mostly mechanical macro emission.
   
   **Cheap surrogate tried and failed**: Frobenius-norm balancing
   (γ = sqrt(‖Z‖_F/‖Y‖_F)) does NOT preserve the DB fixed point because
   ‖W‖_F ≠ ‖W⁻¹‖_F in general — μ doesn't go to 1 at convergence and the
   iteration drifts. Reverted. Need the proper det-based Higham.

4. **(Removed — superseded by item 1 above.)** Originally this was
   the "wire Riccati" task; the primitives are now landed (Phase 6) and
   the remaining work is captured under item 1 as the driver wiring.

5. **Scale-up beyond N=10 — DIAGNOSED + SOLVED via trust tuning (Phase 11).**
   The old framing ("N=10 bails, inner IPM gives up") was **stale**. An N-sweep
   characterization (`diag_larger_n_convergence_sweep`, N∈{5,8,10,12,15,20} on a
   100 m / 800 kg descent) showed:
   - The **inner IPM is healthy at every N** (all `BestFeasible`, no bail) —
     the preconditioning + corrector work fixed that. N is NOT the bottleneck.
   - The **outer SCvx loop** stalled identically at all N: the dynamics defect
     ‖ν‖ froze at ~0.1–0.3 (six orders above `conv_tol_virt`) and the trust
     radius collapsed monotonically.
   - **Root cause**: ρ = actual/predicted compares the LINEARIZED-dynamics cost
     to the true NONLINEAR re-propagation; that gap caps the achievable ρ at
     ≈0.1–0.2, so with the textbook `rho_grow=0.7` the trust can never grow and
     a single early hard step collapses it. `virt_weight` is NOT the lever
     (1e7 breaks the inner IPM, 1e9 leaves ‖ν‖≈0.27 — measured in
     `diag_n10_virt_weight_trust_tuning`).
   - **Fix**: set the trust thresholds to the achievable ρ — `rho_shrink=0.05,
     rho_grow=0.1`. This drove the N=10 defect from **2.1e-1 → 6.2e-11**, and
     the whole sweep N=8..20 to **~1e-10 to 1e-11** (machine-precision dynamics
     feasibility). Demonstrated + asserted in
     `scvx_converges_larger_n_with_tuned_trust` (N=10, min‖ν‖ < 1e-6).
   - **NOT made the default**: those thresholds destabilize some small-scale /
     structured configs (the aggressive trust growth overshoots; 4 small-N
     tests hit `InnerFailure`). The conservative `0.25/0.7` default stays;
     larger/flight-scale problems set `rho_shrink≈0.05, rho_grow≈0.1`
     explicitly (documented on `ScvxAlgoParams::default` in `params.rs`).
   - **Remaining edge**: N=5 on the 100 m scenario still stalls (‖ν‖≈0.5) even
     with the tuned thresholds — its coarse grid (5 s steps) gives the largest
     per-step linearization gap, so ρ never clears even 0.05 and the trust
     still collapses. A genuine, explainable limit of the coarsest grids; the
     finer grids (N≥8) converge. FFI exposes N≤20; with tuned thresholds N≥8
     converge, N≤5/coarse is best-effort.

---

## Phase 10 — v1 audit & packaging (LANDED)

A pre-ship hardening pass: a 4-agent parallel red-team audit (one per crate
group: core+dynamics, ipm, solver, ffi), then triage + fixes, FFI surface
expansion, and an integration guide.

**Critical (fixed + regression-tested).** `use_free_tf` flag ↔ const-generic
dimension mismatch. The per-N FFI entrypoints bake fixed/free-tf dims at compile
time, but the solver branches on the *runtime* `options.use_free_tf`. Since
`scvx_options_default` sets `use_free_tf = 1`, the most natural usage —
`scvx_options_default(&o); scvx_solve_n3(...)` *without clearing the flag* —
paired free-tf-flag with fixed-tf dims and wrote the δτ slot out of bounds
(`precondition.rs` / `assemble.rs`) → panic/abort (UB if linked `panic=unwind`).
Fixed at **three layers**: (1) `api::solve_powered_descent` upgraded its dead
`debug_assert_eq!` dim checks to live `BadInput` returns; (2) `scvx::solve_scvx`
got a runtime dim/flag guard for direct Rust callers; (3) the FFI macro rejects
`options.use_free_tf != FREE_TF` at the boundary. Regression test
`flag_entrypoint_mismatch_rejected_not_aborted` pins both directions → `BadInput`.

**High (fixed).** (a) Unguarded `Isp·g0` divisor in the mass-flow dynamics —
zero/negative injected NaN/Inf into the whole linearization silently; now
validated `> 0` in both `solve_scvx` and the FFI boundary (test
`zero_isp_or_g0_rejected`). (b) Warm-start finiteness only checked element 0 of
each cone slice; added an explicit full-slice finiteness gate in
`init_per_cone_warm_start` (was incidentally safe via NaN-poisoned `margin`, now
explicit + covers the finite-but-overflowing-magnitude path too).

**Medium (fixed).** Unvalidated `pub` cone descriptors could panic on a
hand-built/FFI `SocpProblem` (`dim==0` or `offset+dim>NCT` → OOB slice). Added
`cones_valid()` at the entry of both `solve_socp` and `solve_socp_nt`
(returns a zeroed `NumericalError` result rather than panicking).

**Low (doc accuracy).** Corrected over-claims the audit flagged: the AHO
`StepFactors.h_inv` doc (stores `H⁻¹`, was labelled `H`); the FFI module doc
(cited a nonexistent `SolverStatus::as_u32`); `find_stage_for_cone`'s exact-`==0.0`
invariant (now correctly documented as "zero-ness preserved under the upstream
diagonal preconditioners," not "no fp arithmetic"); and the structured
`numerical_exit` parity comment (it tracks a *local* snapshot, cannot read
`ws.best_*`). Two audit items were intentionally left as **documented
preconditions** (not hot-path guards, per the auditor's own advice): caller must
pass a finite reference trajectory, and `rk4_substeps > 0`.

**FFI surface expansion.** Node counts N ∈ {3,5,8,10,12,15,20}, each fixed- and
free-tf. Made DRY with macros on both sides — `emit_traj_ty!` + `emit_ffi_per_n!`
(Rust) and `SCVX_DEFINE_TRAJ` + `SCVX_DECLARE_N` (C). Header verified to compile
clean as **C99 and C++11**; a pure-C example (from the guide) links against the
cdylib and runs. The all-N release staticlib is ~27 MB (a flight build keeps one
N — instructions in the guide). Smoke test `ffi_solve_n8_n10_run_and_stay_finite`.

**Flight-integration guide** at `crates/scvx-ffi/INTEGRATION.md`: building (host
+ thumb no_std + panic-handler), the caller-allocates memory model + alignment,
the fixed/free-tf flag contract, the full input-validation / status-code tables,
determinism + WCET notes, honest validation scope, and a complete linked C
example.

**Verification:** 124 tests, clippy `-D warnings` clean, thumb (solver + no_std
FFI), host FFI, `mars_descent` deterministic at `4.3699e3`, C99/C++11 header
compile, C example link+run — all green in the Docker clean room.

---

## Phase 12 — whole-project alignment audit (LANDED)

A from-scratch, three-lens parallel audit (vision/discipline, completeness,
doc-consistency) re-grounding the whole project against the original
"research-grade flight-shaped" vision. **Verdict: ALIGNED.**

- **Discipline invariants hold on every production path, mechanically + agent
  verified**: `#![no_std]`, zero heap (confirmed by the bare-metal no_std
  cross-compile linking), zero panics outside `#[cfg(test)]` (all `unwrap`/
  `expect`/`panic!` are test-only; fallible ops use `?`/status enums; inputs
  validated before indexing/clamping), **zero data-dependent unbounded loops**
  (every algorithmic loop is `for _ in 0..bound`; the only `while`/`loop`
  tokens are two comments + the FFI no_std panic-handler spin), `unsafe` ONLY
  in scvx-ffi (the algorithm crates are `#![forbid(unsafe_code)]` /
  zero-unsafe `deny`), `libm` transcendentals, const-generic sizing.
- **Honesty check passed**: the load-bearing claims ("no panic on any solver
  path", "machine precision", "bounded WCET") are backed by code; the docs
  candidly flag the real open frontier rather than over-claiming.
- **Completeness ~90%**: every named component exists and is wired end-to-end;
  no `todo!`/stubs/placeholder returns. The remaining 10% is *algorithmic*
  (NT flight-scale convergence; structured-path end-to-end speedup), correctly
  documented as open — not missing features.
- **Code↔code is on the same page**: dispatch matrix complete + correctly
  fallback-wired; FFI header matches all 28 entrypoints + 5 `repr(C)` types
  across all 7 N; features/cfg consistent; status codes agree on all surfaces.

**Fixes landed in this audit (all maintainability / honesty, no behavior
change):**
- **Single-source layout dims (safety hygiene, FFI-sizing path).** `api.rs`
  `workspace_{np,ne,nct,ncones}` and `precondition.rs` (incl. the δτ index)
  now derive from the authoritative `assemble.rs` constants/const-fns instead
  of bare `19/30/8/7N+6` literals — so the FFI workspace-sizing path can never
  silently drift from the actual layout. (`scvx.rs`'s guard already did this.)
- **Stale "future lift" module docs corrected** in `structured_socp.rs`,
  `reduced_kkt.rs`, and `scvx-ipm/socp.rs` — they claimed NT / free-tf /
  factor-split were "not done yet" when all are landed. Also fixed the
  `params.rs` `use_structured_solve` docstring (claimed structured requires
  fixed-tf AHO — the 4-cell matrix is complete), the FFI "N∈{3,5}" comment
  (now 7 N), and the `mars_descent` example docstring (said N=5/100m; the code
  is N=3/2m). Corrected the `#[allow]` inventory (7→11, category-based).

**Net: the project is internally consistent, honestly documented, and verifiably
flight-grade-disciplined. The only substantive gap is the algorithmic frontier,
which is the genuine forward work (see below).**

### Honest forward path (ranked)

1. **NT flight-scale convergence** (open item #1) — the deepest open problem.
   **Phase 22 narrowed it decisively**: the exact closed-form SOC NT scaling
   (`soc_nt_scaling_exact`, vanishing-cone-stable, landed) does NOT fix it, so
   **per-cone scaling is now exhausted as a lever** (corrector + IR + Higham-DB
   + exact-scaling all tried). The remaining barrier is the **global NT Newton
   direction / centering** on the imbalanced vanishing-cone structure — a
   wide-neighborhood / per-cone-balanced centering scheme, genuine IPM research.
   AHO is the production default and works.
2. **Larger-N as the default path** — the tuned trust thresholds
   (`rho_shrink=0.05, rho_grow=0.1`) make N≥8 converge to ‖ν‖~1e-10 but
   destabilize small-N, so they're explicit per-mission tuning today. A
   scale-adaptive trust rule (auto-pick thresholds from problem scale / observed
   ρ) would make one default work across scales.
3. **Structured-path end-to-end win** (open item #2) — blocked by the same
   AHO-endgame fallback erosion; tied to #1. Until then dense is the default.
4. **Coverage hardening** — **mostly DONE (Phase 21)**: an external-oracle test
   now validates AHO (fixed-tf + free-tf) against CVXPY/Clarabel + Julia/Clarabel
   on the real 57/58-var assembled SCvx subproblem (not just the toy SOCP).
   Remaining: an NT external-oracle gate is blocked until NT converges on the
   flight subproblem (item #1); promote more outer-loop ‖ν‖ assertions into CI.
5. **Optional cleanups** — delete or clearly quarantine the vestigial
   test-only paths (`kkt.rs` Riccati, diagonal-M reduced-KKT, `assemble_lcvx_socp`,
   empty `trust.rs`) to shrink the audit surface; make `RB_MAX` a const generic
   if structured N≥64 is ever needed.

---

## Phase 13 — post-interruption deep audit + free-tf ρ fix (LANDED)

A recovery/verification pass after a service interruption, then a fresh
**correctness bug-hunt** (distinct lens from Phase 12's alignment audit).

- **Integrity: clean.** Full tests (124) + clippy + thumb (solver+FFI) + example
  all green ⇒ the interruption left no half-applied edits; state consistent.
- **Numerics: no bugs.** A deep agent pass independently re-derived the four
  highest-risk kernels (NT Newton step, NT corrector, free-tf Sherman-Morrison,
  `soc_max_step`) against Python oracles — all match to machine precision; the
  Jacobians and cone math verified analytically.
- **Found + fixed one genuine HIGH bug** (orchestration): the **free-tf ρ-ratio
  propagated the nonlinear truth-check at `τ_ref` instead of the candidate's
  actual duration `τ_ref + δτ`**. With δτ≠0 the defect conflated linearization
  error with a spurious time-of-flight mismatch, biasing the free-tf trust
  adapt/accept. Fixed in `scvx.rs` (`solve_scvx`) by computing the candidate
  duration `prop_tau` once, using it for both the ρ propagation and the accepted
  `candidate.tau`. It did NOT break convergence before (the convergence test
  keys off the SOCP's own ν, not this defect) — which is why it slipped past
  prior audits — but the honest ρ measurably **improves** the free-tf example:
  ‖ν‖₁ 2.58e-7 → **3.02e-9** (~85× more feasible), cost 4.3916e3 → **4.3699e3**,
  τ 13.846s → 14.029s. All 124 tests still pass; HANDOFF example figures updated.

---

## Phase 14 — scale-adaptive trust (forward item #1, LANDED)

Makes larger-N converge **by default** without the small-N destabilization that
blocked making the lenient thresholds the global default.

- **New flag** `ScvxAlgoParams::use_adaptive_trust` (default **true**) + workspace
  state `rho_ceiling` / `adaptive_seeded`.
- **Mechanism**: the merit ρ = actual/predicted is capped by the linearization
  gap (≈0.1–0.2 at flight scale), so the textbook fixed thresholds (0.25/0.7)
  never let the trust grow and it collapses. Adaptive trust tracks a running
  achievable-ρ "ceiling": **seeded from the first accepted step's ρ** (instant
  regime detection — a slow EMA-from-1.0 lets the trust collapse before it
  relaxes), then EMA-smoothed. The effective grow/shrink thresholds are
  fractions of the ceiling, **gated to relax ONLY when the ceiling is below the
  conservative `rho_shrink`** (the genuine collapse regime). Relax *shrink* to
  stop collapse; keep *grow* ≥ 0.1 to stop overshoot (a lower grow over-grows
  into the regime where the linearization catastrophically fails).
- **Result**: N=10/100 m converges from the DEFAULT (0.25/0.7) config —
  **min ‖ν‖ = 4.4e-10** (was stuck at 0.21). The trace shows *why* it works: as
  the relaxed grow lets the trust grow (50→200), the steps become productive and
  ρ rises (0.06→0.25→0.65→0.97), the gate auto-closes, and the defect closes.
  Asserted in `scvx_converges_larger_n_adaptive_trust`.
- **No regression**: well-conditioned small problems keep ρ above `rho_shrink`,
  so the gate stays closed and the thresholds are the conservative defaults —
  all 124 tests green, clippy clean, thumb (solver+FFI), example deterministic.
  Self-audited for no-panic / no-alloc / bounded-WCET / valid threshold band /
  determinism / `use_adaptive_trust=false` ⇒ exact prior behavior.
- **Follow-up**: not yet exposed in the C `COptions` (FFI uses the default-on);
  expose it there if flight integrators need opt-out.

---

## Phase 15 — NT flight-scale root cause PINPOINTED (forward item #2)

A non-invasive per-cone analysis (`structured_socp::tests::diag_nt_endgame_per_cone`,
`#[ignore]`d) swept `solve_socp_nt` over increasing `max_iters` on a flight-scale
subproblem and read the per-cone complementarity + interiority margins from the
live iterate. **The breakdown is driven specifically by the virtual-control
(ν) SOC⁸ cones**, and the finding is *structural*, not a tuning gap:

- The SCvx penalty drives the dynamics defect ν→0 by design, so the
  virtual-control SOC⁸ slacks ride onto their cone boundary (measured
  interiority margin collapses to ~3e-11 — a cone sitting *on* its boundary).
- That makes the per-cone complementarity spread enormous (~10¹¹: thrust/trust
  cones ≈3.5e7 vs virtual-control ≈1e-4). NT's geometric-mean scaling `W` blows
  up for the near-boundary cone, and that one ill-conditioned block **poisons
  the global Newton step** — μ stalls at ~1.5e6 and never decreases (raw NT,
  status IterCap), or `W`→non-finite under preconditioning (status
  NumericalError, the production symptom).
- **AHO's arrow scaling is robust to vanishing cones** — which is precisely why
  AHO converges on the identical problem and is the production default.

**Conclusion (honest):** this refines the earlier "linearization degeneracy"
note into a concrete structural mismatch — **NT's scaling cannot handle cones
that vanish at the optimum**, which the SCvx virtual-control relaxation always
produces. A real NT fix would need SDPT3-level per-cone handling of vanishing
cones (per-cone scaling regularization / a wide-neighborhood scheme that keeps
the relaxation cones off their boundary), a substantial IPM-research effort. The
corrector + IR attempts (Phases 9/13) and now this analysis exhaust the
cheap/tuning avenues. **Recommendation: keep AHO as the validated production
direction; NT stays opt-in with graceful AHO fallback.** No production code
changed this phase — the deliverable is the diagnosis + the reproducible probe.

---

## Phase 16 — coverage hardening (forward item #3, LANDED)

Strengthened the test suite to *assert* what was previously only smoke-tested or
printed in traces. All additions are CI-runnable with no external solver at test
time (the external oracle values are baked-in from offline CVXPY/Julia runs).

- **NT validated against the external oracle** (`oracle_diff.rs`): the 3 baked-in
  CVXPY/Julia toy SOCPs now check **both** AHO and NT — NT matches to ≤1.2e-4
  (well under the 1e-3 tol), confirming NT is *correct* on well-conditioned
  problems (it only fails on the flight subproblem's vanishing cones, Phase 15).
- **Self-consistent KKT-optimality oracle** (`assert_kkt_optimal`): every solve
  (AHO + NT) is verified to satisfy the SOCP KKT conditions — primal feasibility
  `A·x=b`/`G·x+s=h`, dual stationarity `c+Aᵀλ+Gᵀy=0`, complementarity `s·y≈0`,
  and cone membership — to ~1e-5–1e-13. This proves the solver returns a TRUE
  optimum, not just an `Optimal` *label*; no external reference needed.
- **Convergence-quality ‖ν‖ assertions** promoted from `eprintln` traces to CI:
  tight where convergence is real (`scvx_converges_larger_n_adaptive_trust` <1e-6;
  free-tf structured <1e-3), and an honest floor-guard where it is not.

- **Finding surfaced by the hardening (honest):** the fixed-tf 2 m / τ=10 case
  (`scvx_converges_on_small_problem`) reaches only ‖ν‖≈6e-2 under the DEFAULT
  conservative thresholds — NOT tight convergence. Its ρ stays above `rho_shrink`
  so the adaptive-trust **gate keeps it conservative**; the un-gated adaptive
  reached ~1e-9 here but destabilized other small-N configs (the Phase-14
  regression), so the gate is the safe compromise. The status-only assertion had
  masked this. The test now carries an honest floor-guard (<1e-1) + a comment;
  tight convergence is demonstrated by the larger-N / free-tf tests.
  **Item-#1 follow-up**: a smarter gate (or per-cone-balanced centering) could
  recover tight convergence for this case without the over-grow that breaks the
  api fixed-tf config — a real but bounded future improvement.

All changes are TEST-ONLY (no production code touched). 124 tests; clippy clean;
thumb (solver+FFI); example deterministic.

---

## Phase 17 — convergence-envelope widening (forward item #3, LANDED)

Makes the AHO **production** path converge on **active-drag** and **non-Mars-
gravity (lunar)** descents — the two regimes the prior envelope explicitly
excluded (TL;DR open item #3) — without touching the validated Mars no-drag path.

- **Root cause, re-confirmed from a live trace (not assumed).** On the
  active-drag N=5 case the inner AHO IPM hits its endgame on ONE over-aggressive
  subproblem (the trust had run to `eta_max`, warm-started from a far
  candidate); it returns `IterCap`/`NumericalError` with no best-feasible
  snapshot, and the outer loop **aborted outright** (`return InnerFailure`),
  discarding the entire remaining outer-iteration budget. That abort — not the
  single hard subproblem — is what capped the envelope. (Confirmed: status was
  `IterCap`/`best_valid=false`, not a NaN leak.)
- **Fix — one site, `scvx.rs::solve_scvx` inner-failure handler.** The standard
  trust-region response to an unsolvable subproblem: on `!inner_ok`, **shrink
  the trust radius (`trust_eta /= trust_alpha`) and re-solve the SAME outer
  iterate** (the reference is unchanged ⇒ a tighter, better-conditioned,
  smaller-linearization-gap subproblem), instead of aborting. Give up
  (`InnerFailure`) only once the trust has collapsed to its floor and the
  subproblem is STILL unsolvable. Bounded by the geometric shrink-to-floor and
  the outer-iteration cap; no new alloc/unsafe/panic; no_std-clean.
- **Measured** (N=5, 100 m / −10 m/s / 800 kg):

  | regime | before (abort) | after (retry) |
  |--------|----------------|---------------|
  | active drag (`rho=0.02, cd_a=50`), AHO + column precond | `InnerFailure` @ outer 3, ‖ν‖ ≈ 0.42 | min ‖ν‖ = **1.58e-10** |
  | lunar (`g=−1.62`), AHO + column precond + cone-row scaling | `InnerFailure` @ outer 3, ‖ν‖ ≈ 0.71 | min ‖ν‖ = **3.45e-10** |

  Both reach machine-precision dynamics feasibility — drag on the base
  AHO+column-precond config; lunar additionally needs cone-row scaling for its
  weaker-gravity slack magnitudes.
- **Honest scope.** The solver reaches ‖ν‖→0 but terminates `OuterIterCap`, not
  formally `Converged`: at `eta_max=200` the step keeps moving (`dx_du >
  conv_tol_x`), so it oscillates around the optimum after hitting feasibility.
  `min ‖ν‖` over accepted iters is the convergence-quality metric (same bar as
  `scvx_converges_larger_n_adaptive_trust`). Damping that residual oscillation
  into a formal `Converged` would need an `eta_max`/stationarity-aware trust
  rule — a bounded future refinement, not required for the envelope claim.
- **Tried and REVERTED (measured dead-end, recorded so it isn't re-treaded).**
  Holding the trust on the forced iter-0 (its ρ=1 is synthetic, and the in-code
  comment already *claimed* it "neither shrinks nor grows") **traded regimes**:
  lunar-base then converged (5.7e-12) but drag-base and lunar+cone-row regressed
  (0.28 / 0.66). The forced iter-0 growth is *incidentally load-bearing* — it
  gives the early subproblems room before the adaptive-trust gate can react,
  preventing premature trust collapse. Reverted; the retry alone covers both
  regimes.
- **Config levers ruled out** (`diag_envelope_widening`, a new `#[ignore]`d
  sweep): adaptive Tikhonov regularization and cone-row scaling do NOT by
  themselves widen the AHO envelope — `use_adaptive_regularization`
  over-regularizes at the vanishing-cone boundary and breaks iter 0 (confirming
  the existing docstring warning); cone-row scaling helps lunar specifically but
  hurts drag. Only the outer-loop retry generalizes.
- **Tests.** `scvx_active_drag_path_exercised_and_handled` **upgraded** from
  graceful-handling to a convergence gate (status ∈ {Converged, OuterIterCap},
  min ‖ν‖ < 1e-6, `InnerFailure` no longer acceptable). New
  `scvx_converges_lunar_gravity` (lunar with gravity-appropriate `t_min`, base
  config, min ‖ν‖ < 1e-6) and `scvx_drag_converges_at_larger_n` (N=8 drag, base
  config — the generalization gate). New `#[ignore]`d `diag_envelope_widening`
  sweep (N ∈ {5,8,10} × config matrix). The retry triggers ONLY on inner
  failure, which the Mars-regime tests never hit, so all prior tests are
  byte-for-byte unaffected.
- **Verification:** 126 tests pass (was 124; +1 lunar convergence, +1 larger-N
  drag gate, drag upgraded in place, +1 ignored diagnostic), clippy `-D warnings`
  clean, thumb (solver) cross-compile clean. Mars no-drag path unchanged.

### Larger-N generalization + lunar `t_min` root cause (follow-up)

A post-commit sweep (`diag_envelope_widening`, now parameterized over
N ∈ {5,8,10} with a config matrix) confirms the retry's envelope win is NOT an
N=5 artifact and pins down the lunar marginality:

- **Active drag converges on the production base config (column preconditioning
  only) across N = 5 / 8 / 10** — min ‖ν‖ = 1.6e-10 / 1.1e-9 / 2.0e-10. Locked in
  as a permanent CI gate (`scvx_drag_converges_at_larger_n`, N=8).
- **Lunar's earlier marginality was a physical-params mismatch, not a solver
  limit.** With Mars's `t_min = 1000 N`, a lunar descent (hover ≈ 324 N at
  m_dry) drives the `σ − T_min` thrust-floor cone hard-active — a vanishing-cone
  stressor — so base converges only at N ≥ 8 and cone-row only at N = 5 (no
  single config spans all N). Scaling the floor to the weaker gravity
  (`t_min = 300 N`, ~5% of `t_max`) removes the stressor: **lunar then converges
  on the base config across N = 5 / 8 / 10** — min ‖ν‖ = 8.2e-11 / 1.0e-10 /
  7.8e-9 (N=5 even reaches inner-IPM `Optimal`). `scvx_converges_lunar_gravity`
  updated to this clean config (was cone-row at N=5).
- **Net envelope statement**: with gravity-appropriate thrust limits, BOTH drag
  and lunar converge to machine-precision dynamics feasibility on the production
  DEFAULT config (no cone-row, no adaptive-reg) across the flight-relevant node
  range. Cone-row scaling helps lunar ONLY in the mis-specified-`t_min` regime
  and hurts elsewhere; adaptive-reg stays ruled out (over-regularizes the
  vanishing-cone boundary, breaks iter 0).

---

## Phase 18 — deployability consolidation (docs only, zero behavior change)

A pure-documentation pass to lock in the Phase-17 envelope win for integrators
and clear the doc-truth nits the alignment audits flagged. NO production logic
changed; all 126 tests, clippy `-D warnings`, and the thumb cross-compile are
unchanged.

- **`t_min`-must-scale-with-gravity footgun documented** (the load-bearing
  deployability item from the Phase-17 follow-up). A `t_min` set too high for
  the local gravity (hover thrust ≲ `t_min`) drives the `σ ≥ T_min` cone
  hard-active and stresses the IPM endgame — a *modeling* mismatch the input
  validator can't catch. Now called out on `PhysicalParams::t_min` (`params.rs`)
  and in `INTEGRATION.md` §5.1, with the Mars-`1000` → lunar-`300` worked
  example.
- **`INTEGRATION.md` §10 validation scope updated** to the Phase-17 reality:
  convergence validated for Mars no-drag, active drag at N = 5/8/10, and lunar
  gravity at N = 5/8/10 on the production default config (reaching
  `OUTER_ITER_CAP` with a feasible trajectory, not formal `CONVERGED`).
- **Vestigial-primitive docstrings corrected** (audit doc-truth): `kkt.rs`
  (Riccati) and `block_tridiag.rs` carried "the whole point / flight WCET must
  use this" framing that read as load-bearing — both are standalone, test-only
  references; the shipped structured path reimplements block-Thomas inline in
  `scvx_solver::reduced_kkt`. `assemble.rs`'s module header now flags that its
  layout doc describes the *vestigial* LCvx assembler, not the production
  `assemble_scvx_socp`.
- **iter-0 trust comment corrected**: it claimed the forced ρ = 1 "neither
  shrinks nor grows" — Phase 17 proved it GROWS the trust and that the growth is
  load-bearing. The comment now states this and warns against "fixing" it.
- **WCET claim made honest**: the safety table said "Hard `max_iters = 25` cap";
  the IPM loop is in fact bounded by the caller-supplied `max_iters` (default 25;
  SCvx/FFI callers pass ≤ 50).

**Noted follow-ups**: the compile-time IPM `max_iters` hard cap is now **done
(Phase 19, below)**; external-oracle coverage for the SCvx outer loop / free-tf /
NT remains open (today only the dense-AHO toy SOCP has a baked CVXPY/Julia
oracle).

---

## Phase 19 — WCET hard cap (LANDED)

Makes the bounded-WCET guarantee **caller-independent**. Previously each inner-IPM
loop ran `params.max_iters` times — a caller-supplied value — so the safety
table's "hard 25 cap" was an over-claim (Phase 18 corrected it to the honest
"caller-bounded"). Now a compile-time `IPM_HARD_MAX_ITERS = 64`
(`scvx_ipm::socp`, re-exported crate-root) bounds **every** IPM loop — the dense
`solve_socp` / `solve_socp_nt` and all four structured drivers — via
`for iter in 0..params.max_iters.min(IPM_HARD_MAX_ITERS)`, and the iter-cap
return sites report the clamped count too (honest telemetry). No Rust or C caller
can push the worst-case inner-iteration count past the compiled bound; the WCET
is a build-time constant.

- **Cap = 64**: comfortably above every shipping config (default 25; SCvx outer
  loop + tests pass ≤ 50), so `min(max_iters, 64)` is a no-op for all current
  callers — **zero behavior change** (the 126 prior tests are byte-for-byte
  unaffected). The toy regression bench (`mehrotra.rs`) is intentionally NOT
  capped — it is a closed-form verification fixture, not a flight path.
- **Test**: `scvx::tests::ipm_iters_respect_hard_cap` runs the active-drag
  problem with `max_iters = 200` (≫ cap); those subproblems run their inner
  solver to the cap, so every recorded `ipm_iters` must stay ≤ 64 — the test
  fails (would report up to 200) if the clamp is ever removed. **127 tests pass.**
- **Docs**: the HANDOFF safety table and `INTEGRATION.md` §9 now state the real
  compile-time cap (replacing the Phase-18 honest-but-weak "caller-bounded"
  wording).

---

## Phase 20 — consolidation: audit-driven hardening (LANDED)

A from-scratch **5-agent line-by-line audit** of all 24 source files (one agent
per crate group, each re-deriving the math by hand). **Verdict: clean on
correctness** — the IPM / cone / NT / Jacobian / FOH-RK4-discretization formulas
were re-derived AND numerically re-verified to machine precision, and every
flight invariant holds. The audit surfaced only minor hardening items, all fixed
here. **Behavior-preserving**: the `mars_descent` example is byte-identical
(`cost=4.3699e3, τ=14.029s`); the validation additions only reject bad input.

- **Input validation gaps closed.** The FFI boundary and `solve_scvx` now reject
  non-finite / negative `rho`, `cd_a` (they feed the drag dynamics — a bad value
  poisoned `v̇`/the STM silently) and non-finite / non-positive `cos_theta_max`,
  `tan_gamma_gs` (the pointing / glide-slope cone rows). `solve_scvx` also rejects
  `m_dry ≤ 0`/non-finite (feeds `log(m_dry)` in the mass-floor cone) and a
  non-finite / non-positive `reference.tau` in BOTH fixed- and free-tf modes
  (fixed-tf previously had no τ guard — a NaN `initial_tau` flowed into the
  discretizer).
- **Dead config removed (false-claim cleanup in a flight crate).**
  `IpmAlgoParams::regularization` was never read → **wired** into the dense IPM's
  adaptive-reg term (default `1e-10` preserved ⇒ zero behavior change, with a
  non-finite/negative fallback). `IpmAlgoParams::cond_bail` (documented an
  IPM-level dense fallback on condition number that does not exist and does not
  fit the architecture) → **removed**. The unused `BoundaryCondition` type (latent
  `ln(0)` footgun if wired) → **removed**.
- **Doc-truth + hygiene.** `precondition.rs`: the trust-cone scale table said
  `max(trust_eta, pos, thrust)` but the code (correctly) uses `trust_eta`;
  `build_scaling_diagonal`'s u/σ reasoning conflated conditioning with
  feasibility — both corrected. `INTEGRATION.md` stale section-ref fixed.
  `assemble_scvx_socp` got `debug_assert!(N >= 1)`. `oracle_diff.rs`'s dead
  `cost_idx` block became a **real assertion** — `cᵀx` (the objective the solver
  minimized) is now checked against the oracle optimal cost at all 6 call sites.

---

## Phase 21 — external-oracle coverage for a flight-scale subproblem (LANDED)

Closes forward-path item #4. `oracle_diff.rs` only validated the IPM on three
**toy** SOCPs (≤ 4 vars, 1–2 cones). The new
`crates/scvx-solver/tests/oracle_scvx_subproblem.rs` extends external-oracle
coverage to a **real assembled SCvx subproblem** — the 19·N-var, 8·N-cone problem
the production outer loop hands the inner IPM, with all eight cone types, both
fixed-tf (57 vars) and free-tf (58 vars, with the global δτ column).

- **Transcription-free approach.** The test assembles a faithful iter-0
  subproblem (replicating `seed_linear_reference` + `discretize_foh` +
  `assemble_scvx_socp` + preconditioning) for a deterministic 100 m Mars
  descent, and (via an `#[ignore]`d `dump_oracle_fixtures` test) writes the
  standard-form matrices `(c, A, b, G, h, cones)` to `tools/oracle-data/`. The
  Python (CVXPY) and Julia (JuMP) oracle scripts re-solve that **exact generic
  SOCP** with Clarabel — no physics re-encoding, so there is zero Python/Rust
  modelling drift (the oracle validates the IPM's *solve* of the assembled
  matrices, taking the separately-unit-tested assembly as given).
- **Result.** CVXPY/Clarabel and Julia/Clarabel agree to **~1e-9** on the
  optimum (fixed-tf `2.176946359299e6` both; free-tf agree to 1e-9). The Rust AHO
  IPM matches the optimal **cost** to **5.3e-4** (fixed-tf) / **7.6e-3** (free-tf)
  with tight primal feasibility (`|Ax−b|`, `|Gx+s−h|` ~1e-11). It nails the
  primal objective; only the dual/complementarity rides the documented AHO
  endgame loose. The assertions therefore check optimal-cost agreement + primal
  feasibility (cost is invariant under the column + cone-row preconditioning, so
  it is directly comparable) and **report**, not assert, the duality gap — an
  honest framing of what AHO achieves.
- Also fixed a pre-existing oversight: `tools/py-oracle/.gitignore` was `*`,
  which ignored even `solve_canonical.py` (the root `.gitignore` intends to keep
  it). Now whitelisted, so both Python oracle scripts are tracked source.
- **127 → 129 tests** (+2 oracle subproblem tests; the dump is `#[ignore]`d).

---

## Phase 22 — exact closed-form SOC NT scaling + frontier measurement (LANDED)

Tackles the **NT / O(N) frontier** (forward item #1). Implements the canonical
Nesterov-Todd scaling for the second-order cone via unit-determinant **normalized
points** (the form ECOS / Clarabel / MOSEK use), in
`scvx_ipm::cone::soc_nt_scaling_exact`, replacing the operator-geometric-mean
`W² = arrow(s)⁻¹ᐟ²·arrow(y)·arrow(s)⁻¹ᐟ²` as the **primary** NT scaling (geomean
kept as `.or_else` fallback in `build_nt_block_for_cone`).

- **What landed (verified correct).** Generic over const `D` (only `soc_det` /
  `sqrt` / dot / `D×D` arithmetic — no eigendecomp or determinant trait, so **no
  per-`D` macro**), returning the symmetric PD pair `(W, W⁻¹)` with `W·s = W⁻¹·y`,
  i.e. `W²·s = y` **EXACTLY** (the geomean is exact only for aligned bars). Three
  new tests: `W²s=y` to machine precision at every SCvx cone dim; **vanishing-cone
  stability** — at `det ~ 1e-9` it stays bounded (`max|W|~1`, `max|W²s−y|~1e-16`)
  where the geomean's `arrow(s)⁻¹ᐟ²` (eigenvalue `1/√(s₀−‖s̄‖)`) overflows to
  ~`1e9`; and supported-action agreement with the geomean. The boundary blowups
  cancel because `w̄` has unit det and `η=(det y/det s)^{1/4}` is a ratio. This is
  the "SDPT3-level per-cone handling of vanishing cones" the frontier called for.

- **HONEST MEASUREMENT (the result).** The exact scaling is **necessary but not
  sufficient**. It eliminates the per-cone overflow failure mode, but NT **still
  diverges** on the flight subproblem at every altitude (2/10/50/100 m,
  `diag_nt_on_flight_subproblem`): primal infeasibility `|Ax−b|` *grows* to ~9–29
  and the duality gap explodes (`s·y/n ~ 1e12–1e14`), while AHO reaches
  BestFeasible. This **independently confirms and refines the Phase-15 finding**:
  the barrier is NOT the per-cone scaling (the exact canonical scaling does not
  help) — it is the **global NT Newton direction / centering** on the imbalanced
  vanishing-cone structure (virtual-control SOC⁸ carry `μ_cone ~ 1e-4` vs
  thrust/trust ~`1e7`, so `H = GᵀW²G` is catastrophically ill-conditioned; AHO's
  asymmetric arrow scaling tolerates this, NT's symmetric `W²` does not). **A real
  fix needs a wide-neighborhood / per-cone-balanced centering scheme — IPM
  research, not a better scaling.** Per-cone scaling is now exhausted as a lever.

- **Net.** A verified, zero-regression correctness upgrade to the opt-in NT path
  (the geomean was a documented approximation; this is the true NT scaling and
  removes an overflow-to-NaN failure mode — it only touches the NT path, AHO/
  production untouched), plus a clean negative result. AHO stays the production
  default; NT stays opt-in with graceful fallback.

---

## Phase 23 — post-implementation adversarial review (LANDED)

Two independent review agents re-derived the new NT scaling math from scratch
(in Python, NOT trusting the code's own tests) and adversarially stress-tested
the new validation + oracle infrastructure. **Verdict: no correctness or safety
bugs.** Independently confirmed: `W²s=y` exact (machine precision on aligned AND
misaligned bars), `W⁻¹` correct, genuine SOC automorphism (0/100000 random points
mapped outside the cone), all overflow/underflow paths gated, AHO numerically
untouched, the NT corrector now MORE exact (`s̃=ỹ` is now machine-exact, was
approximate). On the infra side: the dump↔Python/Julia-parser round-trip is
byte-identical, the `regularization` wiring is default-preserving (proven by the
byte-identical `mars_descent`), validation rejects only bad input, and cone-row
scaling provably preserves cost.

Fixes applied (all honesty / completeness — no behavior change on valid input):
- **Boundedness mechanism corrected (the substantive finding).** The
  `soc_nt_scaling_exact` docstring claimed `W` stays bounded "because `w̄` has
  unit det" — a **non-sequitur** (unit det does NOT imply bounded). The real
  reason is **central-path near-complementarity** (`s∘y≈μe` ⇒ `s̄≈ȳ` ⇒ `w̄≈e`);
  for ANTI-aligned `s̄,ȳ` riding the boundary, `W` grows `~1/√det` even with the
  exact scaling. The docstring + this HANDOFF now state this correctly, and the
  vanishing-cone test was rewritten to assert BOTH regimes: central-path
  (`max|W|≈1.01`, `W²s−y~1e-16` down to `det~1e-9`) AND anti-aligned (`max|W|`
  grows `19.7→197→1968`, finite but unbounded). Refines, does not change, the
  Phase-22 conclusion.
- **Validation completeness.** `cos_theta_max` is now bounded to `[0, 1]` (a
  cosine `> 1` has no real angle; `= 0` ⇒ 90° pointing is a valid degenerate
  config), and a stale contradictory FFI comment was reconciled.

---

## Phase 24 — post-interruption deep re-audit + CI (LANDED)

A recovery / re-grounding pass after a service interruption, then a fresh
**5-agent line-by-line audit of EVERY file** (Phase 20-23 changes, cross-crate
integration, and doc-truth in focus — heading toward a public release). **Git
integrity clean** (no half-applied edits). All math **independently re-verified**
(exact NT scaling re-derived in Python; both dynamics Jacobians + the FOH/RK4
τ-sensitivity ODE; block-tridiag / Sherman-Morrison; all 8 cone rows; header↔Rust
ABI parity; Python↔Julia↔Rust oracle identity reproduced). All flight invariants
hold; **no unfinished/half-applied code; pre-public secret/path scan clean.**

> **WDAC note**: native EXECUTION of the freshly-built `scvx_solver` test binary
> is intermittently blocked by Windows Application Control (`os error 4551`) — it
> drops that suite from the count (71/132) but is **not** a regression. Verify on
> Linux: **WSL2** (`CARGO_TARGET_DIR=$HOME/...`) or the Docker clean-room. WSL2
> gives the full **132 pass** + deterministic `4.3699e3`, matching Windows.

Fixes (1 MEDIUM, 9 LOW — none a flight-path correctness/safety defect):
- **MEDIUM — structured-NT scaling parity.** Phase 22 upgraded the DENSE NT path
  to `soc_nt_scaling_exact` but left the STRUCTURED NT driver
  (`build_per_cone_nt_blocks`) on the old geometric-mean form — a silent
  primitive divergence that re-introduced the `arrow(s)⁻¹ᐟ²` overflow on the
  structured path. Routed it through `soc_nt_scaling_exact` (geomean fallback),
  mirroring the dense path; the structured-NT one-iter equivalence tests now
  match at machine precision. (Matters for the upcoming NT/O(N) frontier.)
- **Validation completeness** (`solve_scvx`): rejects a NaN/negative `t_min`, a
  NaN/`≤ t_min` `t_max` (the only phys params that escaped the `BadInput`
  contract; +2 red-team attacks), and `N == 0` (a release-mode OOB the
  `debug_assert` missed).
- **Doc-truth (pre-public accuracy):** `continuous.rs` thrust-eps "Smaller"→
  "Larger" (it is 100× larger); `socp.rs` `regularization` docstring cites
  `rel_factor`; HANDOFF quick-verify 120→132 + corrected split;
  `jl-oracle/solve_canonical.jl` `Pkg.activate(".")`→`@__DIR__` (the documented
  invocation was broken); FFI header + `INTEGRATION.md` corrected (BadInput /
  NullPointer leave the output buffer **unmodified**, not "a seeded reference" /
  "always finite"); `INTEGRATION.md` §5.1 now documents the
  `rho`/`cd_a`/`cos_theta_max`/`tan_gamma_gs` constraints; `mars_descent`
  dangling test-name reference fixed.
- **CI added** — `.github/workflows/ci.yml` (pinned to the MSRV 1.94) runs
  clippy `-D warnings` + the 132-test workspace + the deterministic example +
  the thumb `no_std` (solver + FFI) and host FFI builds on every push / PR. The
  repo is now self-verifying. (GitHub-side execution is gated by the account's
  Actions billing on the private repo — resolves on billing-fix or the public
  flip, where Actions is free; the workflow is verified green on Linux/WSL2.)

---

## Phase 25 — NT/O(N) frontier: per-cone centering measured + reverted (LANDED)

Continuing the NT-convergence frontier (forward item #1). After the exact
closed-form NT scaling (Phase 22 — a measured no-op for convergence), the next
hypothesized lever was the **centering**: the standard Mehrotra target `σμ·e`
uses the global average `μ`, which on the flight subproblem (per-cone gaps
spanning ~10¹¹ at the optimum) over-centers the vanishing cones. Implemented a
**per-cone / wide-neighborhood target `σ·μ_c·e_c`** behind a flag and
A/B-measured it against global centering on the flight subproblem (altitudes
2 / 10 / 50 / 100 m, `diag_nt_on_flight_subproblem`).

**Result: byte-identical NO-OP.** NT-per-cone == NT-global to all printed digits
(same `NumericalError`, same ~21-32 iters, same cost, same residuals) at every
altitude. Root cause: the Colombo-Gondzio weighted corrector already drives
**ω→0 (rejecting the corrector entirely)** on this ill-conditioned subproblem, so
any change to the corrector's centering target is moot. **The divergence is in
the affine NT step — the `H = GᵀW²G` ill-conditioning from the W² spread — not
the centering.** Reverted (a measured no-op = dead config); a `NOTE` in
`solve_socp_nt` records it so it isn't re-treaded.

**Refined frontier conclusion (honest state after Phases 22 + 25).** Three
incremental NT levers are now measured-exhausted: (a) exact per-cone scaling
(Phase 22 — no-op), (b) per-cone / wide-neighborhood centering (Phase 25 — no-op,
corrector rejected), and the earlier (c) iterative refinement (reverted —
"breakdown is linearization degeneracy, not solve accuracy"). The barrier is
**intrinsic**: symmetric-NT scaling produces an ill-conditioned affine Newton
step when cones vanish, and the corrector cannot rescue it. AHO's **asymmetric**
arrow scaling is empirically robust to vanishing cones (and is the production
default) — there is no incremental tweak that makes symmetric NT behave like
asymmetric AHO. Genuinely advancing NT now requires a **different framework**
(e.g., a homogeneous self-dual embedding / Mehrotra-on-HSD) — a large research
effort essentially equivalent to a new solver, NOT an incremental fix. The O(N)
end-to-end win is tied to the same: structured **NT** is blocked by NT
divergence; structured **AHO** is blocked by the AHO-endgame fallback erosion
(needs a regularized-Schur / QR structured factorization). Both are
research-grade.

**Recommendation: NT stays opt-in with graceful AHO fallback; AHO is the
validated production direction. The NT/O(N) frontier is now thoroughly
characterized — further progress is a research project (HSD), not a tweak.**

No production behavior change (per-cone centering reverted; the only NT artifacts
kept are the documenting `NOTE` and the Phase-24 structured-NT exact-scaling
parity). 132 tests pass (Linux), clippy clean.

---

## Final state summary

```
Tests:      132 passing across 5 crates + 3 integration suites + 3 API + 8 FFI tests
              (127 → 129 → 132: Phase 21 added +2 external-oracle flight-subproblem
                              tests — AHO vs CVXPY/Clarabel + Julia on the real
                              57/58-var assembled SCvx SOCP; Phase 22 added +3
                              scvx-ipm cone tests for the exact NT scaling)
              (Phase 10 v1 added: +1 FFI flag/entrypoint-mismatch regression
                                + 1 FFI Isp/g0=0 rejection
                                + 1 FFI N=8/N=10 expanded-surface smoke test;
               Phase 11 added:   + 1 larger-N convergence (N=10 tuned trust),
                                 + 2 #[ignore]d N-sweep / tuning diagnostics;
               Phase 17 added:   + 1 lunar-gravity convergence (active-drag test
                                   upgraded in place to a convergence gate),
                                 + 1 larger-N drag convergence gate (N=8),
                                 + 1 #[ignore]d envelope-widening sweep;
               Phase 19 added:   + 1 WCET hard-cap regression test)
              (+22 vs phase-5 end: 5 block_tridiag + 7 reduced_kkt
                                 + 1 active-drag flight-envelope coverage
                                 + 1 solve_socp_structured live-driver equivalence
                                 + 1 SCvx-outer-loop structured-solve fixed-tf end-to-end
                                 + 1 SCvx-outer-loop structured-solve free-tf end-to-end
                                 + 1 factor+apply equivalence
                                 + 1 free-tf SMW vs dense LU equivalence
                                 + 1 structured-NT vs dense-NT one-iter equivalence
                                 + 1 structured-NT-free-tf vs dense-NT one-iter equivalence
                                 + 2 FFI defense (misalignment / pathological inputs)
                                 + 1 wcet structured-vs-dense benchmark
                                 + 1 wcet factor+apply-vs-one-shot benchmark)
Crates:     5 (scvx-core, scvx-dynamics, scvx-ipm, scvx-solver, scvx-ffi)
Example:    cargo run --release --example mars_descent — produces a
              real Mars-descent trajectory, exit code 0
C-FFI:      cargo build --release -p scvx-ffi — produces .lib, .dll, .rlib
              with hand-written header at crates/scvx-ffi/include/scvx_ffi.h
Clippy:     clean with --all-targets -- -D warnings
Thumb:      cargo build --release --target thumbv7em-none-eabihf -p scvx-solver clean
              (solver core is no_std). The FFI boundary now ALSO cross-compiles:
              cargo build -p scvx-ffi --no-default-features --features panic-handler
                --target thumbv7em-none-eabihf  → staticlib + rlib for the MCU.
              Default (host) build keeps `std` for the panic handler + #[test];
              flight binaries link the rlib with --no-default-features and supply
              their own #[panic_handler] (watchdog / fault vector).
Unsafe:     zero in flight crates (FFI uses unsafe for raw pointers; all
              audited and bounded by null-check + caller contract)
Alloc:      zero in flight crates (Box only in #[cfg(test)] and examples)
Panic:      zero outside #[cfg(test)] (verified by grep)
Files:      23 .rs files (21 production + 2 integration tests) + 1 example + 1 C header
Lines:      ~7000 production LOC, ~6200 test LOC (rough estimate)
```

**The project is in a clean, audited, demonstrably-deployable state
with C-ABI surface AND a verified structured-KKT primitive ready for
the O(N) IPM lift.** Six SCvx phases have landed: column
preconditioning (P11), cone-row scaling (Phase 2), free-final-time
τ optimization (Phase 3), application API + example binary
(Phase 4), C-FFI + NT Higham + N=10 scale-up (Phase 5), and the
**block-tridiagonal Schur primitive (Phase 6, the P3b foundation)**.

## Phase 6.8 audit (free-tf Sherman-Morrison, post-interruption)

A full line-by-line audit was run across all source after the Phase 6.8
free-tf SMW landing (4 parallel reader agents covering every file). The
codebase came back overwhelmingly clean — nearly all findings were `OK`
confirmations. Three hardening items were applied:

- **γ overflow guard** (`reduced_kkt.rs::factor_reduced_kkt_scvx_block_m_free_tf`):
  added `if !factor.gamma.is_finite() { return SchurSingular }` after the
  SMW denominator division. Documented the PD-structure insight that
  `uᵀv = uᵀS_tridiag⁻¹u ≥ 0` ⇒ `denom ≥ 1` in exact arithmetic, so the
  guard is defense-in-depth against round-off / pathological caller `M`,
  not a normally-reachable path.
- **Exhaustive δτ-cone scan documented**: the `O(2·NP)` column scan that
  identifies the two τ-bound cones is intentional (stays correct if a
  future assemble path couples δτ to a stage variable); the `break
  'col_scan` fast-fails stage cones after one hit.
- **Maintenance cross-reference notes** added to both
  `solve_socp_structured` and `solve_socp_structured_free_tf` — the two
  Mehrotra loop bodies are intentionally duplicated (matching the
  project's existing `solve_socp` / `solve_socp_nt` pattern); a change to
  one must be mirrored in the other.

Verified safe with code/line refs: SMW signs (`Δλ = y₀ − γ(uᵀy₀)v`,
`Δδτ = α(b_x_δτ − Σa_δτ·Δλ)`), a_δτ row indexing (dynamics blocks only),
v_smw caching (built once, reused both RHS), loosened `NP == N·NZ ||
N·NZ+1` asserts (all relevant fns accept both layouts), `block_tridiag_
back_sub` forward/back-sub indices, free-tf driver dl-stitching uses
`sol_buf.base.dlam_*`, `stack_dz_free_tf` sets `dx[N·NZ] = dz_delta_tau`,
RB_MAX bound, FFI free-tf τ-bound validation, header/struct field-order
consistency. Equivalence to dense LU holds at machine precision (Δz 1.2e-13,
Δδτ 2.4e-14, Δλ 1.8e-12). *(That pass was at 117 tests; the count is now
119 after Phases 6.9–6.10 added the NT and NT-free-tf structured drivers.)*
**Clippy/thumb/FFI clean, example trajectory unchanged.**

## Post-consolidation audit (red-team pass)

After a context-window roll-over, a full top-to-bottom audit was performed
across every source file (12,220 LOC, 24 .rs files). Findings:

- **Two false alarms** investigated and dismissed with code reference:
  - `socp.rs:807` "double-W scaling" in NT corrector — verified the
    formula `b_x = -r_x - GᵀW²r_g + GᵀW·r_c_arg` where `r_c_arg =
    arrow(s̃)⁻¹·F̃_4`. The `W * r_c_arg` factor is exactly one `W` outside
    the cached `arrow(s̃)⁻¹` factor; matches the documented derivation.
  - `socp.rs:968` σ=0.5 fallback on non-finite — properly defended by
    line 947-951 affine-finite check upstream and line 998 corrector NaN
    guard downstream. 0.5 is a sane "between predictor and pure centering"
    fallback; verified by the `nt_full_precond_fails_gracefully` test.

- **One missing artifact reconstructed**:
  - `crates/scvx-ffi/include/scvx_ffi.h` was missing from disk (directory
    existed, header did not). Rebuilt from the FFI source as a single
    self-contained C99 header with all `#[repr(C)]` types, status code
    `#define`s, and 9 extern function signatures. Includes inline usage
    example and full safety contract for C callers.

- **Defense-in-depth fixes** added to `scvx-ffi`:
  1. **Runtime workspace alignment check**. Every `scvx_solve_n*` entry
     point now verifies `(workspace as usize) % align_of::<WS>() == 0`
     before casting. Misaligned buffers (a UB-class hazard for f64 fields
     on ARM) are rejected with `BadInput`, not silently corrupted. Note
     the check uses `align_of::<WS>()` (compiler-chosen), not the
     documented 8 bytes — degrades gracefully on SIMD-aligned targets.
  2. **Pathological physical-parameter validation**. Negative dry mass,
     `m_wet ≤ m_dry`, `t_max ≤ t_min`, zero initial τ, NaN initial state
     — all are caught at the FFI boundary with `BadInput`, not allowed
     to propagate into the inner solver. Tests
     (`misaligned_workspace_rejected`, `pathological_inputs_rejected`)
     pin these behaviors.
  3. **Target-mass band check** (added in the second audit pass):
     `target_mass` must be in `[m_dry, m_wet]`. Earlier versions
     accepted any finite value, which let an out-of-band target seed
     the SCvx reference at an infeasible landing mass and waste outer
     iterations chasing an impossible solution. Now rejected at the
     FFI boundary with `BadInput`. Tests added: target_mass below
     m_dry → reject; above m_wet → reject; free-tf initial_tau
     outside [tau_lo, tau_hi] → reject.

- **Hardening fixes** added to `scvx-solver`:
  4. **`tmp` buffer overflow guard** in `reduced_kkt.rs::build_stage_h_blocks_block_m`.
     The function uses a stack-local `tmp[NZ][MAX_CONE_DIM_FOR_BLOCK_M]`
     accumulator. Previously the size constant (11) was inline-magic
     with only a `debug_assert!(d <= 11)`. Release builds would have
     silently overflowed for a hypothetical cone with dim > 11. Now
     the loop body short-circuits via `continue` if `d > MAX` — the
     cone's H contribution is skipped (likely causing IPM
     non-convergence — but no UB). Still gated by debug_assert for
     dev-build diagnosis.
  5. **`RB_MAX` overflow guard** in `reduced_kkt.rs::solve_via_block_tridiag`.
     The block-tridiag work arrays are sized `RB_MAX = 64`. If a caller
     passes N > 63, the inline block-Thomas would have overwritten
     adjacent stack on `d_blocks[k]` etc. Now an explicit
     `if nrb > RB_MAX { return DegenerateInput; }` rejects the request
     cleanly before any out-of-bounds write.
  6. **`== 0.0` floating-point comparison documented as safe**. The
     `find_stage_for_cone` function uses exact-equality on `g_mat`
     entries to detect cone-to-stage ownership. This is safe under
     the current `assemble_scvx_socp` discipline (every G entry is
     either default-zero or an exact assigned constant — no fp
     accumulation). Comment now documents the contract and the
     migration path if future assemble code violates it.

- **Eight specific red-team vectors verified safe** with code/line refs:
  mass-floor enforcement (cone-enforced + log-mass monotone),
  free-tf τ clamping (cone + post-solve `.clamp()`), trust-region
  adaptation (bounded by `[trust_eta_min, trust_eta_max]`), IPM iterate
  explosion (caught at `max_abs > 1.0e50`), workspace reusability after
  `InnerFailure` (entry resets `workspace.iter = 0`), const-generic
  dimensions (compile-time, no overflow path), `solve_scvx` red-team
  test covers 6 attack vectors (NaN trust, negative weights, infinite
  velocity, etc.), and the FFI null-pointer / misalignment / pathological
  guard tests.

The audit reaffirmed: **zero panic in flight code, zero alloc in flight
crates, zero unsafe outside the audited scvx-ffi boundary**. All MISRA-C-
adjacent disciplines are intact.

**Demonstrably "deployable":**
- The Rust API: `cargo run --release --example mars_descent` produces
  a real soft-landing trajectory.
- The C API: `cargo build --release -p scvx-ffi` produces `.lib`,
  `.dll`, and a hand-written `scvx_ffi.h` header. Flight C/C++ code
  can link against the static lib and call `scvx_solve_n3()` /
  `scvx_solve_n5()`.
- N=10 scale-up works cleanly with the right tuning.

The Phase 6 driver substitution is **complete and end-to-end usable**:
`solve_socp_structured` in `scvx-solver/src/structured_socp.rs` mirrors
`scvx_ipm::solve_socp` but uses the block-tridiagonal Schur factorization
internally. Single-iter iterates match the dense driver to machine
precision (1.4e-13). The O(N·NZ³) vs O((N·NZ)³) speedup is empirically
measured at 6.82× at N=7. The dense driver remains the production
default (`use_structured_solve = false`) for safety; setting the flag
to `true` opts into the structured driver with a dense fallback safety
net. Both paths produce equivalent converged trajectories.

Do not undo the `τ` preservation, the `clip01` NaN-safety, the
`numerical_exit` scrubber, the entry validation in `solve_scvx`, the
preconditioning unscale-before-consumers ordering, the trust-cone scale
choice (`trust_eta` only — NOT `max(trust_eta, pos, thrust)`), or any of
the audit-driven defenses without VERY careful thought — they exist
because real attack vectors were caught and patched.

Do not undo the `τ` preservation, the `clip01` NaN-safety, the
`numerical_exit` scrubber, the entry validation in `solve_scvx`, or any of
the audit-driven defenses without VERY careful thought — they exist
because real attack vectors were caught and patched.

Read this file. Run the verification. Then proceed.
