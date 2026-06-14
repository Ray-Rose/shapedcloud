# SCvx Powered-Descent Solver — Flight Integration Guide

This guide is for flight C/C++ engineers integrating the `scvx-ffi` C-ABI
library into an embedded guidance application. It is the authoritative,
code-verified companion to the header `include/scvx_ffi.h`.

The internal algorithm (SCvx successive-convexification over a 3-DoF
log-mass powered-descent model with a second-order-cone IPM subproblem
solver) is documented in `HANDOFF.md` at the repo root. This document is
strictly about the **integration contract**.

---

## 1. What you link against

`scvx-ffi` builds to three artifacts (see `Cargo.toml` `crate-type`):

| Artifact            | File (Linux / Windows)            | Use                                   |
|---------------------|-----------------------------------|---------------------------------------|
| `staticlib`         | `libscvx_ffi.a` / `scvx_ffi.lib`  | Link into your flight binary at build |
| `cdylib`            | `libscvx_ffi.so` / `scvx_ffi.dll` | Dynamic load (ground tools, host sim) |
| `rlib`              | `libscvx_ffi.rlib`                | Rust callers                          |

For flight you want the **staticlib** linked into your firmware image.

Everything below the C boundary is `#![no_std]`, allocation-free, and contains
no panics on any reachable solve path (validated by the test suite and a
red-team audit; see `HANDOFF.md`). The only `unsafe` in the whole workspace is
the pointer handling in this crate's `extern "C"` functions, and every pointer
is null-checked before use.

---

## 2. Building

### 2.1 Host build (ground/sim, default)

```sh
cargo build --release -p scvx-ffi
```

The default `std` feature is on: the crate uses the standard panic handler and
`#[test]` works. This is what CI, the `mars_descent` example, and host
integration use.

### 2.2 Flight cross-compile (bare-metal, e.g. ARM Cortex-M)

The whole algorithm stack is `no_std`; the FFI boundary becomes `no_std` when
you disable the default `std` feature. Two patterns:

**(a) Standalone static library** — you want a self-contained `.a` to hand to
a C build system, with a built-in panic handler:

```sh
rustup target add thumbv7em-none-eabihf
cargo build --release -p scvx-ffi \
    --no-default-features --features panic-handler \
    --target thumbv7em-none-eabihf
# → libscvx_ffi.a + .rlib for the MCU. (cdylib is auto-dropped for bare-metal.)
```

The `panic-handler` feature installs a minimal busy-loop `#[panic_handler]`.
**A panic is never expected** (all reachable error paths return a status code),
but a handler symbol is required to *link* a `no_std` staticlib. For a real
mission you almost certainly want pattern (b) instead, so panics route to your
fault-management code.

**(b) Linked into a Rust flight binary that owns the panic handler**
(recommended for flight): depend on `scvx-ffi` with `default-features = false`
and **omit** `panic-handler`. Your binary crate provides the `#[panic_handler]`
(watchdog kick / fault vector / safe-mode entry). Do not enable `panic-handler`
in this case — two handlers will not link.

### 2.3 Trimming the binary

Each supported `N` (node count) monomorphizes the entire solver, so the
all-`N` static archive is large (~27 MB with N ∈ {3,5,8,10,12,15,20}). The
linker discards unreferenced `N` when it links your final image, but you can
also trim at the source: delete the unused `emit_ffi_per_n!`/`emit_traj_ty!`
lines in `src/lib.rs` and the matching `SCVX_DECLARE_N(...)`/`SCVX_DEFINE_TRAJ(...)`
lines in `include/scvx_ffi.h`. A typical mission keeps exactly one `N`.

---

## 3. Memory model: caller allocates

The C ABI uses a **caller-allocates** convention. No heap allocation occurs on
the Rust side of the boundary. For your chosen `N` and fixed/free-tf mode:

1. Ask for the workspace size:
   ```c
   size_t nbytes = scvx_workspace_size_n10();        /* fixed-tf, N=10 */
   /* or scvx_workspace_size_n10_free_tf() for free-tf */
   ```
2. Provide a buffer of **at least** `nbytes`, aligned to **8 bytes** (f64
   alignment). Static, stack, or pool allocation is fine — no `malloc`
   required.
   - The size **cannot** be checked at the boundary; under-provisioning is
     undefined behavior. Honor it exactly.
   - Misalignment **is** checked: a misaligned buffer returns
     `SCVX_STATUS_BAD_INPUT` (it is verified against the actual compiler-chosen
     alignment of the workspace, so an over-aligned SIMD build degrades cleanly
     rather than corrupting memory).
3. Provide an output `CTrajectoryN<N>*`. The solver writes the result there on
   every return (even on non-converged exits — see §6).

The workspace is reusable across calls (it is fully re-initialized each call),
but **must not** be shared concurrently between threads.

---

## 4. Calling a solve

```c
#include "scvx_ffi.h"

CPhysicalParams phys = { /* ...see §5... */ };
double x0[7] = { rx, ry, rz, vx, vy, vz, log(m0) };   /* note: log mass */
CTerminalCondition term = { .r = {0,0,0}, .v = {0,0,0} };

COptions opts;
scvx_options_default(&opts);     /* sets sensible defaults */
opts.use_free_tf = 0;            /* REQUIRED for a fixed-tf entrypoint! (see below) */
opts.target_mass = 380.0;
opts.initial_tau = 10.0;         /* time-of-flight guess, s */

size_t nbytes = scvx_workspace_size_n10();
uint8_t *ws = aligned_alloc(8, nbytes);    /* or a static 8-aligned buffer */
CTrajectoryN10 traj;

ScvxStatus st = scvx_solve_n10(&phys, x0, &term, &opts, ws, &traj);
if (st == SCVX_STATUS_CONVERGED) {
    /* use traj as the descent plan */
}
```

### 4.1 Fixed-tf vs free-tf, and the `use_free_tf` contract (IMPORTANT)

For each `N` there are two entrypoints:

- `scvx_solve_nN` — **fixed** time-of-flight. Requires `opts.use_free_tf == 0`.
- `scvx_solve_nN_free_tf` — **free** final time (the solver optimizes τ within
  `[phys.tau_lo, phys.tau_hi]`). Requires `opts.use_free_tf != 0`.

The workspace dimensions are baked into the entrypoint at compile time, but the
solver branches on the runtime `use_free_tf` flag. **If they disagree, the call
returns `SCVX_STATUS_BAD_INPUT`** (it would otherwise index past the
workspace). This is enforced at the boundary *and* inside the solver.

> ⚠️ **Trap:** `scvx_options_default` sets `use_free_tf = 1`. If you call a
> *fixed-tf* entrypoint, you **must** clear the flag yourself:
> `scvx_options_default(&o); o.use_free_tf = 0;`. Otherwise you get
> `BAD_INPUT`, not a solve.

---

## 5. Inputs

### 5.1 `CPhysicalParams` (all SI units)

| Field           | Meaning                                              | Constraint (else `BAD_INPUT`) |
|-----------------|------------------------------------------------------|-------------------------------|
| `g[3]`          | Gravity accel vector (m/s²), e.g. `{0,0,-3.7114}`    | finite                        |
| `m_dry`         | Dry mass (kg)                                        | finite, `> 0`                 |
| `m_wet`         | Wet (initial) mass (kg)                              | finite, `> m_dry`             |
| `isp`           | Specific impulse (s)                                 | finite, `> 0`                 |
| `g0`            | Reference gravity for Isp (m/s², ~9.80665)           | finite, `> 0`                 |
| `t_min`         | Min thrust (N)                                       | finite, `>= 0`                |
| `t_max`         | Max thrust (N)                                       | finite, `> t_min`             |
| `cos_theta_max` | Cosine of max thrust-pointing angle                  | (model input)                 |
| `tan_gamma_gs`  | Tangent of glide-slope cone half-angle               | (model input)                 |
| `rho`           | Atmospheric density for drag (0 disables drag)       | (model input)                 |
| `cd_a`          | Drag coefficient × area                              | (model input)                 |
| `tau_lo`        | Min time-of-flight (s) — free-tf only                | free-tf: finite, `> 0`        |
| `tau_hi`        | Max time-of-flight (s) — free-tf only                | free-tf: finite, `> tau_lo`   |

`isp` and `g0` appear as a *divisor* in the mass-flow dynamics; a zero/negative
value would inject NaN/Inf into the linearization, so they are validated to be
strictly positive.

### 5.2 `initial_state` — 7 doubles

`[r_x, r_y, r_z, v_x, v_y, v_z, ln(m)]`. Position (m), velocity (m/s), and the
**natural log of mass** (the solver works in log-mass coordinates). All seven
must be finite. The pointer must address at least 7 doubles — a shorter array
is an out-of-bounds read that cannot be detected at the boundary.

### 5.3 `CTerminalCondition`

`r[3]`, `v[3]` — desired terminal position and velocity (m, m/s). Must be
finite.

### 5.4 `COptions` — see `scvx_options_default` for recommended values

Key fields: `initial_tau` (time-of-flight guess, `> 0`); `target_mass` (must be
in `[m_dry, m_wet]`); `use_free_tf` (see §4.1); `use_preconditioning` (keep
`1`); `use_nt_scaling` (NT direction — opt-in, see §8); `max_outer_iters`,
`max_inner_iters` (compile-time-capped at 20 outer in the FFI); the `conv_tol_*`
/ `ipm_tol` tolerances; and the trust-region `trust_eta*` / `virt_weight`
parameters.

Validation scope (be precise here):
- `initial_tau` and `target_mass` are validated at the FFI boundary (see the
  table in §6's siblings above) → `BAD_INPUT`.
- The trust-region parameters are validated by the solver: `trust_eta*` and
  `virt_weight` must be finite, `trust_eta_min >= 0`, `trust_eta_max >=
  trust_eta_min`, and `virt_weight >= 0`, else `BAD_INPUT`. (Enforced inside
  `solve_scvx`, not at the FFI boundary — the effect is the same `BAD_INPUT`.)
- The convergence tolerances (`conv_tol_*`, `ipm_tol`) are **not** validated —
  they are taken as given. Pass finite, positive values; `scvx_options_default`
  provides sensible ones.

---

## 6. Status codes — ALWAYS check before using the trajectory

`ScvxStatus` is a `uint8_t`:

| Value | Macro                        | Meaning                                                        |
|-------|------------------------------|----------------------------------------------------------------|
| `0`   | `SCVX_STATUS_CONVERGED`      | Converged to tolerance. The trajectory is a valid plan.        |
| `1`   | `SCVX_STATUS_OUTER_ITER_CAP` | Hit the outer-iteration cap. Best-found trajectory returned; may be usable but did not meet the convergence test. |
| `2`   | `SCVX_STATUS_INNER_FAILURE`  | An inner SOCP subproblem failed. Trajectory is stale-but-finite; do **not** use as a plan. |
| `3`   | `SCVX_STATUS_INFEASIBLE`     | Problem detected infeasible.                                   |
| `4`   | `SCVX_STATUS_BAD_INPUT`      | An input/contract was violated (see §3–§5). No SCvx iteration ran; the output buffer holds only a seeded reference — do not use it. |
| `255` | `SCVX_STATUS_NULL_POINTER`   | A pointer argument was NULL.                                   |

Only `CONVERGED` (and, with mission-specific care, `OUTER_ITER_CAP`) should be
treated as a usable plan. The output trajectory is **always finite** on return
regardless of status (no NaN/Inf leaks across the boundary), so it is safe to
read/log — but its *meaning* is only guaranteed when `CONVERGED`.

---

## 7. Output: `CTrajectoryN<N>`

Row-major, node-indexed (`k = 0 .. N-1`). Layout mirrors the Rust side exactly:

| Field      | Type          | Meaning                                             |
|------------|---------------|-----------------------------------------------------|
| `r[k][i]`  | `double[N][3]`| Position at node `k` (m)                            |
| `v[k][i]`  | `double[N][3]`| Velocity at node `k` (m/s)                          |
| `mass[k]`  | `double[N]`   | Vehicle mass at node `k` (kg; log-mass already unwrapped) |
| `u[k][i]`  | `double[N][3]`| Thrust/accel command at node `k`                    |
| `sigma[k]` | `double[N]`   | Thrust-magnitude slack at node `k`                  |
| `tau`      | `double`      | Time-of-flight / dilation (s). Meaningful for free-tf. |

---

## 8. Solver-mode notes

- **AHO direction (default, `use_nt_scaling = 0`)** is the validated production
  path.
- **NT direction (`use_nt_scaling = 1`)** is an opt-in alternative. On hard
  flight-scale subproblems its inner IPM can fail to converge in the endgame;
  when it does, the solver **falls back** to the dense AHO driver automatically,
  so enabling NT never breaks a solve — it is at worst a no-op cost. (Details in
  `HANDOFF.md`, "NT endgame" notes.)
- **Structured solver** (block-tridiagonal Schur) is selected internally and is
  numerically equivalent to the dense path (verified to machine precision); it
  falls back to dense on any per-iteration breakdown.

---

## 9. Determinism and WCET

- **Deterministic across toolchains.** All transcendentals route through `libm`
  (not the platform math library), so the same input bits produce the same
  output bits on the flight target as on the host. The Mars-descent example is
  bit-identical between Linux x86-64 and Windows MSVC.
- **Bounded WCET.** There are no data-dependent unbounded loops. The outer loop
  is capped (`FFI_MAX_OUTER = 20`); each inner IPM is capped (`max_inner_iters`);
  the matrix-sqrt and substitution loops are compile-time bounded. Worst-case
  cost scales with `N` and the iteration caps, all known at build time.
- **No heap, no panics on the solve path.** Workspace is caller-provided; all
  error conditions return a status code.

---

## 10. Validation status (honest scope)

- Convergence is validated by the test suite for **N ≤ 5** (fixed- and free-tf,
  with drag and across gravity regimes).
- Larger `N` entrypoints (8, 10, 12, 15, 20) are provided and are verified to
  **run end-to-end and never crash or emit NaN/Inf**, but convergence on a given
  problem is best-effort — expect `OUTER_ITER_CAP` on hard large-`N` cases.
  Tune `max_outer_iters`, the trust-region parameters, and your reference seed
  for your specific mission, and validate against your own scenarios before
  flight.
- This is research-grade flight *software engineering* (MISRA-like discipline,
  no_std, no_alloc, bounded WCET, deterministic), not a flight-qualified product.
  Qualification (DO-178C / NPR 7150.2 etc.) is the integrator's responsibility.

---

## 11. Minimal complete example

```c
#include "scvx_ffi.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    CPhysicalParams phys = {
        .g = {0.0, 0.0, -3.7114},     /* Mars */
        .m_dry = 200.0, .m_wet = 1000.0,
        .isp = 225.0, .g0 = 9.80665,
        .t_min = 1000.0, .t_max = 6000.0,
        .cos_theta_max = 0.7660444,   /* 40 deg */
        .tan_gamma_gs  = 1.0,
        .rho = 0.0, .cd_a = 0.0,      /* drag off */
        .tau_lo = 5.0, .tau_hi = 50.0,
    };
    double x0[7]  = {0.0, 0.0, 2.0, 0.0, 0.0, -0.1, log(400.0)};
    CTerminalCondition term = { .r = {0,0,0}, .v = {0,0,0} };

    COptions opts;
    if (scvx_options_default(&opts) != SCVX_STATUS_CONVERGED) return 1;
    opts.use_free_tf = 0;             /* fixed-tf entrypoint below */
    opts.initial_tau = 10.0;
    opts.target_mass = 380.0;

    size_t nbytes = scvx_workspace_size_n5();
    uint8_t *ws = (uint8_t*)aligned_alloc(8, (nbytes + 7) & ~(size_t)7);
    if (!ws) return 1;

    CTrajectoryN5 traj;
    ScvxStatus st = scvx_solve_n5(&phys, x0, &term, &opts, ws, &traj);

    printf("status=%u  tof=%.3f s\n", (unsigned)st, traj.tau);
    if (st == SCVX_STATUS_CONVERGED) {
        for (int k = 0; k < 5; ++k)
            printf("  k=%d  alt=%.2f m  mass=%.1f kg\n",
                   k, traj.r[k][2], traj.mass[k]);
    }
    free(ws);
    return st == SCVX_STATUS_CONVERGED ? 0 : 2;
}
```

Build (host):

```sh
cargo build --release -p scvx-ffi
cc example.c -Icrates/scvx-ffi/include -L target/release -lscvx_ffi -lm -o example
```
