/* scvx_ffi.h - C-ABI for the SCvx powered-descent solver
 *
 * Generated to mirror crates/scvx-ffi/src/lib.rs. Update both in lockstep
 * if any extern function or repr(C) type changes.
 *
 * Full integration guide (building, cross-compile, memory model, status
 * codes, determinism/WCET, a complete linked example): see
 * crates/scvx-ffi/INTEGRATION.md.
 *
 * --- Usage from C ---
 *
 *   #include "scvx_ffi.h"
 *
 *   CPhysicalParams phys = { ... };
 *   double initial_state[7] = { rx, ry, rz, vx, vy, vz, log(m) };
 *   CTerminalCondition target = { {0, 0, 0}, {0, 0, 0} };
 *   COptions opts;
 *   scvx_options_default(&opts);
 *   opts.use_free_tf = 0;   // REQUIRED for a fixed-tf entrypoint: defaults
 *                           // set use_free_tf = 1, which a fixed-tf solve
 *                           // rejects with SCVX_STATUS_BAD_INPUT.
 *   opts.initial_tau = 10.0;
 *   opts.target_mass = 380.0;
 *
 *   size_t ws_size = scvx_workspace_size_n3();
 *   uint8_t* workspace = aligned_alloc(8, ws_size);
 *
 *   CTrajectoryN3 traj;
 *   ScvxStatus s = scvx_solve_n3(
 *       &phys, initial_state, &target, &opts, workspace, &traj);
 *
 *   if (s == SCVX_STATUS_CONVERGED || s == SCVX_STATUS_OUTER_ITER_CAP) {
 *       // traj.r[k], traj.v[k], traj.mass[k], traj.u[k], traj.sigma[k], traj.tau
 *   }
 *
 *   free(workspace);
 *
 * --- Memory model ---
 *
 * - Caller-allocates: pass pointers to all inputs and outputs.
 * - The `workspace` buffer must be at least `scvx_workspace_size_n*()`
 *   bytes and 8-byte (f64) aligned. Misaligned buffers are rejected with
 *   SCVX_STATUS_BAD_INPUT.
 * - Concurrent access to the same workspace is forbidden.
 * - The solver does not call malloc/free. No Rust heap allocation occurs.
 *
 * --- Safety contract for C callers ---
 *
 * Every pointer argument must be non-null and valid. NULL pointers are
 * detected and return SCVX_STATUS_NULL_POINTER (no UB).
 *
 * `initial_state` MUST point to an array of at least 7 doubles. Shorter
 * arrays cause out-of-bounds reads (the FFI cannot detect length from a
 * raw pointer).
 *
 * The solver guarantees: no panic across the FFI boundary; no NaN/inf in
 * the output trajectory if status is Converged or OuterIterCap.
 *
 * --- Units ---
 *
 * SI throughout: meters, m/s, kg, seconds, Newtons. Mass is stored as
 * `log(m)` internally (the state's 7th component); the output trajectory
 * unwraps this to physical mass in kg.
 */

#ifndef SCVX_FFI_H
#define SCVX_FFI_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ===========================================================================
 * Status codes
 * ===========================================================================
 *
 * Returned by every FFI entrypoint. Match `ScvxStatus` in lib.rs.
 */
typedef uint8_t ScvxStatus;

/* Solver converged within tolerances. Trajectory is valid. */
#define SCVX_STATUS_CONVERGED        ((ScvxStatus) 0)
/* Reached max outer iterations without full convergence. Trajectory is the
 * best-found solution; usable as a warm-start for a re-solve. */
#define SCVX_STATUS_OUTER_ITER_CAP   ((ScvxStatus) 1)
/* Inner IPM failed (cone violation, numerical breakdown). Trajectory
 * contains last consistent iterate but should not be used as a flight
 * plan. */
#define SCVX_STATUS_INNER_FAILURE    ((ScvxStatus) 2)
/* Problem is infeasible (BCs cannot be reached within trust + cones). */
#define SCVX_STATUS_INFEASIBLE       ((ScvxStatus) 3)
/* Caller-supplied input failed validation (negative mass, zero tau,
 * tau_lo >= tau_hi, non-finite Isp/g0/gravity, dim/use_free_tf mismatch,
 * etc.). */
#define SCVX_STATUS_BAD_INPUT        ((ScvxStatus) 4)
/* One or more pointer arguments was NULL. */
#define SCVX_STATUS_NULL_POINTER     ((ScvxStatus) 255)

/* ===========================================================================
 * Physical parameters
 * ===========================================================================
 *
 * Layout matches `CPhysicalParams` in lib.rs.
 */
typedef struct CPhysicalParams {
    double g[3];               /* Gravity vector, m/s^2. (e.g. [0,0,-3.7114] for Mars) */
    double m_dry;              /* Dry mass (kg), >0. Floor on landing mass. */
    double m_wet;              /* Wet (fully fueled) mass (kg), > m_dry. */
    double isp;                /* Specific impulse, seconds. */
    double g0;                 /* Reference gravity for I_sp (9.80665 standard). */
    double t_min;              /* Minimum thrust magnitude (N). */
    double t_max;              /* Maximum thrust magnitude (N), > t_min. */
    double cos_theta_max;      /* cos of max pointing angle from vertical (1=no constraint). */
    double tan_gamma_gs;       /* tan of glide-slope angle (large=no constraint). */
    double rho;                /* Atmospheric density (kg/m^3). 0 = no drag (LCvx-style). */
    double cd_a;               /* CdA = drag coefficient × reference area (m^2). */
    double tau_lo;             /* Lower bound on time-of-flight (s), > 0. */
    double tau_hi;             /* Upper bound on time-of-flight (s), > tau_lo. */
} CPhysicalParams;

/* ===========================================================================
 * Terminal condition (soft-landing target)
 * ===========================================================================
 *
 * Layout matches `CTerminalCondition` in lib.rs.
 */
typedef struct CTerminalCondition {
    double r[3];               /* Target position at landing (m). */
    double v[3];               /* Target velocity at landing (m/s). Typically zero. */
} CTerminalCondition;

/* ===========================================================================
 * Solver options
 * ===========================================================================
 *
 * Layout matches `COptions` in lib.rs.
 *
 * Recommended usage: call `scvx_options_default(&opts)` first, then
 * override only the fields you need (typically `initial_tau`,
 * `target_mass`, and the trust radii).
 */
typedef struct COptions {
    double initial_tau;          /* Initial guess for time-of-flight (s). */
    double target_mass;          /* Estimated terminal mass (kg). Seeds the reference. */

    /* Boolean flags: 0 = disabled, non-zero = enabled. */
    uint8_t use_free_tf;          /* 1 = optimize tau as a free variable. */
    uint8_t use_preconditioning;  /* 1 = per-variable column scaling. RECOMMENDED. */
    uint8_t use_cone_row_scaling; /* 1 = per-cone slack row scaling. AHO-only. */
    uint8_t use_nt_scaling;       /* 1 = NT direction. AHO (=0) is the safer default. */
    uint8_t use_hsd;              /* 1 = HSD (homogeneous self-dual) direction. RECOMMENDED;
                                     converges where NT diverges. Precedes use_nt_scaling. */
    uint8_t use_structured_solve; /* 1 = O(N) block-tridiagonal structured solve; with
                                     use_hsd this is the structured HSD (~7x faster, N=7). */

    uint32_t max_outer_iters;    /* SCvx outer loop cap, e.g. 15. */
    uint32_t max_inner_iters;    /* IPM inner loop cap, e.g. 25. */

    double conv_tol_x;           /* Outer convergence: ‖Δx‖_∞ tolerance. */
    double conv_tol_virt;        /* Outer convergence: ‖ν‖_1 tolerance. */
    double ipm_tol;              /* Inner IPM tolerance (μ, primal, dual). */

    double trust_eta0;           /* Initial trust radius. */
    double trust_eta_min;        /* Floor on trust radius. */
    double trust_eta_max;        /* Ceiling on trust radius. */

    double virt_weight;          /* Penalty on virtual-control slack, e.g. 1e4. */
} COptions;

/* ===========================================================================
 * Output trajectory types (one concrete struct per supported N)
 * ===========================================================================
 *
 * C has no generics, so each node count N gets its own struct. Layout is
 * row-major node-indexed and mirrors `emit_traj_ty!` in lib.rs EXACTLY:
 *   r[k]    = position at node k (m)         [N][3]
 *   v[k]    = velocity at node k (m/s)        [N][3]
 *   mass[k] = vehicle mass at node k (kg; FFI unwraps the internal log-mass)
 *   u[k]    = thrust/accel vector at node k   [N][3]
 *   sigma[k]= thrust-magnitude slack at node k
 *   tau     = time-of-flight / dilation (s; meaningful for free-tf)
 *
 * The macro guarantees the C layout cannot drift field-by-field from Rust.
 */
#define SCVX_DEFINE_TRAJ(N)            \
    typedef struct CTrajectoryN##N {   \
        double r    [N][3];            \
        double v    [N][3];            \
        double mass [N];               \
        double u    [N][3];            \
        double sigma[N];               \
        double tau;                    \
    } CTrajectoryN##N

SCVX_DEFINE_TRAJ(3);
SCVX_DEFINE_TRAJ(5);
SCVX_DEFINE_TRAJ(8);
SCVX_DEFINE_TRAJ(10);
SCVX_DEFINE_TRAJ(12);
SCVX_DEFINE_TRAJ(15);
SCVX_DEFINE_TRAJ(20);

/* ===========================================================================
 * Functions
 * ===========================================================================
 */

/* Populate `options` with production-recommended defaults.
 *
 * Returns SCVX_STATUS_NULL_POINTER if `options` is NULL,
 * otherwise SCVX_STATUS_CONVERGED ("ok"; the status code is reused here
 * because there's no separate "ok-but-not-converged" code).
 *
 * NOTE: the defaults set `use_free_tf = 1`. If you call a fixed-tf solve
 * entrypoint (`scvx_solve_nN`, no suffix), you MUST clear it first:
 *   COptions o; scvx_options_default(&o); o.use_free_tf = 0;
 */
ScvxStatus scvx_options_default(COptions* options);

/* ---------------------------------------------------------------------------
 * Per-N solve entrypoints.
 *
 * For each supported N, two (size, solve) pairs are declared: fixed-tf
 * (`scvx_solve_nN`) and free-tf (`scvx_solve_nN_free_tf`).
 *
 * Common contract for every solve function:
 *  - All six pointer arguments MUST be non-null (else SCVX_STATUS_NULL_POINTER).
 *  - `initial_state` points to >= 7 doubles: [r_x,r_y,r_z, v_x,v_y,v_z, ln(m)].
 *  - `workspace` points to >= `scvx_workspace_size_nN[_free_tf]()` bytes,
 *    aligned to 8 (f64). Misalignment / under-size is rejected with
 *    SCVX_STATUS_BAD_INPUT (size cannot be checked — honor the contract).
 *  - **Flag/entrypoint match is REQUIRED**: `scvx_solve_nN` needs
 *    `options->use_free_tf == 0`; `scvx_solve_nN_free_tf` needs
 *    `options->use_free_tf != 0`. A mismatch returns SCVX_STATUS_BAD_INPUT
 *    (it would otherwise index past the workspace).
 *  - When the solver ran (status Converged / OuterIterCap / InnerFailure /
 *    Infeasible), `*out_trajectory` holds the best trajectory found (possibly
 *    not fully converged) and finite. On BadInput / NullPointer the function
 *    returns BEFORE writing, so the buffer is left UNMODIFIED (caller's bytes)
 *    — do not read it. ALWAYS check the returned status before using the
 *    trajectory as a flight plan.
 *
 * Convergence is validated for N <= 5 (see the scvx-solver test suite).
 * Larger N is best-effort: the entrypoint always returns a status (commonly
 * SCVX_STATUS_OUTER_ITER_CAP on hard large-N problems) and never crashes.
 * A flight build typically keeps only the single N its mission uses — delete
 * the unused SCVX_DECLARE_N(...) lines here and the matching emit_ffi_per_n!
 * lines in lib.rs to shrink the binary.
 * --------------------------------------------------------------------------- */
#define SCVX_DECLARE_N(N)                                                 \
    size_t scvx_workspace_size_n##N(void);                                \
    ScvxStatus scvx_solve_n##N(                                           \
        const CPhysicalParams*    phys,                                   \
        const double*             initial_state,                          \
        const CTerminalCondition* terminal,                               \
        const COptions*           options,                                \
        uint8_t*                  workspace,                              \
        CTrajectoryN##N*          out_trajectory);                        \
    size_t scvx_workspace_size_n##N##_free_tf(void);                      \
    ScvxStatus scvx_solve_n##N##_free_tf(                                 \
        const CPhysicalParams*    phys,                                   \
        const double*             initial_state,                          \
        const CTerminalCondition* terminal,                               \
        const COptions*           options,                                \
        uint8_t*                  workspace,                              \
        CTrajectoryN##N*          out_trajectory)

SCVX_DECLARE_N(3);
SCVX_DECLARE_N(5);
SCVX_DECLARE_N(8);
SCVX_DECLARE_N(10);
SCVX_DECLARE_N(12);
SCVX_DECLARE_N(15);
SCVX_DECLARE_N(20);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* SCVX_FFI_H */
