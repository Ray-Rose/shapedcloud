//! C-ABI wrapper around the SCvx powered-descent solver.
//!
//! Exposes [`scvx_solve_n3`] (and friends, one per supported N) as a
//! `extern "C"` function that flight C/C++ code can call directly.
//! Const generics are baked into the per-N entrypoints — C can't pass
//! const generics, so each supported `N` value gets its own function.
//!
//! ## Crate types
//!
//! - `staticlib` (`.lib` / `.a`): linkable from C/C++ at build time.
//! - `cdylib` (`.dll` / `.so` / `.dylib`): dynamic loading.
//! - `rlib`: usable from Rust callers (tests, examples).
//!
//! ## Memory model
//!
//! The C ABI uses the **caller-allocates** convention: the caller passes
//! a pointer to a `ScvxResult` and a pointer to an opaque workspace
//! (sized via [`scvx_workspace_size_n3`]). The solver writes into both.
//! No Rust heap allocation occurs at FFI boundary.
//!
//! ## Safety
//!
//! The `extern "C"` functions are unsafe (they dereference raw
//! pointers). Callers must ensure:
//! - All pointer arguments are non-null and valid for reads/writes.
//! - The workspace pointer points to a buffer of at least
//!   `scvx_workspace_size_n3()` bytes, aligned to `8` (f64 alignment).
//! - Concurrent access to the same workspace is forbidden.
//!
//! ## Status codes (via the `From<SolverStatus> for ScvxStatus` impl below)
//!
//! - `0` = Converged
//! - `1` = OuterIterCap
//! - `2` = InnerFailure
//! - `3` = Infeasible
//! - `4` = BadInput
//! - `255` = NullPointer (FFI-specific; caller passed a null pointer)

// The FFI crate is the C-ABI boundary layer. The internal algorithm code
// (scvx-core, scvx-dynamics, scvx-ipm, scvx-solver) is `#![no_std]`, and so is
// this crate UNLESS the default `std` feature is enabled.
//
//   * `std` (default): use std's panic handler; `#[test]` works on the host.
//   * `--no-default-features`: pure `#![no_std]`, cross-compiles to bare-metal
//     flight targets. The production flight binary supplies the
//     `#[panic_handler]` (watchdog / fault vector). Enable the `panic-handler`
//     feature to instead get a minimal built-in busy-loop handler so the
//     `staticlib`/`cdylib` artifacts link standalone for the target.
//
// All non-test code in this crate already uses only `core::` items, so the
// switch is purely the attribute below plus the optional handler.
#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_op_in_unsafe_fn)]

// Minimal no_std panic handler for standalone bare-metal artifacts. Gated so
// it never collides with a flight binary that provides its own (link the
// `rlib` with `--no-default-features` and omit `panic-handler` in that case).
#[cfg(all(not(feature = "std"), feature = "panic-handler"))]
#[panic_handler]
fn ffi_panic(_info: &core::panic::PanicInfo) -> ! {
    // No unwinding, no I/O available — spin. A real integration replaces this
    // (via its own binary's handler) with a watchdog signal or fault trap.
    loop {
        core::hint::spin_loop();
    }
}

use core::mem::MaybeUninit;
use core::ptr;

use nalgebra::SVector;
use scvx_core::{PhysicalParams, SolverStatus};
use scvx_solver::{
    solve_powered_descent, workspace_ncones, workspace_nct, workspace_np,
    workspace_ne, PoweredDescentOptions, ScvxWorkspace, TerminalCondition,
};

/// Status returned by the FFI entrypoints. See module docs for codes.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScvxStatus {
    Converged    = 0,
    OuterIterCap = 1,
    InnerFailure = 2,
    Infeasible   = 3,
    BadInput     = 4,
    NullPointer  = 255,
}

impl From<SolverStatus> for ScvxStatus {
    fn from(s: SolverStatus) -> Self {
        match s {
            SolverStatus::Converged    => Self::Converged,
            SolverStatus::OuterIterCap => Self::OuterIterCap,
            SolverStatus::InnerFailure => Self::InnerFailure,
            SolverStatus::Infeasible   => Self::Infeasible,
            SolverStatus::BadInput     => Self::BadInput,
        }
    }
}

/// C-ABI mirror of [`scvx_core::PhysicalParams`].
///
/// All fields are SI units. Match the Rust struct field-for-field.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CPhysicalParams {
    pub g:             [f64; 3],
    pub m_dry:         f64,
    pub m_wet:         f64,
    pub isp:           f64,
    pub g0:            f64,
    pub t_min:         f64,
    pub t_max:         f64,
    pub cos_theta_max: f64,
    pub tan_gamma_gs:  f64,
    pub rho:           f64,
    pub cd_a:          f64,
    pub tau_lo:        f64,
    pub tau_hi:        f64,
}

impl From<&CPhysicalParams> for PhysicalParams {
    fn from(p: &CPhysicalParams) -> Self {
        Self {
            g:             p.g,
            m_dry:         p.m_dry,
            m_wet:         p.m_wet,
            isp:           p.isp,
            g0:            p.g0,
            t_min:         p.t_min,
            t_max:         p.t_max,
            cos_theta_max: p.cos_theta_max,
            tan_gamma_gs:  p.tan_gamma_gs,
            rho:           p.rho,
            cd_a:          p.cd_a,
            tau_lo:        p.tau_lo,
            tau_hi:        p.tau_hi,
        }
    }
}

/// C-ABI terminal target: zero r and v at landing means a soft touch-down.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CTerminalCondition {
    pub r: [f64; 3],
    pub v: [f64; 3],
}

impl From<&CTerminalCondition> for TerminalCondition {
    fn from(t: &CTerminalCondition) -> Self {
        Self { r: t.r, v: t.v }
    }
}

/// C-ABI mirror of [`scvx_solver::PoweredDescentOptions`].
#[repr(C)]
#[derive(Clone, Copy)]
pub struct COptions {
    pub initial_tau:          f64,
    pub target_mass:          f64,
    /// 0 = fixed-tf, non-zero = free-tf.
    pub use_free_tf:          u8,
    /// 0 = disabled, non-zero = enabled.
    pub use_preconditioning:  u8,
    /// 0 = disabled, non-zero = enabled.
    pub use_cone_row_scaling: u8,
    /// 0 = AHO, non-zero = NT. **AHO recommended** (NT is not yet
    /// convergent on all configurations; see HANDOFF.md).
    pub use_nt_scaling:       u8,
    /// 0 = disabled, non-zero = **HSD** (homogeneous self-dual) direction — the
    /// recommended direction; takes precedence over `use_nt_scaling`. Converges
    /// where NT diverges (see HANDOFF.md "Phase 26").
    pub use_hsd:              u8,
    /// 0 = disabled, non-zero = O(N) block-tridiagonal structured solve. With
    /// `use_hsd` non-zero this selects the structured HSD (~7× faster at N=7,
    /// zero fallbacks). Default off.
    pub use_structured_solve: u8,
    pub max_outer_iters:      u32,
    pub max_inner_iters:      u32,
    pub conv_tol_x:           f64,
    pub conv_tol_virt:        f64,
    pub ipm_tol:              f64,
    pub trust_eta0:           f64,
    pub trust_eta_min:        f64,
    pub trust_eta_max:        f64,
    pub virt_weight:          f64,
}

impl From<&COptions> for PoweredDescentOptions {
    fn from(o: &COptions) -> Self {
        Self {
            initial_tau:          o.initial_tau,
            target_mass:          o.target_mass,
            use_free_tf:          o.use_free_tf          != 0,
            use_preconditioning:  o.use_preconditioning  != 0,
            use_cone_row_scaling: o.use_cone_row_scaling != 0,
            use_nt_scaling:       o.use_nt_scaling       != 0,
            use_hsd:              o.use_hsd              != 0,
            use_structured_solve: o.use_structured_solve != 0,
            max_outer_iters:      o.max_outer_iters,
            max_inner_iters:      o.max_inner_iters,
            conv_tol_x:           o.conv_tol_x,
            conv_tol_virt:        o.conv_tol_virt,
            ipm_tol:              o.ipm_tol,
            trust_eta0:           o.trust_eta0,
            trust_eta_min:        o.trust_eta_min,
            trust_eta_max:        o.trust_eta_max,
            virt_weight:          o.virt_weight,
        }
    }
}

// ===========================================================================
// Default constructor (lets C callers get sensible starting values)
// ===========================================================================

/// Populate `*options` with the recommended production defaults.
/// Equivalent to `PoweredDescentOptions::default()` in Rust.
///
/// # Safety
///
/// `options` must be non-null and point to a writable `COptions`.
#[no_mangle]
pub unsafe extern "C" fn scvx_options_default(options: *mut COptions) -> ScvxStatus {
    if options.is_null() {
        return ScvxStatus::NullPointer;
    }
    let d = PoweredDescentOptions::default();
    // SAFETY: caller-asserted non-null + writable.
    unsafe {
        *options = COptions {
            initial_tau:          d.initial_tau,
            target_mass:          d.target_mass,
            use_free_tf:          d.use_free_tf          as u8,
            use_preconditioning:  d.use_preconditioning  as u8,
            use_cone_row_scaling: d.use_cone_row_scaling as u8,
            use_nt_scaling:       d.use_nt_scaling       as u8,
            use_hsd:              d.use_hsd              as u8,
            use_structured_solve: d.use_structured_solve as u8,
            max_outer_iters:      d.max_outer_iters,
            max_inner_iters:      d.max_inner_iters,
            conv_tol_x:           d.conv_tol_x,
            conv_tol_virt:        d.conv_tol_virt,
            ipm_tol:              d.ipm_tol,
            trust_eta0:           d.trust_eta0,
            trust_eta_min:        d.trust_eta_min,
            trust_eta_max:        d.trust_eta_max,
            virt_weight:          d.virt_weight,
        };
    }
    ScvxStatus::Converged // placeholder for "ok"; only error is NullPointer here
}

// ===========================================================================
// Per-N entrypoints
//
// One pair (`scvx_workspace_size_nN`, `scvx_solve_nN`) per supported N,
// for both fixed-tf and free-tf. Supported N ∈ {3, 5, 8, 10, 12, 15, 20}
// (see the `emit_ffi_per_n!` block below). Add more by emitting more macro
// invocations here AND the matching `SCVX_DECLARE_N`/`SCVX_DEFINE_TRAJ` in
// the header.
// ===========================================================================

/// Emit a `#[repr(C)]` C-ABI output trajectory struct `$name` for a fixed
/// node count `$N`. Stored as flat row-major arrays the C side iterates over
/// directly. Layout (must mirror the `double`-array layout in `scvx_ffi.h`):
/// - `r`:     N×3 = [r_0_x, r_0_y, r_0_z, r_1_x, …, r_{N-1}_z]  (position, m)
/// - `v`:     N×3 = same shape as `r`                           (velocity, m/s)
/// - `mass`:  N   = vehicle mass at each node (kg, = exp(z_k))
/// - `u`:     N×3 = thrust/accel vector at each node
/// - `sigma`: N   = thrust-magnitude slack at each node
/// - `tau`:   scalar = time-dilation (s); meaningful for free-tf solves
///
/// One struct per supported `N` (C cannot express const generics, so each
/// `N` gets a concrete type). Adding a new `N` to the FFI is two lines: one
/// `emit_traj_ty!` and one pair of `emit_ffi_per_n!` (fixed + free-tf).
macro_rules! emit_traj_ty {
    ($name:ident, $N:literal) => {
        #[repr(C)]
        pub struct $name {
            pub r:     [[f64; 3]; $N],
            pub v:     [[f64; 3]; $N],
            pub mass:  [f64;       $N],
            pub u:     [[f64; 3]; $N],
            pub sigma: [f64;       $N],
            pub tau:   f64,
        }
    };
}

// Supported node counts. Convergence is validated at N ≤ 5 (see scvx-solver
// tests); larger N is best-effort — the entrypoints always return a status
// (often `OuterIterCap` on hard large-N problems), never crash. A flight
// build typically keeps only the single `N` its mission uses.
emit_traj_ty!(CTrajectoryN3,  3);
emit_traj_ty!(CTrajectoryN5,  5);
emit_traj_ty!(CTrajectoryN8,  8);
emit_traj_ty!(CTrajectoryN10, 10);
emit_traj_ty!(CTrajectoryN12, 12);
emit_traj_ty!(CTrajectoryN15, 15);
emit_traj_ty!(CTrajectoryN20, 20);

/// `MAX_OUTER` baked into the FFI entrypoints. Production callers
/// rarely need more than this; if you do, fork the FFI.
const FFI_MAX_OUTER: usize = 20;

/// Generate the per-N entrypoints (workspace_size, solve) for a given
/// `N` and choice of fixed/free-tf at compile time. The result is two
/// pub `extern "C"` functions per (N, free_tf) combination.
macro_rules! emit_ffi_per_n {
    ($size_fn:ident, $solve_fn:ident, $N:literal, $FREE_TF:literal, $traj_ty:ident) => {
        /// Return the number of bytes needed for the SCvx workspace at
        /// this `N` / free-tf configuration. The buffer must be
        /// `f64`-aligned (8 bytes on every supported target).
        #[no_mangle]
        pub extern "C" fn $size_fn() -> usize {
            const N:      usize = $N;
            const FREE_TF: bool = $FREE_TF;
            const NP:     usize = workspace_np(N, FREE_TF);
            const NE:     usize = workspace_ne(N);
            const NCT:    usize = workspace_nct(N, FREE_TF);
            const NCONES: usize = workspace_ncones(N, FREE_TF);
            const MAX_OUTER: usize = FFI_MAX_OUTER;
            core::mem::size_of::<ScvxWorkspace<N, NP, NE, NCT, NCONES, MAX_OUTER>>()
        }

        /// Solve the SCvx powered-descent problem at this `N`/free-tf
        /// combination.
        ///
        /// # Arguments
        ///
        /// - `phys`: physical parameters (see [`CPhysicalParams`])
        /// - `initial_state`: 7-element array `[r_x, r_y, r_z, v_x, v_y, v_z, ln(m)]`
        /// - `terminal`: target r/v at landing
        /// - `options`: solver knobs (see [`COptions`]); pass
        ///   `scvx_options_default()` output if unsure
        /// - `workspace`: pre-allocated buffer of size
        ///   [`$size_fn`] bytes, 8-byte aligned
        /// - `out_trajectory`: output trajectory (see type docs)
        ///
        /// Returns a [`ScvxStatus`]. The trajectory is only meaningful
        /// when `Converged` or `OuterIterCap`.
        ///
        /// # Safety
        ///
        /// All pointer arguments must be non-null and valid. The
        /// `workspace` pointer must be aligned to 8 and have at least
        /// [`$size_fn()`] bytes of writable memory.
        #[no_mangle]
        pub unsafe extern "C" fn $solve_fn(
            phys:           *const CPhysicalParams,
            initial_state:  *const f64, // length 7
            terminal:       *const CTerminalCondition,
            options:        *const COptions,
            workspace:      *mut u8,
            out_trajectory: *mut $traj_ty,
        ) -> ScvxStatus {
            if phys.is_null() || initial_state.is_null() || terminal.is_null()
                || options.is_null() || workspace.is_null() || out_trajectory.is_null()
            {
                return ScvxStatus::NullPointer;
            }

            // Const generic plumbing — needed inside the alignment check too.
            const N:      usize = $N;
            const FREE_TF: bool = $FREE_TF;
            const NP:     usize = workspace_np(N, FREE_TF);
            const NE:     usize = workspace_ne(N);
            const NCT:    usize = workspace_nct(N, FREE_TF);
            const NCONES: usize = workspace_ncones(N, FREE_TF);
            const MAX_OUTER: usize = FFI_MAX_OUTER;
            type WS = ScvxWorkspace<N, NP, NE, NCT, NCONES, MAX_OUTER>;

            // **Red-team defense**: the workspace buffer holds `f64` fields
            // (via nalgebra SMatrix); writing to a misaligned address is
            // undefined behavior on ARM and on x86 with `repr(align(8))`
            // structures. The C-ABI doc promises 8-byte alignment, which
            // is the minimum f64 alignment on every shipped target.
            // We additionally verify against the actual compiler-chosen
            // alignment of `WS` at runtime, in case a future nalgebra
            // version or SIMD-target chooses a larger alignment (e.g.,
            // 16 or 32 for AVX). The contract degrades gracefully: a
            // caller who provides an 8-aligned buffer is rejected
            // cleanly if WS needs 16-byte alignment, rather than
            // corrupting memory silently.
            let ws_addr = workspace as usize;
            let ws_align = core::mem::align_of::<WS>();
            if ws_addr % ws_align != 0 {
                return ScvxStatus::BadInput;
            }

            // SAFETY: caller asserted non-null + valid.
            let phys_rs    = unsafe { PhysicalParams::from(&*phys) };
            let term_rs    = unsafe { TerminalCondition::from(&*terminal) };
            let options_rs = unsafe { PoweredDescentOptions::from(&*options) };
            let mut x_init = SVector::<f64, 7>::zeros();
            // SAFETY: caller asserted `initial_state` points to 7 f64s.
            // The contract is documented in the C header and module docs;
            // shorter arrays cause OOB reads (no way to detect at the
            // ABI boundary).
            for i in 0..7 {
                x_init[i] = unsafe { *initial_state.add(i) };
            }

            // Validate the unwrapped Rust options before letting the
            // solver consume them — protects against pathological
            // configurations (e.g. negative mass, zero tau bounds) that
            // the deeper `solve_powered_descent` would also reject, but
            // catching at the boundary lets the caller learn from a
            // single status code rather than tracing through inner
            // failures.
            if !x_init.iter().all(|v| v.is_finite())
                || !phys_rs.g.iter().all(|c| c.is_finite())
                || !phys_rs.m_dry.is_finite() || phys_rs.m_dry <= 0.0
                || !phys_rs.m_wet.is_finite() || phys_rs.m_wet <= phys_rs.m_dry
                || !phys_rs.t_min.is_finite() || phys_rs.t_min < 0.0
                || !phys_rs.t_max.is_finite() || phys_rs.t_max <= phys_rs.t_min
                // `Isp·g0` is a divisor in the mass-flow dynamics; zero/neg/NaN
                // injects Inf/NaN into the linearization silently. Reject.
                // `g` (gravity vector) is a dynamics bias — non-finite poisons
                // it the same way, so require finiteness too.
                || !phys_rs.isp.is_finite() || phys_rs.isp <= 0.0
                || !phys_rs.g0.is_finite()  || phys_rs.g0  <= 0.0
                // Drag params feed `v̇` and the STM (continuous.rs / jacobian.rs);
                // a non-finite `rho`/`cd_a` injects Inf/NaN into the dynamics, and
                // a negative value is unphysical anti-drag (energy-adding, sign-
                // flipped `A_vv`) that the solver would silently optimize against.
                || !phys_rs.rho.is_finite()  || phys_rs.rho  < 0.0
                || !phys_rs.cd_a.is_finite() || phys_rs.cd_a < 0.0
                // Cone-shape params feed the pointing / glide-slope cone rows in
                // `assemble_scvx_socp`; non-finite poisons the SOCP `G`/`h`.
                // `cos θ_max ∈ [0, 1]` (real cosine of the pointing half-angle:
                // `0` ⇒ 90° ⇒ `u_z ≥ 0`, a valid degenerate config; `> 1` has no
                // real angle); `tan γ_gs ≥ 0`.
                || !phys_rs.cos_theta_max.is_finite()
                || phys_rs.cos_theta_max < 0.0 || phys_rs.cos_theta_max > 1.0
                || !phys_rs.tan_gamma_gs.is_finite()  || phys_rs.tan_gamma_gs  < 0.0
                || !options_rs.initial_tau.is_finite() || options_rs.initial_tau <= 0.0
                // target_mass MUST be in [m_dry, m_wet]. Outside this band,
                // the SCvx outer loop seeds an infeasible reference and
                // either fails silently (returning a garbage trajectory at
                // OuterIterCap) or wastes iterations chasing an impossible
                // landing mass. Reject early — caller-facing contract is
                // tighter than the solver's internal robustness, by design.
                || !options_rs.target_mass.is_finite()
                || options_rs.target_mass < phys_rs.m_dry
                || options_rs.target_mass > phys_rs.m_wet
                // Validate the free-tf bounds if the option is enabled.
                || (options_rs.use_free_tf && (
                       !phys_rs.tau_lo.is_finite() || phys_rs.tau_lo <= 0.0
                    || !phys_rs.tau_hi.is_finite() || phys_rs.tau_hi <= phys_rs.tau_lo
                    || options_rs.initial_tau < phys_rs.tau_lo
                    || options_rs.initial_tau > phys_rs.tau_hi
                  ))
            {
                return ScvxStatus::BadInput;
            }

            // **Entrypoint contract**: this function's workspace dims are baked
            // for `FREE_TF` at compile time, but the solver branches on the
            // runtime `options.use_free_tf`. If they disagree, the solver would
            // index past the (smaller) fixed-tf workspace — an OOB write. The
            // inner `solve_powered_descent` now also rejects this, but checking
            // here gives the C caller a precise, local contract:
            //   `scvx_solve_nK`         REQUIRES `options.use_free_tf == 0`
            //   `scvx_solve_nK_free_tf` REQUIRES `options.use_free_tf != 0`
            // Note: `scvx_options_default` sets `use_free_tf = 1`, so callers of
            // the fixed-tf entrypoints must clear it explicitly.
            if options_rs.use_free_tf != FREE_TF {
                return ScvxStatus::BadInput;
            }

            // SAFETY: cast pointer + initialize via ptr::write. We
            // initialize the workspace from its `Default` impl —
            // overwriting whatever bytes were in the caller's buffer.
            let ws: &mut WS = unsafe {
                let ptr = workspace.cast::<MaybeUninit<WS>>();
                let uninit = &mut *ptr;
                uninit.write(WS::default());
                (*ptr).assume_init_mut()
            };

            // Run the high-level solver.
            let status = solve_powered_descent(
                ws, &phys_rs, &x_init, &term_rs, &options_rs,
            );

            // Write the output trajectory (always, even on partial
            // convergence — caller checks status before using).
            // SAFETY: caller asserted out_trajectory is non-null and
            // points to a $traj_ty.
            unsafe {
                let out = &mut *out_trajectory;
                for k in 0..N {
                    out.r[k][0]    = ws.reference.x[(0, k)];
                    out.r[k][1]    = ws.reference.x[(1, k)];
                    out.r[k][2]    = ws.reference.x[(2, k)];
                    out.v[k][0]    = ws.reference.x[(3, k)];
                    out.v[k][1]    = ws.reference.x[(4, k)];
                    out.v[k][2]    = ws.reference.x[(5, k)];
                    out.mass[k]    = libm::exp(ws.reference.x[(6, k)]);
                    out.u[k][0]    = ws.reference.u[(0, k)];
                    out.u[k][1]    = ws.reference.u[(1, k)];
                    out.u[k][2]    = ws.reference.u[(2, k)];
                    out.sigma[k]   = ws.reference.sigma[k];
                }
                out.tau = ws.reference.tau;
            }

            // Drop the workspace in place so any Box / heap items in it
            // are released. (Our workspace is currently entirely on-stack
            // / in the caller's buffer, so this is a no-op, but it's
            // future-proof if anything inside grows a Drop impl.)
            // SAFETY: ws is initialized above; we drop it before the
            // function returns. The caller's buffer is then logically
            // uninit again — they must re-call this function or use
            // it as raw bytes only.
            unsafe { ptr::drop_in_place(ws); }

            ScvxStatus::from(status)
        }
    };
}

emit_ffi_per_n!(scvx_workspace_size_n3,         scvx_solve_n3,         3, false, CTrajectoryN3);
emit_ffi_per_n!(scvx_workspace_size_n3_free_tf, scvx_solve_n3_free_tf, 3, true,  CTrajectoryN3);
emit_ffi_per_n!(scvx_workspace_size_n5,         scvx_solve_n5,         5, false, CTrajectoryN5);
emit_ffi_per_n!(scvx_workspace_size_n5_free_tf, scvx_solve_n5_free_tf, 5, true,  CTrajectoryN5);
emit_ffi_per_n!(scvx_workspace_size_n8,          scvx_solve_n8,          8,  false, CTrajectoryN8);
emit_ffi_per_n!(scvx_workspace_size_n8_free_tf,  scvx_solve_n8_free_tf,  8,  true,  CTrajectoryN8);
emit_ffi_per_n!(scvx_workspace_size_n10,         scvx_solve_n10,         10, false, CTrajectoryN10);
emit_ffi_per_n!(scvx_workspace_size_n10_free_tf, scvx_solve_n10_free_tf, 10, true,  CTrajectoryN10);
emit_ffi_per_n!(scvx_workspace_size_n12,         scvx_solve_n12,         12, false, CTrajectoryN12);
emit_ffi_per_n!(scvx_workspace_size_n12_free_tf, scvx_solve_n12_free_tf, 12, true,  CTrajectoryN12);
emit_ffi_per_n!(scvx_workspace_size_n15,         scvx_solve_n15,         15, false, CTrajectoryN15);
emit_ffi_per_n!(scvx_workspace_size_n15_free_tf, scvx_solve_n15_free_tf, 15, true,  CTrajectoryN15);
emit_ffi_per_n!(scvx_workspace_size_n20,         scvx_solve_n20,         20, false, CTrajectoryN20);
emit_ffi_per_n!(scvx_workspace_size_n20_free_tf, scvx_solve_n20_free_tf, 20, true,  CTrajectoryN20);

// ===========================================================================
// Tests (rlib-only — link against the static lib externally)
// ===========================================================================

#[cfg(test)]
mod tests {
    extern crate std;
    use std::{eprintln, thread, vec, vec::Vec};

    use super::*;

    fn run_in_big_stack<F>(f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        thread::Builder::new()
            .stack_size(32 * 1024 * 1024)
            .spawn(f)
            .expect("spawn")
            .join()
            .expect("inner panic");
    }

    fn mars_params() -> CPhysicalParams {
        CPhysicalParams {
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

    /// `scvx_options_default` returns sensible values.
    #[test]
    fn options_default_returns_ok() {
        let mut opts: COptions = unsafe { core::mem::zeroed() };
        // SAFETY: opts is on our stack and writable.
        let status = unsafe { scvx_options_default(&mut opts as *mut COptions) };
        assert_eq!(status, ScvxStatus::Converged);
        assert_eq!(opts.use_free_tf, 1);
        assert_eq!(opts.use_preconditioning, 1);
        // HSD is the promoted production default (Phase 33); structured stays
        // opt-in. A C caller calling scvx_options_default gets HSD.
        assert_eq!(opts.use_hsd, 1);
        assert_eq!(opts.use_structured_solve, 0);
        assert_eq!(opts.use_nt_scaling, 0);
        assert_eq!(opts.max_outer_iters, 15);
    }

    /// Null-pointer arguments yield `NullPointer` status without
    /// panicking. Critical for safety in flight code.
    #[test]
    fn null_inputs_return_null_pointer_status() {
        let status = unsafe { scvx_options_default(core::ptr::null_mut()) };
        assert_eq!(status, ScvxStatus::NullPointer);

        let status = unsafe {
            scvx_solve_n3(
                core::ptr::null(),
                core::ptr::null(),
                core::ptr::null(),
                core::ptr::null(),
                core::ptr::null_mut(),
                core::ptr::null_mut(),
            )
        };
        assert_eq!(status, ScvxStatus::NullPointer);
    }

    /// **Red-team: misaligned workspace must reject with BadInput**, not
    /// silently corrupt memory.
    ///
    /// We construct a 1-byte-aligned buffer (by taking a `&mut u8` and
    /// shifting one byte into it) and pass that as the workspace. The
    /// FFI should detect the misalignment and return BadInput.
    #[test]
    fn misaligned_workspace_rejected() {
        run_in_big_stack(|| {
            let phys = mars_params();
            let initial_state = [0.0_f64, 0.0, 2.0, 0.0, 0.0, -0.1, (400.0_f64).ln()];
            let terminal = CTerminalCondition { r: [0.0; 3], v: [0.0; 3] };
            let mut options: COptions = unsafe { core::mem::zeroed() };
            unsafe { scvx_options_default(&mut options); }
            options.initial_tau = 10.0;
            options.target_mass = 380.0;
            options.use_free_tf = 0;

            let size = scvx_workspace_size_n3();
            // Allocate one byte extra and shift to force misalignment.
            let mut buffer: Vec<u8> = vec![0; size + 8];
            // Find a 1-byte-misaligned position (assuming Vec gives 8-byte
            // alignment as base).
            let base = buffer.as_mut_ptr();
            let misaligned = unsafe { base.add(1) };
            assert_eq!((misaligned as usize) % 8, 1, "did not produce misalignment");
            let mut traj: CTrajectoryN3 = unsafe { core::mem::zeroed() };
            let status = unsafe {
                scvx_solve_n3(
                    &phys as *const _,
                    initial_state.as_ptr(),
                    &terminal as *const _,
                    &options as *const _,
                    misaligned,
                    &mut traj as *mut _,
                )
            };
            assert_eq!(status, ScvxStatus::BadInput,
                       "misaligned workspace should reject with BadInput, got {:?}",
                       status);
        });
    }

    /// **Red-team: pathological inputs rejected at the FFI boundary**.
    /// Negative dry mass, zero tau, m_wet ≤ m_dry — all must come back as
    /// BadInput, not as an inner-solver crash or InnerFailure.
    #[test]
    fn pathological_inputs_rejected() {
        run_in_big_stack(|| {
            let initial_state = [0.0_f64, 0.0, 2.0, 0.0, 0.0, -0.1, (400.0_f64).ln()];
            let terminal = CTerminalCondition { r: [0.0; 3], v: [0.0; 3] };
            let mut options: COptions = unsafe { core::mem::zeroed() };
            unsafe { scvx_options_default(&mut options); }
            options.target_mass = 380.0;
            options.use_free_tf = 0;

            let size = scvx_workspace_size_n3();
            let mut buffer: Vec<u8> = vec![0; size];
            let mut traj: CTrajectoryN3 = unsafe { core::mem::zeroed() };

            // Attack 1: zero initial_tau.
            options.initial_tau = 0.0;
            let mut phys = mars_params();
            let status = unsafe {
                scvx_solve_n3(
                    &phys as *const _, initial_state.as_ptr(),
                    &terminal as *const _, &options as *const _,
                    buffer.as_mut_ptr(), &mut traj as *mut _,
                )
            };
            assert_eq!(status, ScvxStatus::BadInput, "zero tau should reject");

            // Attack 2: m_wet ≤ m_dry (inconsistent).
            options.initial_tau = 10.0;
            phys.m_wet = phys.m_dry; // not strictly greater
            let status = unsafe {
                scvx_solve_n3(
                    &phys as *const _, initial_state.as_ptr(),
                    &terminal as *const _, &options as *const _,
                    buffer.as_mut_ptr(), &mut traj as *mut _,
                )
            };
            assert_eq!(status, ScvxStatus::BadInput, "m_wet = m_dry should reject");

            // Attack 3: t_max ≤ t_min.
            phys.m_wet = 1000.0;
            phys.t_max = phys.t_min;
            let status = unsafe {
                scvx_solve_n3(
                    &phys as *const _, initial_state.as_ptr(),
                    &terminal as *const _, &options as *const _,
                    buffer.as_mut_ptr(), &mut traj as *mut _,
                )
            };
            assert_eq!(status, ScvxStatus::BadInput, "t_max = t_min should reject");

            // Attack 4: NaN in initial_state.
            phys.t_max = 6000.0;
            let mut bad_state = initial_state;
            bad_state[0] = f64::NAN;
            let status = unsafe {
                scvx_solve_n3(
                    &phys as *const _, bad_state.as_ptr(),
                    &terminal as *const _, &options as *const _,
                    buffer.as_mut_ptr(), &mut traj as *mut _,
                )
            };
            assert_eq!(status, ScvxStatus::BadInput, "NaN initial state should reject");

            // Attack 5: target_mass below m_dry (impossible — solver
            // would silently underrun the mass floor).
            let bad_state = initial_state;
            options.target_mass = phys.m_dry - 1.0;
            let status = unsafe {
                scvx_solve_n3(
                    &phys as *const _, bad_state.as_ptr(),
                    &terminal as *const _, &options as *const _,
                    buffer.as_mut_ptr(), &mut traj as *mut _,
                )
            };
            assert_eq!(status, ScvxStatus::BadInput,
                       "target_mass < m_dry should reject");

            // Attack 6: target_mass above m_wet (unphysical — solver
            // would seed a reference with more fuel than the tank holds).
            options.target_mass = phys.m_wet + 1.0;
            let status = unsafe {
                scvx_solve_n3(
                    &phys as *const _, bad_state.as_ptr(),
                    &terminal as *const _, &options as *const _,
                    buffer.as_mut_ptr(), &mut traj as *mut _,
                )
            };
            assert_eq!(status, ScvxStatus::BadInput,
                       "target_mass > m_wet should reject");

            // Attack 7: free-tf enabled but initial_tau outside [tau_lo, tau_hi].
            options.target_mass = 380.0; // restore
            options.use_free_tf = 1;
            options.initial_tau = phys.tau_lo - 0.5; // below floor
            let status = unsafe {
                scvx_solve_n3_free_tf(
                    &phys as *const _, bad_state.as_ptr(),
                    &terminal as *const _, &options as *const _,
                    buffer.as_mut_ptr(), &mut traj as *mut _,
                )
            };
            assert_eq!(status, ScvxStatus::BadInput,
                       "free-tf initial_tau below tau_lo should reject");
        });
    }

    /// **Red-team regression (v1 Critical):** the const-generic workspace
    /// dims are baked for the entrypoint's fixed/free-tf choice, but the
    /// solver branches on the *runtime* `options.use_free_tf`. A mismatch
    /// used to index past the (smaller) fixed-tf workspace → OOB write →
    /// panic/abort (UB if linked `panic=unwind`). The trap: `scvx_options_default`
    /// sets `use_free_tf = 1`, so the most natural usage —
    /// `scvx_options_default(&opts); scvx_solve_n3(...)` *without clearing the
    /// flag* — hit it. Both mismatch directions must now return `BadInput`
    /// cleanly (no panic, no abort).
    #[test]
    fn flag_entrypoint_mismatch_rejected_not_aborted() {
        run_in_big_stack(|| {
            let phys = mars_params();
            let initial_state = [0.0_f64, 0.0, 2.0, 0.0, 0.0, -0.1, (400.0_f64).ln()];
            let terminal = CTerminalCondition { r: [0.0; 3], v: [0.0; 3] };

            // The natural-misuse path: defaults (use_free_tf = 1) + the
            // FIXED-tf entrypoint. Must be BadInput, not an abort.
            let mut options: COptions = unsafe { core::mem::zeroed() };
            unsafe { scvx_options_default(&mut options); }
            options.initial_tau = 10.0;
            options.target_mass = 380.0;
            assert_eq!(options.use_free_tf, 1, "default is free-tf (the trap)");

            let size = scvx_workspace_size_n3();      // fixed-tf-sized buffer
            let mut buffer: Vec<u8> = vec![0; size];
            let mut traj: CTrajectoryN3 = unsafe { core::mem::zeroed() };
            let status = unsafe {
                scvx_solve_n3(
                    &phys as *const _, initial_state.as_ptr(),
                    &terminal as *const _, &options as *const _,
                    buffer.as_mut_ptr(), &mut traj as *mut _,
                )
            };
            assert_eq!(status, ScvxStatus::BadInput,
                       "fixed-tf entrypoint + use_free_tf=1 must reject, got {:?}",
                       status);

            // The converse: free-tf entrypoint with use_free_tf = 0.
            options.use_free_tf = 0;
            let size_ft = scvx_workspace_size_n3_free_tf();
            let mut buffer_ft: Vec<u8> = vec![0; size_ft];
            let mut traj_ft: CTrajectoryN3 = unsafe { core::mem::zeroed() };
            let status = unsafe {
                scvx_solve_n3_free_tf(
                    &phys as *const _, initial_state.as_ptr(),
                    &terminal as *const _, &options as *const _,
                    buffer_ft.as_mut_ptr(), &mut traj_ft as *mut _,
                )
            };
            assert_eq!(status, ScvxStatus::BadInput,
                       "free-tf entrypoint + use_free_tf=0 must reject, got {:?}",
                       status);
        });
    }

    /// **Red-team: `Isp·g0 = 0` (divisor) must reject, not inject NaN.**
    #[test]
    fn zero_isp_or_g0_rejected() {
        run_in_big_stack(|| {
            let initial_state = [0.0_f64, 0.0, 2.0, 0.0, 0.0, -0.1, (400.0_f64).ln()];
            let terminal = CTerminalCondition { r: [0.0; 3], v: [0.0; 3] };
            let mut options: COptions = unsafe { core::mem::zeroed() };
            unsafe { scvx_options_default(&mut options); }
            options.initial_tau = 10.0;
            options.target_mass = 380.0;
            options.use_free_tf = 0;

            let size = scvx_workspace_size_n3();
            let mut buffer: Vec<u8> = vec![0; size];
            let mut traj: CTrajectoryN3 = unsafe { core::mem::zeroed() };

            let mut phys_isp0 = mars_params();
            phys_isp0.isp = 0.0;
            let mut phys_g00 = mars_params();
            phys_g00.g0 = 0.0;
            let mut phys_gnan = mars_params();
            phys_gnan.g[2] = f64::NAN;   // non-finite gravity component

            for (label, phys) in
                [("isp=0", phys_isp0), ("g0=0", phys_g00), ("g=NaN", phys_gnan)]
            {
                let status = unsafe {
                    scvx_solve_n3(
                        &phys as *const _, initial_state.as_ptr(),
                        &terminal as *const _, &options as *const _,
                        buffer.as_mut_ptr(), &mut traj as *mut _,
                    )
                };
                assert_eq!(status, ScvxStatus::BadInput,
                           "{label} should reject with BadInput, got {status:?}");
            }
        });
    }

    /// **End-to-end FFI test**: solve a small Mars problem through the
    /// C-ABI and verify the resulting trajectory is sensible.
    #[test]
    fn ffi_solve_n3_returns_clean_trajectory() {
        run_in_big_stack(|| {
            let phys = mars_params();
            let initial_state = [
                0.0, 0.0, 2.0,        // r = (0, 0, 2 m)
                0.0, 0.0, -0.1,       // v = (0, 0, -0.1 m/s)
                (400.0_f64).ln(),     // log mass for m = 400 kg
            ];
            let terminal = CTerminalCondition { r: [0.0; 3], v: [0.0; 3] };

            let mut options: COptions = unsafe { core::mem::zeroed() };
            unsafe { scvx_options_default(&mut options); }
            // Smaller trust for the small-scale problem.
            options.initial_tau   = 10.0;
            options.target_mass   = 380.0;
            options.trust_eta0    =  5.0;
            options.trust_eta_max = 20.0;
            options.use_free_tf   = 0; // fixed-tf for the simplest path

            // Allocate a workspace buffer via Vec<u8> (heap).
            let size = scvx_workspace_size_n3();
            // Vec is 8-byte aligned for f64 on every Rust target.
            let mut buffer: Vec<u8> = vec![0; size];
            let mut traj: CTrajectoryN3 = unsafe { core::mem::zeroed() };

            let status = unsafe {
                scvx_solve_n3(
                    &phys as *const _,
                    initial_state.as_ptr(),
                    &terminal as *const _,
                    &options as *const _,
                    buffer.as_mut_ptr(),
                    &mut traj as *mut _,
                )
            };

            eprintln!("FFI status: {:?}", status);
            eprintln!("FFI trajectory:");
            for k in 0..3 {
                eprintln!(
                    "  k={k}: r=({:>+.3}, {:>+.3}, {:>+.3}), v=({:>+.3}, {:>+.3}, {:>+.3}), m={:.2}, σ={:.1}",
                    traj.r[k][0], traj.r[k][1], traj.r[k][2],
                    traj.v[k][0], traj.v[k][1], traj.v[k][2],
                    traj.mass[k], traj.sigma[k],
                );
            }
            eprintln!("  τ = {:.3} s", traj.tau);

            // Status: not BadInput.
            assert_ne!(status, ScvxStatus::BadInput);
            assert_ne!(status, ScvxStatus::NullPointer);

            // Trajectory must be finite. (On InnerFailure the contents
            // are stale-but-finite; we just need to verify no NaN/inf
            // leaks out.)
            for k in 0..3 {
                for i in 0..3 {
                    assert!(traj.r[k][i].is_finite());
                    assert!(traj.v[k][i].is_finite());
                    assert!(traj.u[k][i].is_finite());
                }
                assert!(traj.mass[k].is_finite() && traj.mass[k] > 0.0);
                assert!(traj.sigma[k].is_finite());
            }
            assert!(traj.tau.is_finite() && traj.tau > 0.0);
        });
    }

    /// **Smoke test for the expanded N surface (v1).** The N=8 and N=10
    /// entrypoints (added alongside N=3/5) must run end-to-end through the
    /// C-ABI: return a non-error status and emit a fully finite trajectory.
    /// Convergence is NOT asserted (validated only at N ≤ 5) — this guards
    /// that the new monomorphizations are wired correctly and never crash or
    /// leak NaN/Inf.
    #[test]
    fn ffi_solve_n8_n10_run_and_stay_finite() {
        run_in_big_stack(|| {
            let phys = mars_params();
            let initial_state = [
                0.0, 0.0, 2.0,
                0.0, 0.0, -0.1,
                (400.0_f64).ln(),
            ];
            let terminal = CTerminalCondition { r: [0.0; 3], v: [0.0; 3] };
            let mut options: COptions = unsafe { core::mem::zeroed() };
            unsafe { scvx_options_default(&mut options); }
            options.initial_tau = 10.0;
            options.target_mass = 380.0;
            options.use_free_tf = 0;

            // ---- N=8 ----
            let size8 = scvx_workspace_size_n8();
            let mut buf8: Vec<u8> = vec![0; size8];
            let mut traj8: CTrajectoryN8 = unsafe { core::mem::zeroed() };
            let st8 = unsafe {
                scvx_solve_n8(
                    &phys as *const _, initial_state.as_ptr(),
                    &terminal as *const _, &options as *const _,
                    buf8.as_mut_ptr(), &mut traj8 as *mut _,
                )
            };
            eprintln!("N=8 FFI status: {st8:?}");
            assert_ne!(st8, ScvxStatus::BadInput);
            assert_ne!(st8, ScvxStatus::NullPointer);
            for k in 0..8 {
                for i in 0..3 {
                    assert!(traj8.r[k][i].is_finite() && traj8.v[k][i].is_finite()
                            && traj8.u[k][i].is_finite());
                }
                assert!(traj8.mass[k].is_finite() && traj8.mass[k] > 0.0);
                assert!(traj8.sigma[k].is_finite());
            }
            assert!(traj8.tau.is_finite() && traj8.tau > 0.0);

            // ---- N=10 ----
            let size10 = scvx_workspace_size_n10();
            let mut buf10: Vec<u8> = vec![0; size10];
            let mut traj10: CTrajectoryN10 = unsafe { core::mem::zeroed() };
            let st10 = unsafe {
                scvx_solve_n10(
                    &phys as *const _, initial_state.as_ptr(),
                    &terminal as *const _, &options as *const _,
                    buf10.as_mut_ptr(), &mut traj10 as *mut _,
                )
            };
            eprintln!("N=10 FFI status: {st10:?}");
            assert_ne!(st10, ScvxStatus::BadInput);
            assert_ne!(st10, ScvxStatus::NullPointer);
            for k in 0..10 {
                for i in 0..3 {
                    assert!(traj10.r[k][i].is_finite() && traj10.v[k][i].is_finite()
                            && traj10.u[k][i].is_finite());
                }
                assert!(traj10.mass[k].is_finite() && traj10.mass[k] > 0.0);
                assert!(traj10.sigma[k].is_finite());
            }
            assert!(traj10.tau.is_finite() && traj10.tau > 0.0);
        });
    }

    /// **FFI smoke — the HSD direction is reachable through the C-ABI
    /// (promotion checklist #1).** `COptions::use_hsd` (+ `use_structured_solve`)
    /// now thread `COptions → PoweredDescentOptions → IpmAlgoParams/
    /// ScvxAlgoParams → solve_scvx` dispatch. A C caller can select BOTH the
    /// dense HSD and the O(N) structured HSD and get a finite trajectory back
    /// (no BadInput / NullPointer / abort).
    #[test]
    fn ffi_solve_with_hsd_runs_and_stays_finite() {
        run_in_big_stack(|| {
            let phys = mars_params();
            let initial_state = [0.0, 0.0, 2.0, 0.0, 0.0, -0.1, (400.0_f64).ln()];
            let terminal = CTerminalCondition { r: [0.0; 3], v: [0.0; 3] };

            // (a) dense HSD, then (b) the O(N) structured HSD — both via use_hsd.
            for (label, structured) in [("dense-HSD", 0u8), ("structured-HSD", 1u8)] {
                let mut options: COptions = unsafe { core::mem::zeroed() };
                unsafe { scvx_options_default(&mut options); }
                options.initial_tau = 10.0;
                options.target_mass = 380.0;
                options.use_free_tf = 0;
                options.use_hsd = 1;
                options.use_structured_solve = structured;

                let size8 = scvx_workspace_size_n8();
                let mut buf8: Vec<u8> = vec![0; size8];
                let mut traj8: CTrajectoryN8 = unsafe { core::mem::zeroed() };
                let st8 = unsafe {
                    scvx_solve_n8(
                        &phys as *const _, initial_state.as_ptr(),
                        &terminal as *const _, &options as *const _,
                        buf8.as_mut_ptr(), &mut traj8 as *mut _,
                    )
                };
                eprintln!("FFI {label} N=8 status: {st8:?}");
                assert_ne!(st8, ScvxStatus::BadInput, "{label}: HSD via FFI returned BadInput");
                assert_ne!(st8, ScvxStatus::NullPointer);
                for k in 0..8 {
                    for i in 0..3 {
                        assert!(traj8.r[k][i].is_finite() && traj8.v[k][i].is_finite()
                                && traj8.u[k][i].is_finite(), "{label}: non-finite at k={k}");
                    }
                    assert!(traj8.mass[k].is_finite() && traj8.mass[k] > 0.0);
                    assert!(traj8.sigma[k].is_finite());
                }
                assert!(traj8.tau.is_finite() && traj8.tau > 0.0);
            }
            // NOTE: HSD spans the full FFI N range (3..20), but at large N the
            // dense NCT×NCT scaling matrices the inner solve builds on the stack
            // (NCT = 30·N → ~2.9 MB each at N=20) require a correspondingly large
            // thread stack — a general property of ALL directions (AHO/NT/HSD),
            // not HSD-specific. The committed N≤10 coverage here + the
            // `scvx_converges_with_hsd_larger_n` (N=10) gate + the O(N) time
            // benchmark establish the scaling; the production integrator sizes the
            // static arena / stack for the chosen max N (see INTEGRATION.md).
        });
    }
}
