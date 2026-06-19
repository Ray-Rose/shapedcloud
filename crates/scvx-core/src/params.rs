/// Physical parameters for 3-DoF powered descent with aerodynamic drag.
#[derive(Clone, Copy)]
pub struct PhysicalParams {
    pub g:             [f64; 3], // gravity vector, inertial frame, m/s²
    pub m_dry:         f64,      // dry mass, kg
    pub m_wet:         f64,      // initial mass upper bound, kg
    pub isp:           f64,      // specific impulse, s
    pub g0:            f64,      // standard gravity for Isp normalization, m/s²
    /// Thrust-magnitude lower bound, N (the `σ ≥ T_min` cone). **Scale this to
    /// your gravity regime** — set it near the engine's true deep-throttle floor
    /// (often ~5–10% of `t_max`), not a value copied from a different-gravity
    /// mission. If `t_min` exceeds the vehicle's weight in the local gravity
    /// (hover thrust near or below `t_min`), the floor cone rides hard against
    /// its boundary and stresses the IPM endgame, hurting convergence — a
    /// modeling mismatch, not a solver bug. E.g. Mars `1000` is too high for the
    /// same vehicle on the Moon (lower to ~`300`). See HANDOFF "Phase 17".
    pub t_min:         f64,      // N
    pub t_max:         f64,      // thrust magnitude upper bound, N
    pub cos_theta_max: f64,      // cos of thrust pointing half-angle
    pub tan_gamma_gs:  f64,      // tan of glide-slope angle (from horizontal)
    pub rho:           f64,      // atmospheric density, kg/m³ (constant in v1)
    pub cd_a:          f64,      // C_d · A, m² (drag area)
    pub tau_lo:        f64,      // free-final-time lower bound, s
    pub tau_hi:        f64,      // free-final-time upper bound, s
}

/// SCvx outer-loop algorithmic parameters.
#[derive(Clone, Copy)]
pub struct ScvxAlgoParams {
    pub max_outer_iters: u32,
    pub trust_eta0:      f64,
    pub trust_eta_min:   f64,
    pub trust_eta_max:   f64,
    pub trust_alpha:     f64, // shrink factor on bad step
    pub trust_beta:      f64, // grow factor on great step
    pub rho_reject:      f64, // reject step if ρ < ρ_reject
    pub rho_shrink:      f64,
    pub rho_grow:        f64,
    pub virt_weight:     f64, // virtual-control L1 penalty weight
    pub conv_tol_x:      f64, // ‖dx‖+‖du‖ tolerance
    pub conv_tol_virt:   f64, // ‖v‖₁ tolerance
    /// If `true`, the SCvx subproblem includes `δτ` (time dilation
    /// adjustment) as a SOCP decision variable, bounded by
    /// `[tau_lo, tau_hi]` from `PhysicalParams`. The candidate
    /// trajectory's `τ` is updated each iteration to `τ_ref + δτ_opt`.
    /// Required for problems where the landing time isn't known a priori.
    /// When `false` (default), `τ` is held at the user-provided
    /// `reference.tau` throughout (fixed-final-time).
    ///
    /// **Layout impact**: when set, callers must pass `NP = 19·N + 1`,
    /// `NCT = 30·N + 2`, `NCONES = 8·N + 2` (one extra primal variable
    /// for `δτ`, two extra SOC^1 bound cones for `tau_lo ≤ τ ≤ tau_hi`).
    pub use_free_tf:     bool,
    /// If `true`, the SCvx outer loop dispatches the inner SOCP solve to the
    /// structured block-tridiagonal-Schur driver (`O(N·NZ³)`) instead of the
    /// dense one (`O((N·NZ)³)`). All four configurations are supported — the
    /// dispatch picks the matching structured driver:
    /// `solve_socp_structured` (AHO/fixed-tf), `_free_tf` (AHO/free-tf),
    /// `_nt` (NT/fixed-tf), `_nt_free_tf` (NT/free-tf) — and on any
    /// per-iteration breakdown falls back to the **direction-matched** dense
    /// driver (`solve_socp` / `solve_socp_nt`) for that iteration.
    ///
    /// Defaults to `false` (production-safe: dense is the hardened path, and
    /// the structured path does not yet win end-to-end — fallback erosion in
    /// the AHO endgame makes it a wash on wall-clock; see HANDOFF "structured
    /// fallback"). The structured solve is per-step numerically equivalent to
    /// dense (verified to machine precision); multi-iter trajectories may
    /// drift by ≤ 1e-3 from fp roundoff but track the same fixed point.
    pub use_structured_solve: bool,
    /// If `true` (default), the trust-region grow/shrink thresholds ADAPT to
    /// the problem's achievable merit-ρ instead of using fixed `rho_shrink`/
    /// `rho_grow`. Rationale: the merit ρ is actual/predicted where `predicted`
    /// uses the LINEARIZED dynamics and `actual` re-propagates the NONLINEAR
    /// dynamics; that gap caps the achievable ρ at ≈0.1–0.2 on flight-scale
    /// problems, so the textbook fixed thresholds (0.25/0.7) would never let
    /// the trust grow and it collapses (‖ν‖ freezes at ~0.2). Adaptive trust
    /// tracks an EMA of accepted-step ρ as a running "achievable-ρ ceiling"
    /// (initialized to 1.0, so it STARTS conservative) and derives the
    /// effective thresholds as fractions of that ceiling, capped by the
    /// configured `rho_shrink`/`rho_grow`. Well-conditioned problems (ρ→1)
    /// keep the conservative thresholds; linearization-gap-capped problems
    /// auto-relax toward the lenient values that converge — so larger-N
    /// converges WITHOUT manual tuning and small-N is not destabilized.
    /// Set `false` to use the fixed `rho_shrink`/`rho_grow` directly.
    pub use_adaptive_trust: bool,
}

impl Default for ScvxAlgoParams {
    fn default() -> Self {
        Self {
            max_outer_iters: 15,
            trust_eta0:      1.0,
            trust_eta_min:   1.0e-6,
            trust_eta_max:   10.0,
            trust_alpha:     2.0,
            trust_beta:      2.0,
            rho_reject:      0.0,
            // Trust-region grow/shrink thresholds. These conservative
            // (textbook) values suit small-scale problems, where ρ→1 near the
            // solution. NOTE: for LARGER / flight-scale problems they are too
            // aggressive — the merit ρ is actual/predicted where `predicted`
            // uses the LINEARIZED dynamics and `actual` re-propagates the true
            // NONLINEAR dynamics, and that linearization gap caps the achievable
            // ρ at ≈0.1–0.2, so the trust radius can never grow and collapses
            // monotonically (the dynamics defect ‖ν‖ then freezes at ~0.2). For
            // those problems set `rho_shrink≈0.05, rho_grow≈0.1` explicitly —
            // that drove the N=10 / 100 m defect from 2.1e-1 to 6.2e-11. See
            // the `scvx_converges_larger_n_*` test and the HANDOFF "larger-N"
            // notes. (Not made the default: those thresholds destabilize some
            // small-scale/structured configs, which is unacceptable for the
            // conservative default.)
            rho_shrink:      0.25,
            rho_grow:        0.7,
            virt_weight:     1.0e5,
            conv_tol_x:      1.0e-3,
            conv_tol_virt:   1.0e-7,
            use_free_tf:     false,
            use_structured_solve: false,
            use_adaptive_trust:   true,
        }
    }
}

/// Inner IPM algorithmic parameters.
#[derive(Clone, Copy)]
pub struct IpmAlgoParams {
    pub max_iters:        u32,
    pub tol_mu:           f64,
    pub tol_primal:       f64,
    pub tol_dual:         f64,
    pub tol_gap:          f64,
    pub refine_thresh:    f64,
    pub max_refine_iters: u32,
    /// Relative Tikhonov factor for the **adaptive** reduced-Hessian
    /// regularization. When `use_adaptive_regularization` is set, the dense
    /// IPM uses `reg = max(1e-8, regularization · tr(H)/n)`; when it is unset
    /// (the default) the fixed `1e-8` floor is used and this field has no
    /// effect. A non-finite or negative value falls back to the `1e-10`
    /// default. Default `1e-10`.
    pub regularization:   f64,
    /// If `true`, `solve_socp` does NOT reset `ws.x` and `ws.lambda` at
    /// entry — caller is responsible for pre-seeding them with a good
    /// initial point. Defaults to `false` (legacy behavior: cold start
    /// from x = 0, λ = 0). The SCvx outer loop sets this `true` and
    /// initializes the workspace at the reference trajectory, giving the
    /// inner IPM a near-primal-feasible starting point.
    pub warm_start_x:     bool,
    /// If `true`, use the Nesterov-Todd-direction IPM (`solve_socp_nt`)
    /// instead of the default AHO direction (`solve_socp`). NT gives a
    /// symmetric PD reduced Hessian and avoids AHO's endgame degeneracy
    /// near the cone boundary. Recommended for SCvx subproblems.
    pub use_nt_scaling:   bool,
    /// If `true`, the SCvx outer loop applies per-variable diagonal
    /// preconditioning to the assembled SOCP before handing it to the
    /// inner IPM. Required for NT-direction convergence on flight-scale
    /// subproblems (cones spanning ~6 orders of magnitude). Has no effect
    /// outside `scvx-solver::solve_scvx` — the standalone IPM stays
    /// oblivious. Defaults to `false` (backward-compat: existing oracle-
    /// diff / regression tests solve raw, unscaled SOCPs).
    pub use_preconditioning: bool,
    /// If `true`, the inner IPM uses adaptive Tikhonov regularization on
    /// the reduced Hessian: `reg = max(1e-8, 1e-10 · tr(H)/n)` instead of
    /// the fixed `1e-8`. Required for preconditioned problems where `H'`
    /// can have diagonal entries ~10^7 (the fixed floor becomes negligible).
    /// Defaults to `false` because the adaptive value grows when the
    /// iterates approach the cone boundary (where `arrow(s)^{-1}` blows
    /// up), and can over-regularize unscaled problems that were tuned
    /// for the fixed floor. Set to `true` whenever `use_preconditioning
    /// = true`; rarely useful otherwise.
    pub use_adaptive_regularization: bool,
    /// If `true`, the SCvx outer loop applies per-cone slack rescaling
    /// (normalizes cone slacks `s = h − G·x` to ~unit magnitude per cone
    /// by dividing the corresponding rows of `G` and entries of `h`).
    /// Complements `use_preconditioning` (which only rescales the primal):
    /// what NT's `arrow(s)^{-1}` actually sees is the slack vector, so
    /// row-scaling is what directly normalizes the IPM's per-cone
    /// conditioning. Required to make NT converge on flight-scale SCvx
    /// subproblems where cone slack magnitudes span ~6 orders. Has no
    /// effect outside `scvx-solver::solve_scvx`. Defaults to `false`.
    pub use_cone_row_scaling:    bool,
    /// If `true`, the inner solve uses the **homogeneous self-dual (HSD)
    /// embedded** driver (`solve_socp_hsd`) instead of the AHO/NT directions.
    /// HSD converges on the flight-scale SCvx subproblem where the plain NT
    /// direction DIVERGES (the virtual-control cones vanish at the optimum),
    /// matching the external CVXPY/Clarabel + Julia oracle far tighter and faster
    /// than even AHO — see HANDOFF "Phase 26". It cold-starts from the self-dual
    /// central point, so it IGNORES `warm_start_x` and any seeded `ws.x`, and is
    /// dimension-generic over fixed-/free-tf (the `δτ` variable and `τ`-bound
    /// cones are handled transparently). When set, it takes PRECEDENCE over
    /// `use_nt_scaling`; combined with `ScvxAlgoParams::use_structured_solve` it
    /// dispatches to the **O(N)** block-tridiagonal structured HSD
    /// (`solve_socp_structured_hsd` / `_free_tf`, Phases 28–29) with a dense-HSD
    /// fallback, else the dense `solve_socp_hsd`. Defaults to `false` *at this
    /// low level* so direct-IPM callers and regression tests keep the AHO
    /// reference behavior — but the **product default is now HSD**: the high-level
    /// `PoweredDescentOptions::default()` and the FFI `scvx_options_default` set
    /// this `true` (Phase 33 promotion; HANDOFF Phases 31–33).
    pub use_hsd: bool,
}

impl Default for IpmAlgoParams {
    fn default() -> Self {
        Self {
            max_iters:        25,
            tol_mu:           1.0e-8,
            tol_primal:       1.0e-7,
            tol_dual:         1.0e-7,
            tol_gap:          1.0e-7,
            refine_thresh:    1.0e-10,
            max_refine_iters: 1,
            regularization:   1.0e-10,
            warm_start_x:                false,
            use_nt_scaling:              false,
            use_preconditioning:         false,
            use_adaptive_regularization: false,
            use_cone_row_scaling:        false,
            use_hsd:                     false,
        }
    }
}

pub const G0_EARTH: f64 = 9.80665;
pub const G_MARS:   f64 = 3.7114;
pub const G_LUNA:   f64 = 1.62;
