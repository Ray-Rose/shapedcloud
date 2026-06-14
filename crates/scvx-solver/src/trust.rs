//! Trust-region update helpers.
//!
//! The original P7 plan was for `rho_ratio` and `update_trust` to live here
//! as standalone helpers. In practice the SCvx outer loop in `scvx.rs`
//! inlines this logic: the (now-real) LM ρ-ratio computed against the
//! nonlinear forward integration, the `clamp(eta_min, eta_max)` policy,
//! and the accept/reject branching all ended up coupled too tightly to
//! the outer-loop state to factor cleanly.
//!
//! Kept as a placeholder so the module path resolves. If the trust-region
//! logic ever needs to be reused (e.g., by an alternate outer-loop variant
//! like Sequential Quadratic Programming), this is where the factored
//! helpers should land.
