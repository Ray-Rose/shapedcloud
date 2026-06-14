/// Top-level SCvx solver status. Plain enum; no panic on any path.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum SolverStatus {
    Converged,
    OuterIterCap,
    InnerFailure,
    Infeasible,
    BadInput,
}

/// Inner IPM termination status.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum IpmStatus {
    Optimal,
    BestFeasible,
    Infeasible,
    NumericalError,
    IterCap,
}

impl IpmStatus {
    pub fn as_u32(self) -> u32 {
        match self {
            IpmStatus::Optimal        => 0,
            IpmStatus::BestFeasible   => 1,
            IpmStatus::Infeasible     => 2,
            IpmStatus::NumericalError => 3,
            IpmStatus::IterCap        => 4,
        }
    }
}

/// Per-outer-iteration record. Plain data; the integrator decides how to
/// surface this (log, telemetry, throw away). No `Debug` derive in flight
/// crates — that pulls `core::fmt` into flash.
#[derive(Clone, Copy, Default)]
pub struct ScvxIterRecord {
    pub iter:       u32,
    pub cost:       f64,
    pub trust_eta:  f64,
    pub virt_l1:    f64,
    pub rho_ratio:  f64,
    pub accepted:   bool,
    pub ipm_status: u32, // IpmStatus as u32
    pub ipm_iters:  u32,
}
