use nalgebra::{SMatrix, SVector};

/// State + control trajectory + free-final-time scalar.
///
/// `N` is the number of temporal nodes. State is `[r(3), v(3), z=ln(m)] ∈ ℝ⁷`,
/// control is the thrust vector `u ∈ ℝ³`, plus a per-node thrust-magnitude
/// slack `σ` (the LCvx slack variable; `‖u‖ ≤ σ`, `Tmin ≤ σ ≤ Tmax`).
/// `tau` is the time-dilation scalar (free-final-time).
#[derive(Clone)]
pub struct Trajectory<const N: usize> {
    pub x:     SMatrix<f64, 7, N>,
    pub u:     SMatrix<f64, 3, N>,
    pub sigma: SVector<f64, N>,
    pub tau:   f64,
}

impl<const N: usize> Default for Trajectory<N> {
    fn default() -> Self {
        Self {
            x:     SMatrix::zeros(),
            u:     SMatrix::zeros(),
            sigma: SVector::zeros(),
            tau:   1.0,
        }
    }
}
