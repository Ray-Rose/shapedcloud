#![no_std]
#![forbid(unsafe_code)]

pub mod api;
pub mod assemble;
pub mod precondition;
pub mod reduced_kkt;
pub mod scvx;
pub mod structured_socp;
pub mod trust;

pub use api::*;
pub use assemble::*;
pub use precondition::*;
pub use reduced_kkt::*;
pub use scvx::*;
pub use structured_socp::*;
