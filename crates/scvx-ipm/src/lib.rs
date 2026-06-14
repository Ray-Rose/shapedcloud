#![no_std]
// Crate currently uses **zero** `unsafe` code (verified by `grep`); we use
// `deny(unsafe_op_in_unsafe_fn)` rather than `forbid(unsafe_code)` so a
// future P3b lift could selectively add audited unsafe to the Riccati
// inner loop after `cargo asm` confirms bounds checks survive `-O3`.
// `forbid` would block that future option; `deny` keeps a non-zero `unsafe`
// from sneaking in elsewhere while preserving the optimization escape hatch.
#![deny(unsafe_op_in_unsafe_fn)]

pub mod block_tridiag;
pub mod cone;
pub mod kkt;
pub mod mehrotra;
pub mod socp;

pub use block_tridiag::*;
pub use cone::*;
pub use kkt::*;
pub use mehrotra::*;
pub use socp::*;
