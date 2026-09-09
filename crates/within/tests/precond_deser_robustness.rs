//! Deserializing untrusted bytes into a [`Preconditioner`] must return a typed
//! error, never panic or read out of bounds — the pickle path accepts bytes
//! that may originate outside the producing process. Exhaustive coverage lives
//! in the `preconditioner_from_bytes` fuzz target (see `fuzz/`); this pins the
//! specific inputs a stable-toolchain CI job can guard without a fuzzer.

use rstest::rstest;
use within::Preconditioner;

// The fuzzer's first find: a serialized Schwarz preconditioner whose `n_dofs`
// sat below its covered subdomain index span — formerly a debug-assert panic
// and, in release, an out-of-bounds subdomain scatter at apply time.
const CRASH_NDOFS_SPAN: &[u8] = include_bytes!("fixtures/precond_deser_crash_ndofs_span.bin");

// A `BlockElimSolver` whose reduced factor is a `Cover` nested inside another
// `Cover` — a shape no real build produces. Deriving `Deserialize` let it
// decode, then `scratch_len` recursed down the chain until the stack overflowed
// (#166). The wire format now decodes a cover's inner factor as a leaf, so the
// nested discriminant fails to decode instead.
const CRASH_NESTED_COVER: &[u8] = include_bytes!("fixtures/precond_deser_crash_nested_cover.bin");

/// Crash fixtures and malformed byte strings alike must decode to a typed error, never panic.
#[rstest]
#[case::n_dofs_below_span(CRASH_NDOFS_SPAN)]
#[case::nested_cover(CRASH_NESTED_COVER)]
#[case::empty(&[])]
#[case::one_zero_byte(&[0x00])]
#[case::saturated_8(&[0xFF; 8])]
#[case::saturated_64(&[0xFF; 64])]
#[case::truncated_half(&CRASH_NDOFS_SPAN[..CRASH_NDOFS_SPAN.len() / 2])]
#[case::truncated_to_one_byte(&CRASH_NDOFS_SPAN[..1])]
fn deserializing_untrusted_bytes_returns_error_not_panic(#[case] bytes: &[u8]) {
    assert!(postcard::from_bytes::<Preconditioner>(bytes).is_err());
}
