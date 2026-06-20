//! Wire-format byte fixture for the serialized [`Preconditioner`]: deserializes
//! a payload from a known-good build and drives a solve, catching silent
//! cross-version encoding shifts the same-build round-trip test cannot.
//! Regenerate via the `#[ignore]`d `regenerate_wire_format_fixture` test.

use ndarray::array;
use within::{LsmrOptions, Preconditioner, Solver};

const WIRE_FORMAT_VERSION: u32 = 3;
const PRECOND_BYTES: &[u8] = include_bytes!("fixtures/preconditioner_v3.postcard");

fn fixture_problem() -> (ndarray::Array2<u32>, Vec<f64>) {
    // Fixed-effects problem with two factors. Small enough to keep the
    // fixture compact, large enough to exercise multi-subdomain Schwarz.
    let categories = array![
        [0u32, 0],
        [1, 0],
        [0, 1],
        [1, 1],
        [2, 0],
        [2, 1],
        [0, 0],
        [1, 1],
    ];
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    (categories, y)
}

#[test]
fn wire_format_fixture_deserializes_and_solves() {
    let _ = WIRE_FORMAT_VERSION;

    let (categories, y) = fixture_problem();
    let prebuilt: Preconditioner =
        postcard::from_bytes(PRECOND_BYTES).expect("deserialize fixture preconditioner");

    let solver = Solver::new(categories.view(), None, prebuilt).expect("build solver from fixture");
    let result = solver
        .solve(&y, &LsmrOptions::default())
        .expect("solve with fixture preconditioner");

    assert!(result.converged, "fixture-built solver should converge");

    // Compare against a fresh build to detect any semantic regression.
    let fresh = Solver::new(categories.view(), None, None).expect("fresh solver");
    let fresh_result = fresh
        .solve(&y, &LsmrOptions::default())
        .expect("fresh solve");
    for (a, b) in result.x.iter().zip(fresh_result.x.iter()) {
        assert!(
            (a - b).abs() < 1e-9,
            "fixture vs fresh coefficient drift: {} vs {}",
            a,
            b,
        );
    }
}

/// Generate the wire-format fixture. Run with `--ignored` to overwrite
/// `crates/within/tests/fixtures/preconditioner_v3.postcard`. Intended for
/// intentional wire-format bumps only; CI runs the non-ignored test above.
#[test]
#[ignore]
fn regenerate_wire_format_fixture() {
    use std::io::Write;
    use std::path::PathBuf;

    let (categories, _) = fixture_problem();
    let solver = Solver::new(categories.view(), None, None).expect("build solver");
    let prec = solver
        .preconditioner()
        .expect("default solver has a preconditioner");
    let bytes = postcard::to_stdvec(prec).expect("serialize");

    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/fixtures/preconditioner_v3.postcard");
    let mut f = std::fs::File::create(&path).expect("create fixture file");
    f.write_all(&bytes).expect("write fixture bytes");
    eprintln!("wrote {} bytes to {}", bytes.len(), path.display());
}
