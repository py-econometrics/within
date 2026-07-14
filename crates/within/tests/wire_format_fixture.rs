//! Wire-format byte fixture for the serialized [`Preconditioner`]: deserializes
//! a payload from a known-good build and drives a solve, catching silent
//! cross-version encoding shifts the same-build round-trip test cannot.
//! Regenerate via the `#[ignore]`d `regenerate_wire_format_fixture` test.

use within::{Effect, LsmrOptions, Preconditioner, PreconditionerConfig, Solver};

const WIRE_FORMAT_VERSION: u32 = 6;
const PRECOND_BYTES: &[u8] = include_bytes!("fixtures/preconditioner_v6.postcard");
const PRE_BUMP_BYTES: &[u8] = include_bytes!("fixtures/preconditioner_v5.postcard");

fn fixture_problem() -> (Vec<u32>, Vec<u32>, Vec<f64>, Vec<f64>) {
    // Slope term against a three-level factor: the (f-slope, g) pair is
    // frustrated (negative 4-cycle), so the payload pins a single signed
    // operator with Scaled coordinates and a Cover reduced factor, alongside
    // the plain (f-int, g) pair's Canonical ones.
    let f = vec![0u32, 0, 0, 1, 1, 1];
    let g = vec![0u32, 1, 2, 0, 1, 2];
    let z = vec![-2.0, 1.0, 1.0, -1.0, -1.0, 2.0];
    let y = vec![1.0, -2.0, 0.5, 3.0, -1.5, 2.5];
    (f, g, z, y)
}

fn fixture_effects<'a>(f: &'a [u32], g: &'a [u32], z: &'a [f64]) -> Vec<Effect<'a>> {
    vec![
        Effect::new(f, true, [z]).expect("slope effect"),
        Effect::new(g, true, []).expect("plain effect"),
    ]
}

#[test]
fn wire_format_fixture_deserializes_and_solves() {
    let _ = WIRE_FORMAT_VERSION;

    let (f, g, z, y) = fixture_problem();
    let prebuilt: Preconditioner =
        postcard::from_bytes(PRECOND_BYTES).expect("deserialize fixture preconditioner");

    let solver = Solver::new(fixture_effects(&f, &g, &z), None, prebuilt)
        .expect("build solver from fixture");
    let result = solver
        .solve(&y, &LsmrOptions::default())
        .expect("solve with fixture preconditioner");

    assert!(result.converged, "fixture-built solver should converge");

    // Compare against a fresh build to detect any semantic regression.
    let fresh = Solver::new(
        fixture_effects(&f, &g, &z),
        None,
        PreconditionerConfig::default(),
    )
    .expect("fresh solver");
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

#[test]
fn signed_route_preconditioner_round_trips() {
    let (f, g, z, y) = fixture_problem();
    let solver1 = Solver::new(
        fixture_effects(&f, &g, &z),
        None,
        PreconditionerConfig::default(),
    )
    .expect("build solver");
    let r1 = solver1.solve(&y, &LsmrOptions::default()).expect("solve 1");

    let bytes = postcard::to_stdvec(solver1.preconditioner().expect("has preconditioner"))
        .expect("serialize");
    let restored: Preconditioner = postcard::from_bytes(&bytes).expect("deserialize");

    let solver2 =
        Solver::new(fixture_effects(&f, &g, &z), None, restored).expect("solver from round-trip");
    let r2 = solver2.solve(&y, &LsmrOptions::default()).expect("solve 2");

    for (a, b) in r1.x.iter().zip(r2.x.iter()) {
        assert!(
            (a - b).abs() < 1e-12,
            "round-trip coefficient mismatch: {} vs {}",
            a,
            b,
        );
    }
}

#[test]
fn pre_bump_fixture_no_longer_decodes() {
    assert!(postcard::from_bytes::<Preconditioner>(PRE_BUMP_BYTES).is_err());
}

/// Generate the wire-format fixture. Run with `--ignored` to overwrite
/// `crates/within/tests/fixtures/preconditioner_v6.postcard`. Intended for
/// intentional wire-format bumps only; CI runs the non-ignored tests above.
#[test]
#[ignore]
fn regenerate_wire_format_fixture() {
    use std::io::Write;
    use std::path::PathBuf;

    let (f, g, z, _) = fixture_problem();
    let solver = Solver::new(
        fixture_effects(&f, &g, &z),
        None,
        PreconditionerConfig::default(),
    )
    .expect("build solver");
    let prec = solver
        .preconditioner()
        .expect("default solver has a preconditioner");
    let bytes = postcard::to_stdvec(prec).expect("serialize");

    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/fixtures/preconditioner_v6.postcard");
    let mut out = std::fs::File::create(&path).expect("create fixture file");
    out.write_all(&bytes).expect("write fixture bytes");
    eprintln!("wrote {} bytes to {}", bytes.len(), path.display());
}
