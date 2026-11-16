//! Wire-format byte fixture for the serialized [`Preconditioner`]: deserializes
//! a payload from a known-good build and drives a solve, catching silent
//! cross-version encoding shifts the same-build round-trip test cannot.
//! Regenerate via the `#[ignore]`d `regenerate_wire_format_fixture` test.

use rstest::rstest;
use within::{
    CoarseBuild, CoarseCorrection, CoarseSpace, Effect, LsmrOptions, Preconditioner,
    PreconditionerConfig, Rung, Solver, TwoLevelForm,
};

#[path = "common/orchestrate_helpers.rs"]
mod common;
use common::schwarz;

const WIRE_FORMAT_VERSION: u32 = 27;
const PRECOND_BYTES: &[u8] = include_bytes!("fixtures/preconditioner_v27.postcard");
const FULL_SYSTEM_BYTES: &[u8] = include_bytes!("fixtures/preconditioner_v26.postcard");
const ONE_LEVEL_BYTES: &[u8] = include_bytes!("fixtures/preconditioner_v25.postcard");
const COARSELESS_BYTES: &[u8] = include_bytes!("fixtures/preconditioner_v24.postcard");
const LAYOUTLESS_BYTES: &[u8] = include_bytes!("fixtures/preconditioner_v23.postcard");
const LADDERLESS_BYTES: &[u8] = include_bytes!("fixtures/preconditioner_v22.postcard");
const PRE_BUMP_BYTES: &[u8] = include_bytes!("fixtures/preconditioner_v21.postcard");
const EARLIER_PRE_BUMP_BYTES: &[u8] = include_bytes!("fixtures/preconditioner_v20.postcard");
const RELEASE_0_3_0_BYTES: &[u8] = include_bytes!("fixtures/preconditioner_v12.postcard");

fn fixture_problem() -> (Vec<u32>, Vec<u32>, Vec<f64>, Vec<f64>) {
    // The frustrated (f-slope, g) pair pins a signed operator with Scaled coords and a Cover.
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

/// Workers staying in 40 firm clusters and movers whose one row elsewhere glues nothing, a
/// slope on both terms: rigid components only a coarse space sees.
struct Clustered {
    worker: Vec<u32>,
    firm: Vec<u32>,
    z_worker: Vec<f64>,
    z_firm: Vec<f64>,
}

impl Clustered {
    fn new() -> Self {
        let (mut worker, mut firm) = (Vec::new(), Vec::new());
        let mut spell = |w: u32, f: u32, rows: usize| {
            worker.extend(std::iter::repeat_n(w, rows));
            firm.extend(std::iter::repeat_n(f, rows));
        };
        for k in 0..120 {
            spell(k, k / 3, 8);
        }
        for k in 0..30 {
            let a = (7 * k + 3) % 40;
            spell(120 + k, a, 7);
            spell(120 + k, (a + 1 + (13 * k) % 39) % 40, 1);
        }
        let n = worker.len();
        Self {
            worker,
            firm,
            z_worker: (0..n).map(|i| (0.37 * i as f64).sin()).collect(),
            z_firm: (0..n).map(|i| (0.61 * i as f64).cos()).collect(),
        }
    }

    fn effects(&self) -> Vec<Effect<'_>> {
        vec![
            Effect::new(&self.worker, true, [&self.z_worker[..]]).expect("worker"),
            Effect::new(&self.firm, true, [&self.z_firm[..]]).expect("firm"),
        ]
    }

    fn y(&self) -> Vec<f64> {
        (0..self.worker.len())
            .map(|i| (0.23 * i as f64).sin())
            .collect()
    }
}

fn two_level() -> PreconditionerConfig {
    PreconditionerConfig::Schwarz(within::Schwarz {
        coarse: Some(CoarseCorrection {
            space: CoarseSpace::RigidComponents,
            form: TwoLevelForm::Balanced,
            build: CoarseBuild::WithSchwarz,
        }),
        ..Default::default()
    })
}

#[test]
fn wire_format_fixture_deserializes_and_solves() {
    let _ = WIRE_FORMAT_VERSION;

    let panel = Clustered::new();
    let prebuilt: Preconditioner =
        postcard::from_bytes(PRECOND_BYTES).expect("deserialize fixture preconditioner");
    assert_eq!(prebuilt.rung(), Rung::TwoLevel);
    let solver = Solver::new(panel.effects(), None, prebuilt).expect("solver from fixture");
    let result = solver
        .solve(&panel.y(), &LsmrOptions::default())
        .expect("solve with fixture preconditioner");
    assert!(result.converged, "fixture-built solver should converge");

    let fresh = Solver::new(panel.effects(), None, two_level()).expect("fresh solver");
    let fresh_result = fresh
        .solve(&panel.y(), &LsmrOptions::default())
        .expect("fresh solve");
    for (a, b) in result.x.iter().zip(fresh_result.x.iter()) {
        assert!((a - b).abs() < 1e-9, "fixture vs fresh: {a} vs {b}");
    }
}

#[derive(Clone, Copy)]
enum Shape {
    Signed,
    Uniform,
}

#[rstest]
#[case::signed(Shape::Signed)]
#[case::uniform(Shape::Uniform)]
fn preconditioner_round_trips(#[case] shape: Shape) {
    let (f, g, z, y) = fixture_problem();
    let effects = || match shape {
        Shape::Signed => fixture_effects(&f, &g, &z),
        Shape::Uniform => vec![
            Effect::new(&f, true, []).expect("first intercept"),
            Effect::new(&g, true, []).expect("second intercept"),
        ],
    };
    let solver1 = Solver::new(effects(), None, schwarz()).expect("build solver");
    let r1 = solver1.solve(&y, &LsmrOptions::default()).expect("solve 1");
    let original = solver1.preconditioner().expect("has preconditioner");
    let bytes = postcard::to_stdvec(original).expect("serialize");
    let restored: Preconditioner = postcard::from_bytes(&bytes).expect("deserialize");
    assert_eq!(restored.config(), &schwarz());
    assert_eq!(restored.nrows(), original.nrows());
    assert_eq!(restored.ncols(), original.ncols());

    let input: Vec<f64> = (0..original.ncols()).map(|i| i as f64 + 0.25).collect();
    let mut before = vec![0.0; original.nrows()];
    let mut after = vec![0.0; restored.nrows()];
    original.apply(&input, &mut before).expect("original apply");
    restored.apply(&input, &mut after).expect("restored apply");
    assert_eq!(before, after);

    let solver2 = Solver::new(effects(), None, restored).expect("solver from round-trip");
    let r2 = solver2.solve(&y, &LsmrOptions::default()).expect("solve 2");
    assert!(r1.converged && r2.converged);
    assert_eq!(r1.x.len(), r2.x.len());
    for (a, b) in r1.x.iter().zip(r2.x.iter()) {
        assert!(
            (a - b).abs() < 1e-12,
            "round-trip coefficient mismatch: {} vs {}",
            a,
            b,
        );
    }
}

#[rstest]
#[case::v26(FULL_SYSTEM_BYTES)]
#[case::v25(ONE_LEVEL_BYTES)]
#[case::v24(COARSELESS_BYTES)]
#[case::v23(LAYOUTLESS_BYTES)]
#[case::v22(LADDERLESS_BYTES)]
#[case::v21(PRE_BUMP_BYTES)]
#[case::v20(EARLIER_PRE_BUMP_BYTES)]
#[case::release_0_3_0(RELEASE_0_3_0_BYTES)]
fn legacy_fixture_no_longer_decodes(#[case] bytes: &[u8]) {
    assert!(postcard::from_bytes::<Preconditioner>(bytes).is_err());
}

/// Generate the wire-format fixture. Run with `--ignored` to overwrite
/// `crates/within/tests/fixtures/preconditioner_v27.postcard`. Intended for
/// intentional wire-format bumps only; CI runs the non-ignored tests above.
#[test]
#[ignore]
fn regenerate_wire_format_fixture() {
    use std::io::Write;
    use std::path::PathBuf;

    let panel = Clustered::new();
    let solver = Solver::new(panel.effects(), None, two_level()).expect("build solver");
    let prec = solver.preconditioner().expect("a two-level map");
    assert_eq!(prec.rung(), Rung::TwoLevel);
    let bytes = postcard::to_stdvec(prec).expect("serialize");

    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/fixtures/preconditioner_v27.postcard");
    let mut out = std::fs::File::create(&path).expect("create fixture file");
    out.write_all(&bytes).expect("write fixture bytes");
    eprintln!("wrote {} bytes to {}", bytes.len(), path.display());
}
