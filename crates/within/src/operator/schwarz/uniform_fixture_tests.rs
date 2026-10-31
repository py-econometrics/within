use super::*;
use crate::{Effect, LsmrOptions, Solver};
use schwarz_precond::PartitionWeights;

fn problem() -> (Vec<u32>, Vec<u32>, Vec<f64>) {
    (
        vec![0, 0, 0, 1, 1, 1],
        vec![0, 1, 2, 0, 1, 2],
        vec![1.0, -2.0, 0.5, 3.0, -1.5, 2.5],
    )
}

fn effects<'a>(f: &'a [u32], g: &'a [u32]) -> Vec<Effect<'a>> {
    vec![
        Effect::new(f, true, []).expect("first intercept"),
        Effect::new(g, true, []).expect("second intercept"),
    ]
}

fn config() -> PreconditionerConfig {
    PreconditionerConfig::Additive {
        local_solver: LocalSolverConfig::default(),
    }
}

fn uniform_domains(preconditioner: &Preconditioner) {
    let Variant::Additive(additive) = &preconditioner.inner else {
        panic!("fixture must be additive Schwarz");
    };
    let subdomains = additive.inner.subdomains();
    assert_eq!(subdomains.len(), 1, "connected two-factor fixture");
    let mut counts = vec![0; preconditioner.nrows()];
    for subdomain in subdomains {
        assert!(!matches!(
            subdomain.partition_weights(),
            PartitionWeights::NonUniform(_)
        ));
        for (local, &global) in subdomain.global_indices().iter().enumerate() {
            counts[global as usize] += 1;
            assert_eq!(subdomain.partition_weights().get(local), 1.0);
        }
    }
    assert!(counts.iter().all(|&count| count == 1));
}

fn fixture(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn equivalent(old: &Preconditioner, fresh: &Preconditioner) {
    assert_eq!(old.nrows(), fresh.nrows());
    assert_eq!(old.ncols(), fresh.ncols());
    assert_eq!(old.config(), fresh.config());
    uniform_domains(old);
    let input: Vec<f64> = (0..fresh.ncols()).map(|i| i as f64 + 0.25).collect();
    let mut before = vec![0.0; fresh.nrows()];
    let mut after = vec![0.0; fresh.nrows()];
    fresh.apply(&input, &mut before).expect("fresh apply");
    old.apply(&input, &mut after).expect("restored apply");
    assert_eq!(before.len(), after.len());
    for (a, b) in before.iter().zip(after.iter()) {
        assert!((a - b).abs() < 1e-12, "fixture apply drift: {a} vs {b}");
    }
    let (f, g, y) = problem();
    let a = Solver::new(effects(&f, &g), None, old.clone())
        .expect("restored solver")
        .solve(&y, &LsmrOptions::default())
        .expect("restored solve");
    let b = Solver::new(effects(&f, &g), None, fresh.clone())
        .expect("fresh solver")
        .solve(&y, &LsmrOptions::default())
        .expect("fresh solve");
    assert!(a.converged && b.converged);
    assert_eq!(a.x.len(), b.x.len());
    for (a, b) in a.x.iter().zip(b.x.iter()) {
        assert!((a - b).abs() < 1e-9, "fixture solve drift: {a} vs {b}");
    }
}

#[test]
#[ignore = "explicit fixture generation only"]
fn regenerate_uniform_partition_fixture() {
    let version =
        std::env::var("WITHIN_UNIFORM_FIXTURE_VERSION").unwrap_or_else(|_| "21".to_owned());
    assert!(version == "20" || version == "21");
    let (f, g, _) = problem();
    let solver = Solver::new(effects(&f, &g), None, config()).expect("fixture solver");
    let preconditioner = solver.preconditioner().expect("Schwarz preconditioner");
    uniform_domains(preconditioner);
    let bytes = postcard::to_stdvec(preconditioner).expect("serialize fixture");
    let decoded: Preconditioner = postcard::from_bytes(&bytes).expect("same-build decode");
    equivalent(&decoded, preconditioner);
    let path = fixture(&format!("preconditioner_uniform_v{version}.postcard"));
    assert!(!path.exists(), "old fixtures must never be overwritten");
    std::fs::write(path, bytes).expect("write fixture");
}

#[test]
fn uniform_fixture_deserializes_and_solves() {
    let bytes = std::fs::read(fixture("preconditioner_uniform_v21.postcard"))
        .expect("generated candidate fixture");
    let decoded: Preconditioner = postcard::from_bytes(&bytes).expect("candidate decode");
    let (f, g, _) = problem();
    let solver = Solver::new(effects(&f, &g), None, config()).expect("fresh fixture solver");
    equivalent(&decoded, solver.preconditioner().expect("fresh map"));
}

#[test]
fn previous_uniform_fixture_must_not_silently_change_the_map() {
    let bytes = std::fs::read(fixture("preconditioner_uniform_v20.postcard"))
        .expect("unmodified-base fixture");
    match postcard::from_bytes::<Preconditioner>(&bytes) {
        Err(_) => eprintln!("OLD_UNIFORM_OUTCOME=REJECTED"),
        Ok(decoded) => {
            let (f, g, _) = problem();
            let solver = Solver::new(effects(&f, &g), None, config()).expect("fresh map");
            equivalent(&decoded, solver.preconditioner().expect("fresh map"));
            eprintln!("OLD_UNIFORM_OUTCOME=COMPATIBLE");
        }
    }
}
