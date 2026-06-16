use ndarray::array;
use within::{solve, BuildError, Fe, LsmrOptions, PreconditionerConfig, Solver};

/// A model expressed as a `Vec<Fe>` of plain terms must solve to the same fit as
/// the equivalent observation-major category array through the established
/// `solve` path — i.e. the term-list lowering builds the correct design.
#[test]
fn plain_term_list_solves_like_category_array() {
    // 5 observations, 2 factors. The array is observation-major; the Fe terms
    // carry the same factors as per-observation level columns.
    let categories = array![[0u32, 0], [1, 0], [0, 1], [1, 1], [2, 0]];
    let y = [1.0, 2.0, 3.0, 4.0, 5.0];
    let params = LsmrOptions::default();
    let precond = PreconditionerConfig::default();

    let from_array = solve(categories.view(), &y, None, &params, &precond).expect("array solve");

    let terms = vec![
        Fe::new(vec![0u32, 1, 0, 1, 2], vec![], true).expect("factor 0"),
        Fe::new(vec![0u32, 0, 1, 1, 0], vec![], true).expect("factor 1"),
    ];
    let from_terms = Solver::new(terms, None, &precond)
        .expect("term-list solver")
        .solve(&y, &params)
        .expect("term-list solve");

    assert!(from_terms.converged, "term-list solve did not converge");
    assert_eq!(from_terms.demeaned.len(), from_array.demeaned.len());
    for (t, a) in from_terms.demeaned.iter().zip(from_array.demeaned.iter()) {
        assert!(
            (t - a).abs() < 1e-9,
            "term-list fit diverged from array fit: {t} vs {a}"
        );
    }
}

/// A term carrying varying slopes is rejected at the public solver boundary —
/// the lowering gate surfaces as `BuildError::Unsupported`, not a panic or a
/// silently-wrong solve.
#[test]
fn slope_term_is_rejected_at_solver_construction() {
    let terms = vec![Fe::new(vec![0u32, 1, 0], vec![vec![1.0, 2.0, 3.0]], true)
        .expect("slope term is valid input")];
    let result = Solver::new(terms, None, &PreconditionerConfig::default());
    assert!(matches!(result, Err(BuildError::Unsupported(msg)) if msg.contains("slope")));
}
