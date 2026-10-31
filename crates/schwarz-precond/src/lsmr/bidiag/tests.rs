//! White-box checks on the bidiagonalization itself.

use rstest::rstest;

use faer::dyn_stack::{MemBuffer, MemStack, StackReq};

use faer::matrix_free::{BiLinOp, LinOp};
use faer::{MatMut, MatRef};

use super::{
    alpha_from_vp, dot, par_dot, BidiagStep, Bidiagonalization, Ctx, ModifiedGolubKahan,
    StagedStart,
};
use crate::lsmr::fixtures::*;
use crate::par::Par;
use crate::SolveError;

/// The unpreconditioned stream from `b`, and its first step.
fn cold_stream<'a, A: BiLinOp<f64>>(
    op: &'a A,
    identity: &'a IdentityPrecond,
    b: &[f64],
    local_size: usize,
    ctx: Ctx<'a>,
) -> Result<(ModifiedGolubKahan<'a, A, IdentityPrecond>, BidiagStep), SolveError> {
    let staged = StagedStart::cold(op, identity, b, local_size, ctx);
    let (stream, step1, _) = staged.restart(crate::lsmr::vec_norm(b))?;
    Ok((stream, step1))
}

/// A NaN `vp` clamped to α = 0 via `f64::max`; a product bound overflowed to ∞ or underflowed to 0.
#[rstest]
#[case::nan(&[1.0, f64::NAN], &[1.0, 1.0])]
#[case::infinity(&[1.0, f64::INFINITY], &[1.0, 1.0])]
#[case::indefinite(&[-1e100], &[1e100])]
#[case::overflowing_bound(&[-1e302, f64::MAX, f64::MAX], &[1.0, 0.0, 0.0])]
#[case::underflowing_bound(&[-2e300, 1e308], &[1e-316, 0.0])]
fn alpha_from_vp_rejects_non_finite_and_indefinite_pairs(#[case] v: &[f64], #[case] p: &[f64]) {
    assert!(
        matches!(
            alpha_from_vp(v, p, Par::Seq).map_err(SolveError::from),
            Err(SolveError::InvalidInput { .. })
        ),
        "{v:?}·{p:?} accepted"
    );
}

#[rstest]
#[case::unit_scale(&[1.0, 1.0], &[1.0, -1.0 - 1e-12])]
#[case::tiny_scale(&[1e-100, 1e-100], &[1e-100, -1.000000000001e-100])]
#[case::extreme_scales(&[1e-316, 1e-316], &[1e308, -1.000000000001e308])]
fn alpha_from_vp_clamps_a_pair_within_root_epsilon(#[case] v: &[f64], #[case] p: &[f64]) {
    assert_eq!(
        alpha_from_vp(v, p, Par::Seq).expect("within √ε"),
        0.0,
        "{v:?}·{p:?}"
    );
}

/// `cos(v, p̃) ≈ 1e-328` underflows a re-sum normalized to 1, though `α ≈ 1e-158` does not.
#[test]
fn a_tiny_cosine_keeps_its_gradient_length() {
    let alpha = alpha_from_vp(&[1e-158, 0.0], &[1e-158, 1e170], Par::Seq).expect("positive pair");
    assert!((alpha / 1e-158 - 1.0).abs() < 1e-12, "{alpha:e}");
}

/// A `vp` whose terms round to subnormals takes its sign from the scaled re-sum.
#[test]
fn a_negative_subnormal_pair_keeps_its_gradient_length() {
    let t = 2f64.powi(-537);
    let alpha =
        alpha_from_vp(&[t; 3], &[1.49 * t, 1.49 * t, -2.9 * t], Par::Seq).expect("positive pair");
    assert!(
        (alpha / (0.08f64.sqrt() * t) - 1.0).abs() < 1e-10,
        "{alpha:e}"
    );
}

/// A `vp` rounded below zero is 0 however the fallback's normalized re-sum rounds.
#[test]
fn a_negatively_rounded_pair_clamps_however_the_resum_rounds() {
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    let mut draw = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    for _ in 0..40 {
        // Longer than one 4096-item block, so `vp` and the re-sum associate differently.
        let v: Vec<f64> = (0..20_000).map(|_| draw()).collect();
        let mut p: Vec<f64> = (0..20_000).map(|_| draw()).collect();
        let c = dot(&v, &p) / dot(&v, &v);
        p.iter_mut().zip(&v).for_each(|(pi, vi)| *pi -= c * vi);
        if par_dot(&v, &p, Par::Seq) < 0.0 {
            assert_eq!(alpha_from_vp(&v, &p, Par::Seq).expect("within √ε"), 0.0);
        }
    }
}

/// `beta == 0.0` and `alpha > 0.0` are both false for NaN; unguarded, it poisons the run.
#[test]
fn a_non_finite_operator_norm_is_an_error() {
    let result = lsmr(&diag(&[f64::NAN, 1.0]), &[1.0, 1.0], 1e-10, 50, None);
    assert!(matches!(result, Err(SolveError::InvalidInput { .. })));
}

/// A finite adjoint keeps `init` clean, so the overflow reaches the `step` guard on β.
#[test]
fn an_overflow_after_initialization_is_an_error() {
    /// No matrix: the forward apply saturates at `f64::MAX` while the adjoint is the identity.
    #[derive(Debug)]
    struct OverflowingForward;
    impl LinOp<f64> for OverflowingForward {
        fn apply_scratch(&self, _: usize, _: Par) -> StackReq {
            StackReq::EMPTY
        }
        fn nrows(&self) -> usize {
            2
        }
        fn ncols(&self) -> usize {
            2
        }
        fn apply(&self, mut out: MatMut<'_, f64>, _: MatRef<'_, f64>, _: Par, _: &mut MemStack) {
            out.fill(f64::MAX);
        }
        fn conj_apply(&self, out: MatMut<'_, f64>, rhs: MatRef<'_, f64>, p: Par, s: &mut MemStack) {
            self.apply(out, rhs, p, s)
        }
    }
    impl BiLinOp<f64> for OverflowingForward {
        fn transpose_apply_scratch(&self, _: usize, _: Par) -> StackReq {
            StackReq::EMPTY
        }
        fn transpose_apply(
            &self,
            mut out: MatMut<'_, f64>,
            rhs: MatRef<'_, f64>,
            _: Par,
            _: &mut MemStack,
        ) {
            out.copy_from(rhs);
        }
        fn adjoint_apply(
            &self,
            out: MatMut<'_, f64>,
            rhs: MatRef<'_, f64>,
            p: Par,
            s: &mut MemStack,
        ) {
            self.transpose_apply(out, rhs, p, s)
        }
    }
    let mut buffer = MemBuffer::new(StackReq::EMPTY);
    let ctx = Ctx {
        par: Par::Seq,
        stack: MemStack::new(&mut buffer),
    };
    let (op, identity) = (OverflowingForward, IdentityPrecond { dim: 2 });
    let (mut bidiag, first) =
        cold_stream(&op, &identity, &[1.0, 1.0], 0, ctx).expect("finite init");
    assert!(first.alpha > 0.0);
    let err = bidiag.step().err().expect("an overflowing β was accepted");
    assert!(
        matches!(&err, SolveError::InvalidInput { message, .. } if message.contains("β")),
        "{err}"
    );
}

/// Window smaller than the iteration count: the ring must wrap correctly.
/// We re-run the bidiagonalization manually with the same window and
/// verify the last `local_size` `v` vectors are mutually orthogonal to
/// tighter tolerance than they would be without reorthogonalization.
#[test]
fn local_reorth_keeps_the_window_vectors_orthogonal() {
    let (op, b) = vandermonde_ls();

    let local_size = 3;
    let n_iters = 10;

    // Run the bidiagonalization directly so we can capture v_k after each
    // step. Mirrors the body of `lsmr_from_bidiag` minus the recurrence.
    let collect_vs = |window_size: usize| -> Vec<Vec<f64>> {
        let mut buffer = MemBuffer::new(StackReq::EMPTY);
        let ctx = Ctx {
            par: Par::Seq,
            stack: MemStack::new(&mut buffer),
        };
        let identity = IdentityPrecond { dim: op.ncols() };
        let (mut bidiag, _) = cold_stream(&op, &identity, &b, window_size, ctx).expect("init");
        let mut vs = vec![bidiag.v().to_vec()];
        for _ in 0..n_iters {
            bidiag.step().expect("step");
            vs.push(bidiag.v().to_vec());
        }
        vs
    };

    let vs_no_reorth = collect_vs(0);
    let vs_windowed = collect_vs(local_size);

    // Compare the maximum |⟨v_i, v_j⟩| over the last `local_size` vectors.
    let max_off_diag = |vs: &[Vec<f64>]| -> f64 {
        let n = vs.len();
        let start = n.saturating_sub(local_size);
        let mut worst: f64 = 0.0;
        for i in start..n {
            for j in (i + 1)..n {
                worst = worst.max(dot(&vs[i], &vs[j]).abs());
            }
        }
        worst
    };

    let drift_no = max_off_diag(&vs_no_reorth);
    let drift_yes = max_off_diag(&vs_windowed);
    assert!(
        drift_yes < drift_no,
        "windowed drift ({drift_yes:e}) should be smaller than \
         unwindowed drift ({drift_no:e})"
    );
    assert!(
        drift_yes < 1e-10,
        "last {local_size} v's not mutually orthogonal: {drift_yes:e}"
    );
}
