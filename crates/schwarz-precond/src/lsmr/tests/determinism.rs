//! Budget independence: the same solve returns the same bits under any `Par`.

use rstest::rstest;

use super::super::*;
use crate::fold::assert_budget_independent;
use crate::lsmr::fixtures::*;
use faer::sparse::{SparseColMat, Triplet};

/// `y[i] = x[i mod n] + ½ x[(7i + 3) mod n]`: large enough that every reduction splits.
fn wrapped(m: usize, n: usize) -> SparseColMat<usize, f64> {
    let entries = (0..m)
        .flat_map(|i| [(i % n, 1.0), ((7 * i + 3) % n, 0.5)].map(|(j, v)| Triplet::new(i, j, v)));
    SparseColMat::try_new_from_triplets(m, n, &entries.collect::<Vec<_>>()).expect("triplets")
}

fn spread(len: usize, scale: f64) -> Vec<f64> {
    (0..len)
        .map(|i| ((i as u64 * 2_654_435_761) % 1_000) as f64 * scale / 1_000.0 - 0.5 * scale)
        .collect()
}

/// Everything a caller can read off a result, as bits.
fn fingerprint(x: &[f64], r: &LsmrInfo) -> (Vec<u64>, usize, u64, u64, LsmrStopReason) {
    (
        x.iter().map(|v| v.to_bits()).collect(),
        r.iterations,
        r.residual_norm.to_bits(),
        r.normal_eq_residual.to_bits(),
        r.stop_reason,
    )
}

/// `n` above `fold`'s sequential cutoff, so column-length reductions also run across threads.
const M: usize = 300_000;
const N: usize = 80_000;

#[rstest]
fn lsmr_returns_the_same_bits_under_any_budget(#[values(None, Some(4))] local_size: Option<usize>) {
    let op = wrapped(M, N);
    let b = spread(M, 1.0);
    let options = LsmrOptions {
        local_size,
        ..opts(1e-14, 40)
    };
    assert_budget_independent(|par| {
        let mut x = vec![0.0; N];
        let info = mlsmr(
            &op,
            &b,
            &IdentityPrecond { dim: x.len() },
            &mut x,
            InitialGuess::Zero,
            &options,
            None,
            par,
            &mut MemBuffer::new(StackReq::EMPTY),
        )
        .expect("lsmr");
        fingerprint(&x, &info)
    });
}

#[rstest]
fn mlsmr_returns_the_same_bits_under_any_budget(
    #[values(None, Some(4))] local_size: Option<usize>,
) {
    let op = wrapped(M, N);
    let b = spread(M, 1.0);
    // Entries over three decades, so `⟨v, M⁻¹p̃⟩` is not a rescaled norm.
    let m: Vec<f64> = spread(N, 1.0)
        .iter()
        .map(|v| 10f64.powf(3.0 * (v + 0.5)))
        .collect();
    let m = diag(&m);
    let options = LsmrOptions {
        local_size,
        ..opts(1e-14, 40)
    };
    assert_budget_independent(|par| {
        let mut x = vec![0.0; N];
        let info = mlsmr(
            &op,
            &b,
            &m,
            &mut x,
            InitialGuess::Zero,
            &options,
            None,
            par,
            &mut MemBuffer::new(StackReq::EMPTY),
        )
        .expect("mlsmr");
        fingerprint(&x, &info)
    });
}
