//! Fixest-like worker/year(/firm) panel DGP, shared between the `fixest`
//! bench and the `profile_hot_loop` example via `#[path]` includes (this
//! directory is not auto-discovered as a bench target).

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use within::observation::ObservationFrame;
use within::Design;

/// Build a panel with 10 observations per worker and ~23 workers per firm.
/// `difficult` assigns firms round-robin (`i % n_firm`, high connectivity);
/// otherwise firms are drawn uniformly at random.
pub fn generate_fixest_like_case(
    n_obs: usize,
    n_fe: usize,
    difficult: bool,
    seed: u64,
) -> (Design<'static>, Vec<f64>) {
    let mut rng = SmallRng::seed_from_u64(seed);
    let n_years = 10usize;
    let n_indiv_per_firm = 23usize;

    let n_indiv = ((n_obs as f64 / n_years as f64).round() as usize).max(1);
    let n_firm = ((n_indiv as f64 / n_indiv_per_firm as f64).round() as usize).max(1);

    let mut indiv_id = Vec::with_capacity(n_obs);
    let mut year = Vec::with_capacity(n_obs);
    let mut firm_id = Vec::with_capacity(n_obs);

    for i in 0..n_obs {
        indiv_id.push((i / n_years) as u32);
        year.push((i % n_years) as u32);
        let firm = if difficult {
            (i % n_firm) as u32
        } else {
            rng.random_range(0..n_firm) as u32
        };
        firm_id.push(firm);
    }

    let factor_levels: Vec<Vec<u32>> = if n_fe == 2 {
        vec![indiv_id, year]
    } else {
        vec![indiv_id, year, firm_id]
    };

    let frame = ObservationFrame::new(
        factor_levels.into_iter().map(Into::into).collect(),
        Vec::new(),
    )
    .expect("valid frame");
    let design = Design::from_frame(frame).expect("valid design");

    // Random y — callers measure iteration time on an arbitrary RHS, not
    // ground-truth recovery.
    let mut y = vec![0.0; n_obs];
    for yi in &mut y {
        *yi = rng.random_range(-1.0..1.0);
    }

    (design, y)
}
