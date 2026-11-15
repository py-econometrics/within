use rstest::rstest;
use schwarz_precond::Operator;

use super::{build_two_level, TwoLevelMap};
use crate::config::{CoarseBuild, CoarseCorrection, CoarseSpace, LocalSolverConfig, TwoLevelForm};
use crate::domain::{Design, Effect, PreparedDesign, Unknowns};
use crate::linalg::dot;
use crate::operator::design::ReducedDesign;
use crate::operator::schwarz::{build_schwarz, RungBuild, RungMap};
use crate::CoarseDecline;

const COARSE: CoarseCorrection = CoarseCorrection {
    space: CoarseSpace::RigidComponents,
    form: TwoLevelForm::Balanced,
    build: CoarseBuild::WithSchwarz,
};

/// Rows of a worker × firm panel, with a slope per row for each.
#[derive(Default)]
struct Panel {
    worker: Vec<u32>,
    firm: Vec<u32>,
    z_worker: Vec<f64>,
    z_firm: Vec<f64>,
}

impl Panel {
    fn spell(&mut self, worker: u32, firm: u32, rows: usize) {
        for _ in 0..rows {
            let k = self.worker.len() as f64;
            self.worker.push(worker);
            self.firm.push(firm);
            self.z_worker.push((0.37 * k).sin() + 0.01 * worker as f64);
            self.z_firm.push((0.61 * k).cos());
        }
    }

    /// `stayers` workers per firm with `stay` rows each, then `movers` workers moving between
    /// firms `a` and `b` with `long` and `short` rows: only the long spell glues.
    fn clustered(
        n_firms: u32,
        stayers: u32,
        stay: usize,
        movers: u32,
        long: usize,
        short: usize,
    ) -> Self {
        let mut panel = Self::default();
        let mut worker = 0;
        for firm in 0..n_firms {
            for _ in 0..stayers {
                panel.spell(worker, firm, stay);
                worker += 1;
            }
        }
        for k in 0..movers {
            let a = (7 * k + 3) % n_firms;
            let b = (a + 1 + (13 * k) % (n_firms - 1)) % n_firms;
            panel.spell(worker, a, long);
            panel.spell(worker, b, short);
            worker += 1;
        }
        panel
    }

    /// Every worker has `home` rows at one firm and one row at each of `touches` others.
    fn scattered(n_firms: u32, workers: u32, home: usize, touches: u32) -> Self {
        let mut panel = Self::default();
        for k in 0..workers {
            panel.spell(k, k % n_firms, home);
            for j in 1..=touches {
                panel.spell(k, (k + 7 * j) % n_firms, 1);
            }
        }
        panel
    }

    /// Both terms with an intercept and a slope.
    fn design(&self) -> Design<'_> {
        Design::new(vec![
            Effect::new(&self.worker, true, [&self.z_worker[..]]).expect("worker"),
            Effect::new(&self.firm, true, [&self.z_firm[..]]).expect("firm"),
        ])
        .expect("design")
    }
}

#[derive(Clone, Copy, Debug)]
enum System {
    Full,
    Reduced,
}

#[derive(Clone, Copy, Debug)]
enum Weights {
    Unit,
    Varying,
}

impl Weights {
    fn of(self, panel: &Panel) -> Option<Vec<f64>> {
        match self {
            Self::Unit => None,
            Self::Varying => Some(
                (0..panel.worker.len())
                    .map(|k| 0.5 + 0.4 * (0.23 * k as f64).sin())
                    .collect(),
            ),
        }
    }
}

/// `f` on the system's prepared design, its normal matrix's operator and its unknowns.
fn on_system<R>(
    design: Design<'_>,
    weights: Option<&[f64]>,
    system: System,
    f: impl FnOnce(&PreparedDesign<'_>, &dyn Operator, Unknowns) -> R,
) -> R {
    match system {
        System::Full => {
            let prepared = PreparedDesign::new(design, weights).expect("prepared");
            f(&prepared, &prepared, Unknowns::All)
        }
        System::Reduced => {
            let reduced = ReducedDesign::new(design, weights).expect("reduced");
            f(reduced.prepared(), &reduced, Unknowns::WithoutDominant)
        }
    }
}

fn two_level(
    prepared: &PreparedDesign<'_>,
    unknowns: Unknowns,
) -> Result<TwoLevelMap, CoarseDecline> {
    let RungBuild {
        map: Some(RungMap::Schwarz(one_level)),
        ..
    } = build_schwarz(prepared, &LocalSolverConfig::default(), unknowns).expect("schwarz")
    else {
        panic!("no one-level map");
    };
    build_two_level(prepared, unknowns, &one_level, &COARSE)
}

/// `AᵀA x`.
fn normal(a: &dyn Operator, x: &[f64]) -> Vec<f64> {
    let (mut ax, mut out) = (vec![0.0; a.nrows()], vec![0.0; a.ncols()]);
    a.apply(x, &mut ax).expect("A x");
    a.apply_adjoint(&ax, &mut out).expect("Aᵀ y");
    out
}

/// The balanced form is exact on its coarse space, `A M S Z c = A Z c`: what it adds to `Z c` lies
/// in the null space of `A`, where the one-level map is free.
#[rstest]
fn the_two_level_map_inverts_the_normal_matrix_on_its_coarse_space(
    #[values(System::Full, System::Reduced)] system: System,
    #[values(Weights::Unit, Weights::Varying)] weights: Weights,
) {
    let panel = Panel::clustered(40, 3, 8, 30, 7, 1);
    let weights = weights.of(&panel);
    on_system(
        panel.design(),
        weights.as_deref(),
        system,
        |prepared, a, unknowns| {
            let map = two_level(prepared, unknowns).expect("built");
            let (n, m) = (map.nrows(), map.coarse.z.ncols());
            assert!(m >= 40, "{m} coarse columns");
            let image = |x: &[f64]| {
                let mut ax = vec![0.0; a.nrows()];
                a.apply(x, &mut ax).expect("A x");
                ax
            };
            for column in 0..m {
                let mut c = vec![0.0; m];
                c[column] = 1.0;
                let mut zc = vec![0.0; n];
                map.coarse.z.mul_add(&c, 1.0, &mut zc);
                let mut y = vec![0.0; n];
                map.apply(&normal(a, &zc), &mut y).expect("apply");
                let (ay, azc) = (image(&y), image(&zc));
                let error: f64 = ay.iter().zip(&azc).map(|(y, z)| (y - z).powi(2)).sum();
                assert!(
                    error.sqrt() <= 1e-10 * dot(&azc, &azc).sqrt(),
                    "column {column}: ‖A (M S Z c − Z c)‖ = {:.3e}",
                    error.sqrt()
                );
            }
        },
    );
}

#[rstest]
fn the_two_level_map_is_symmetric(
    #[values(System::Full, System::Reduced)] system: System,
    #[values(Weights::Unit, Weights::Varying)] weights: Weights,
) {
    let panel = Panel::clustered(40, 3, 8, 30, 7, 1);
    let weights = weights.of(&panel);
    on_system(
        panel.design(),
        weights.as_deref(),
        system,
        |prepared, _, unknowns| {
            let map = two_level(prepared, unknowns).expect("built");
            let n = map.nrows();
            let x: Vec<f64> = (0..n).map(|i| (0.3 * i as f64).sin()).collect();
            let v: Vec<f64> = (0..n).map(|i| (0.7 * i as f64 + 1.0).cos()).collect();
            let (mut mx, mut mv) = (vec![0.0; n], vec![0.0; n]);
            map.apply(&x, &mut mx).expect("M x");
            map.apply(&v, &mut mv).expect("M v");
            let (a, b) = (dot(&v, &mx), dot(&x, &mv));
            assert!((a - b).abs() <= 1e-10 * a.abs().max(b.abs()), "{a} vs {b}");
        },
    );
}

#[derive(Clone, Copy, Debug)]
enum Unglued {
    /// No kept term shares the eliminated term's columns.
    NoAnchor,
    /// Every worker stays at one firm.
    NothingSplits,
    /// Movers touching many firms once each make `E` denser than the design.
    DenseCoarseSystem,
}

#[rstest]
fn a_design_without_a_fitting_coarse_space_declines(
    #[values(System::Full, System::Reduced)] system: System,
    #[values(Unglued::NoAnchor, Unglued::NothingSplits, Unglued::DenseCoarseSystem)]
    unglued: Unglued,
) {
    let (panel, expected) = match unglued {
        Unglued::NoAnchor => (
            Panel::clustered(40, 3, 8, 30, 7, 1),
            CoarseDecline::NoCoarseSpace,
        ),
        Unglued::NothingSplits => (
            Panel::clustered(40, 3, 8, 0, 0, 0),
            CoarseDecline::NoCoarseSpace,
        ),
        Unglued::DenseCoarseSystem => (
            Panel::scattered(200, 500, 60, 19),
            CoarseDecline::OverBudget,
        ),
    };
    let design = match unglued {
        Unglued::NoAnchor => Design::new(vec![
            Effect::new(&panel.worker, true, [&panel.z_worker[..]]).expect("worker"),
            Effect::new(&panel.firm, true, []).expect("firm"),
        ])
        .expect("design"),
        _ => panel.design(),
    };
    on_system(design, None, system, |prepared, _, unknowns| {
        let decline = two_level(prepared, unknowns).expect_err("declines");
        assert_eq!(decline, expected);
    });
}

/// A row of zero weight is the row removed: it touches no component and lifts nothing.
#[rstest]
fn a_zero_weight_row_leaves_the_two_level_map_as_its_removal(
    #[values(System::Full, System::Reduced)] system: System,
) {
    let panel = Panel::clustered(40, 3, 8, 30, 7, 1);
    let dropped = |k: usize| k % 5 == 0 && k >= 40 * 3 * 8;
    let weights: Vec<f64> = (0..panel.worker.len())
        .map(|k| if dropped(k) { 0.0 } else { 1.0 })
        .collect();
    let mut kept = Panel::default();
    for k in (0..panel.worker.len()).filter(|&k| !dropped(k)) {
        kept.worker.push(panel.worker[k]);
        kept.firm.push(panel.firm[k]);
        kept.z_worker.push(panel.z_worker[k]);
        kept.z_firm.push(panel.z_firm[k]);
    }
    let applied = |panel: &Panel, weights: Option<&[f64]>| {
        on_system(panel.design(), weights, system, |prepared, _, unknowns| {
            let map = two_level(prepared, unknowns).expect("built");
            let x: Vec<f64> = (0..map.nrows()).map(|i| (0.3 * i as f64).sin()).collect();
            let mut y = vec![0.0; map.nrows()];
            map.apply(&x, &mut y).expect("apply");
            y
        })
    };
    let (zeroed, removed) = (applied(&panel, Some(&weights)), applied(&kept, None));
    let gap: f64 = zeroed
        .iter()
        .zip(&removed)
        .map(|(a, b)| (a - b).powi(2))
        .sum();
    assert!(
        gap.sqrt() <= 1e-10 * dot(&removed, &removed).sqrt(),
        "gap {:.3e}",
        gap.sqrt()
    );
}
