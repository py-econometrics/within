//! Measures what a diagonal-then-Schwarz restart costs against a cold Schwarz
//! solve, so a switching rule can be chosen from data rather than guessed.
//!
//! `cargo test --release -p within --lib hybrid -- --ignored --nocapture`

use crate::config::PreconditionerConfig;
use crate::domain::{Design, Effect};
use crate::operator::design::DesignOperator;
use crate::operator::schwarz::build_preconditioner;
use schwarz_precond::{mlsmr, MlsmrOptions};
use std::time::Instant;

#[path = "../../benches/shared/fixest_dgp.rs"]
mod fixest_dgp;
use fixest_dgp::generate_fixest_like_case;

const TOL: f64 = 1e-8;
const MAXITER: usize = 5000;
/// Budget for the diagonal-only baseline. Low-mobility AKM never finishes, and
/// an unbounded baseline would dominate the sweep's runtime.
const DIAG_CAP: usize = 5000;

struct Leg {
    iterations: usize,
    converged: bool,
    seconds: f64,
}

fn report(design: &Design<'_>, y: &[f64], budgets: &[usize]) {
    let op = DesignOperator::new(design, None);
    let b = op.weighted_rhs(y);

    let t = Instant::now();
    let (schwarz, _) = build_preconditioner(design, None, Some(&PreconditionerConfig::default()))
        .expect("schwarz build");
    let schwarz_setup = t.elapsed().as_secs_f64();
    let schwarz = schwarz.expect("schwarz preconditioner");

    let t = Instant::now();
    let (diagonal, _) = build_preconditioner(design, None, Some(&PreconditionerConfig::Diagonal))
        .expect("diagonal build");
    let diagonal_setup = t.elapsed().as_secs_f64();
    let diagonal = diagonal.expect("diagonal preconditioner");

    let run = |p: &_, budget: usize| {
        let t = Instant::now();
        let r = mlsmr(&op, &b, p, TOL, budget, MlsmrOptions::default()).expect("solve");
        (
            r.x,
            Leg {
                iterations: r.iterations,
                converged: r.converged,
                seconds: t.elapsed().as_secs_f64(),
            },
        )
    };

    let (_, cold) = run(&schwarz, MAXITER);
    let (_, diag_only) = run(&diagonal, MAXITER);
    // A leg that hit MAXITER would make every ratio below meaningless.
    assert!(
        cold.converged && diag_only.converged,
        "baseline leg stalled"
    );

    // Setup dominates the decision, so every total below carries its build cost.
    let cold_total = schwarz_setup + cold.seconds;
    println!(
        "  cold schwarz:  {:>4} iters, setup {:.3}s + solve {:.3}s = {:.3}s",
        cold.iterations, schwarz_setup, cold.seconds, cold_total
    );
    println!(
        "  diagonal only: {:>4} iters, setup {:.3}s + solve {:.3}s = {:.3}s  ({:.2}x cold)",
        diag_only.iterations,
        diagonal_setup,
        diag_only.seconds,
        diagonal_setup + diag_only.seconds,
        (diagonal_setup + diag_only.seconds) / cold_total
    );

    for &budget in budgets {
        let (x0, warm_up) = run(&diagonal, budget);
        let t = Instant::now();
        let restarted = mlsmr(
            &op,
            &b,
            &schwarz,
            TOL,
            MAXITER,
            MlsmrOptions {
                warm_start: Some(&x0),
                ..Default::default()
            },
        )
        .expect("restart");
        let total = diagonal_setup + warm_up.seconds + schwarz_setup + t.elapsed().as_secs_f64();
        assert!(
            restarted.converged,
            "restarted leg stalled at budget {budget}"
        );

        // Iterations the discarded Krylov subspace was worth: what the restarted
        // Schwarz leg still needs, against what a cold Schwarz solve needed.
        let penalty = restarted.iterations as isize - cold.iterations as isize;
        println!(
            "  budget {budget:>4}:  diag {} + schwarz {} iters (penalty {penalty:+}), \
             {total:.3}s ({:.2}x cold)",
            warm_up.iterations,
            restarted.iterations,
            total / cold_total,
        );
    }
}

#[test]
#[ignore = "measurement harness, not an assertion"]
fn hybrid_restart_penalty() {
    for (n_obs, n_fe, difficult) in [
        (200_000, 2, false),
        (200_000, 3, false),
        (200_000, 3, true),
        (1_000_000, 3, true),
    ] {
        let (design, y) = generate_fixest_like_case(n_obs, n_fe, difficult, 42);
        println!(
            "n_obs={n_obs} n_fe={n_fe} difficult={difficult} n_dofs={}",
            design.n_dofs
        );
        report(&design, &y, &[10, 25, 50, 100]);
    }
}

/// Cases spanning both verdicts, so a switching rule is judged against more
/// than one hard design.
const CASES: [(usize, usize, bool); 7] = [
    (200_000, 2, false),
    (200_000, 3, false),
    (200_000, 3, true),
    (1_000_000, 2, true),
    (1_000_000, 3, false),
    (1_000_000, 3, true),
    (2_000_000, 3, true),
];

/// Can a rule read off the first few diagonal iterations tell "converging now"
/// from "hopeless"? Prints the per-iteration contraction of the relative
/// normal-equation residual against the ground truth both preconditioners reach.
#[test]
#[ignore = "measurement harness, not an assertion"]
fn diagonal_stall_signal() {
    const PROBE: usize = 12;

    println!(
        "{:>28} | schwarz | diagonal | contraction of rel-NE-residual per diagonal iter",
        "design"
    );
    for (n_obs, n_fe, difficult) in CASES {
        let (design, y) = generate_fixest_like_case(n_obs, n_fe, difficult, 42);
        let op = DesignOperator::new(&design, None);
        let b = op.weighted_rhs(&y);

        let build = |cfg| {
            build_preconditioner(&design, None, Some(&cfg))
                .expect("build")
                .0
                .expect("preconditioner")
        };
        let schwarz = build(PreconditionerConfig::default());
        let diagonal = build(PreconditionerConfig::Diagonal);

        let iters = |p: &_| {
            mlsmr(&op, &b, p, TOL, MAXITER, MlsmrOptions::default())
                .expect("solve")
                .iterations
        };
        let schwarz_iters = iters(&schwarz);
        let diagonal_iters = iters(&diagonal);

        // Re-solved per budget rather than recorded via `EscalationRule`; the
        // recurrence is deterministic, making the truncations one trajectory.
        let traj: Vec<f64> = (1..=PROBE)
            .map(|j| {
                mlsmr(&op, &b, &diagonal, TOL, j, MlsmrOptions::default())
                    .expect("solve")
                    .normal_eq_residual
            })
            .collect();

        let ratios: Vec<String> = traj
            .windows(2)
            .map(|w| format!("{:.3}", w[1] / w[0]))
            .collect();
        println!(
            "{:>28} | {schwarz_iters:>7} | {diagonal_iters:>8} | {}",
            format!(
                "{n_obs}/{n_fe}FE/{}",
                if difficult { "hard" } else { "easy" }
            ),
            ratios.join(" ")
        );
    }
}

/// How firms are assigned to individuals — the axis that drives how well the
/// factor graph connects, and with it the conditioning the diagonal must face.
#[derive(Clone, Copy, PartialEq)]
enum Firms {
    /// Round-robin: every firm touches every year, maximal connectivity.
    Spread,
    /// Uniform draw.
    Random,
    /// Contiguous blocks: firms barely overlap.
    Blocked,
}

impl Firms {
    fn tag(self) -> &'static str {
        match self {
            Firms::Spread => "spread",
            Firms::Random => "random",
            Firms::Blocked => "block",
        }
    }
}

struct Shape {
    n_obs: usize,
    n_fe: usize,
    depth: usize,
    per_firm: usize,
    firms: Firms,
}

impl Shape {
    fn label(&self) -> String {
        format!(
            "{}k/{}FE/d{}/f{}/{}",
            self.n_obs / 1000,
            self.n_fe,
            self.depth,
            self.per_firm,
            self.firms.tag()
        )
    }

    fn generate(&self, seed: u64) -> Panel {
        use rand::rngs::SmallRng;
        use rand::{RngExt, SeedableRng};

        let mut rng = SmallRng::seed_from_u64(seed);
        let n_indiv = (self.n_obs / self.depth).max(1);
        let n_firm = (n_indiv / self.per_firm).max(1);
        let block = (self.n_obs / n_firm).max(1);

        let mut indiv = Vec::with_capacity(self.n_obs);
        let mut period = Vec::with_capacity(self.n_obs);
        let mut firm = Vec::with_capacity(self.n_obs);
        for i in 0..self.n_obs {
            indiv.push((i / self.depth) as u32);
            period.push((i % self.depth) as u32);
            firm.push(match self.firms {
                Firms::Spread => (i % n_firm) as u32,
                Firms::Random => rng.random_range(0..n_firm) as u32,
                Firms::Blocked => ((i / block).min(n_firm - 1)) as u32,
            });
        }

        let levels: Vec<Vec<u32>> = if self.n_fe == 2 {
            vec![indiv, period]
        } else {
            vec![indiv, period, firm]
        };
        finish(levels, &mut rng)
    }
}

/// Iterations the probe spends under the diagonal before the rule decides. A
/// diagonal that will finish dips below the threshold by iteration 8 at the
/// latest; shorter windows misread the slow-turning designs as stalled.
const PROBE: usize = 8;
/// A contraction window staying above this means the diagonal is stalling.
/// The classes nearly touch — 0.428 vs 0.493 across five design families — so
/// this sits mid-gap with only ~7% either way. AKM mobility is a continuum, and
/// designs near the boundary are ones where both preconditioners cost the same.
const STALL: f64 = 0.46;

struct Outcome {
    cold_iters: usize,
    cold_total: f64,
    switched: bool,
    contraction: f64,
    rule_iters: usize,
    rule_total: f64,
    oracle_total: f64,
    oracle_is_diag: bool,
    diag_iters: usize,
}

fn has_slopes(design: &Design<'_>) -> bool {
    design.terms.iter().any(|t| {
        t.columns
            .iter()
            .any(|l| matches!(l, crate::domain::Loading::Covariate(_)))
    })
}

fn apply_rule(design: &Design<'_>, y: &[f64], probe: usize, stall: f64) -> Outcome {
    let op = DesignOperator::new(design, None);
    let b = op.weighted_rhs(y);

    let timed_build = |cfg| {
        let t = Instant::now();
        let p = build_preconditioner(design, None, Some(&cfg))
            .expect("build")
            .0
            .expect("preconditioner");
        (p, t.elapsed().as_secs_f64())
    };
    let (schwarz, schwarz_setup) = timed_build(PreconditionerConfig::default());
    let (diagonal, diagonal_setup) = timed_build(PreconditionerConfig::Diagonal);

    let timed = |p: &_, budget: usize| {
        let t = Instant::now();
        let r = mlsmr(&op, &b, p, TOL, budget, MlsmrOptions::default()).expect("solve");
        (r, t.elapsed().as_secs_f64())
    };

    let (cold, cold_solve) = timed(&schwarz, MAXITER);
    let (diag_full, diag_solve) = timed(&diagonal, DIAG_CAP);
    // Slope designs can defeat both preconditioners; infinity keeps such a leg
    // out of the oracle instead of silently scoring its truncated time as a win.
    let cold_total = if cold.converged {
        schwarz_setup + cold_solve
    } else {
        f64::INFINITY
    };
    // A diagonal that never finishes can never be the oracle, and staying on it
    // is unbounded — infinity makes both facts fall out of the comparisons.
    let diag_total = if diag_full.converged {
        diagonal_setup + diag_solve
    } else {
        f64::INFINITY
    };

    // A stalling diagonal contracts steadily; one that will finish dips hard on
    // some step, so the window minimum is what separates them. `EscalationRule`
    // reads that window from one run, so only the full-length run is charged.
    // Slope designs get no probe: the contraction classes overlap completely
    // there, so the window minimum carries no signal and the probe is pure cost.
    if has_slopes(design) {
        return Outcome {
            cold_iters: cold.iterations,
            cold_total,
            switched: true,
            contraction: f64::NAN,
            rule_iters: cold.iterations,
            rule_total: cold_total,
            oracle_total: cold_total.min(diag_total),
            oracle_is_diag: diag_total < cold_total,
            diag_iters: diag_full.iterations,
        };
    }

    let (probed, probe_solve) = timed(&diagonal, probe);
    let nq: Vec<f64> = (2..=probe)
        .map(|j| {
            mlsmr(&op, &b, &diagonal, TOL, j, MlsmrOptions::default())
                .expect("solve")
                .normal_eq_residual
        })
        .collect();
    let contraction = nq.windows(2).map(|w| w[1] / w[0]).fold(f64::MAX, f64::min);
    let switched = !probed.converged && contraction > stall;

    let (rule_iters, rule_total) = if switched {
        let t = Instant::now();
        let restarted = mlsmr(
            &op,
            &b,
            &schwarz,
            TOL,
            MAXITER,
            MlsmrOptions {
                warm_start: Some(&probed.x),
                ..Default::default()
            },
        )
        .expect("restart");
        let elapsed = if restarted.converged {
            diagonal_setup + probe_solve + schwarz_setup + t.elapsed().as_secs_f64()
        } else {
            f64::INFINITY
        };
        (probed.iterations + restarted.iterations, elapsed)
    } else {
        // Not switching means the probe iterations were simply the first few of
        // an uninterrupted diagonal solve, so that solve is the whole cost.
        (diag_full.iterations, diag_total)
    };

    Outcome {
        cold_iters: cold.iterations,
        cold_total,
        switched,
        contraction,
        rule_iters,
        rule_total,
        oracle_total: cold_total.min(diag_total),
        oracle_is_diag: diag_total < cold_total,
        diag_iters: diag_full.iterations,
    }
}

#[test]
#[ignore = "measurement harness, not an assertion"]
fn validate_switching_rule() {
    let shapes = all_cases();
    for (probe, stall) in [(PROBE, STALL)] {
        let mut worst_vs_cold: f64 = 0.0;
        let mut worst_vs_oracle: f64 = 0.0;
        let (mut total_cold, mut total_rule, mut total_oracle) = (0.0, 0.0, 0.0);
        let mut wrong: Vec<String> = Vec::new();
        let mut unsolvable: Vec<String> = Vec::new();

        for shape in &shapes {
            for seed in [42u64, 7] {
                let panel = shape.generate(seed);
                let design = panel.design();
                let o = apply_rule(&design, &panel.y, probe, stall);
                if !o.oracle_total.is_finite() {
                    unsolvable.push(shape.label());
                    continue;
                }
                let vs_cold = o.rule_total / o.cold_total;
                let vs_oracle = o.rule_total / o.oracle_total;
                worst_vs_cold = worst_vs_cold.max(vs_cold);
                worst_vs_oracle = worst_vs_oracle.max(vs_oracle);
                total_cold += o.cold_total;
                total_rule += o.rule_total;
                total_oracle += o.oracle_total;
                if o.switched == o.oracle_is_diag {
                    wrong.push(format!(
                        "{}#{seed} {} (contr {:.3}, cold {} it / diag {} it / rule {} it, {vs_oracle:.1}x oracle)",
                        shape.label(),
                        if o.switched { "switched" } else { "stayed" },
                        o.contraction,
                        o.cold_iters,
                        o.diag_iters,
                        o.rule_iters,
                    ));
                }
            }
        }

        println!("\n=== probe {probe} iters, stall threshold {stall} ===");
        println!(
            "  rule {total_rule:.3}s | cold {total_cold:.3}s ({:.2}x) | oracle {total_oracle:.3}s ({:.2}x)",
            total_rule / total_cold,
            total_rule / total_oracle,
        );
        println!("  worst vs cold {worst_vs_cold:.2}x | worst vs oracle {worst_vs_oracle:.2}x | misclassified {}", wrong.len());
        for w in &wrong {
            println!("    {w}");
        }
        if !unsolvable.is_empty() {
            println!("  neither preconditioner converged: {unsolvable:?}");
        }
    }
}

/// What switching costs on designs where the diagonal is genuinely hopeless.
/// Best-of-`REPS` per component, because single-shot ratios on these sizes sit
/// inside the machine's noise band.
#[test]
#[ignore = "measurement harness, not an assertion"]
fn hard_case_loss() {
    const REPS: usize = 5;
    let shapes = [
        (50_000usize, 10usize, 23usize),
        (200_000, 10, 23),
        (1_000_000, 10, 23),
        (2_000_000, 10, 23),
        (200_000, 4, 200),
        (1_000_000, 4, 200),
    ];

    println!(
        "{:>18} | diag it | cold setup+solve=total (it) | rule dsetup+probe+ssetup+warm=total (it) | overhead",
        "design"
    );
    for (n_obs, depth, per_firm) in shapes {
        let shape = Shape {
            n_obs,
            n_fe: 3,
            depth,
            per_firm,
            firms: Firms::Spread,
        };
        let panel = shape.generate(42);
        let design = panel.design();
        let op = DesignOperator::new(&design, None);
        let b = op.weighted_rhs(&panel.y);

        let mut best = [f64::MAX; 5];
        let mut cold_iters = 0;
        let mut warm_iters = 0;
        let mut diag_iters = 0;
        for _ in 0..REPS {
            let build = |cfg| {
                let t = Instant::now();
                let p = build_preconditioner(&design, None, Some(&cfg))
                    .expect("build")
                    .0
                    .expect("preconditioner");
                (p, t.elapsed().as_secs_f64())
            };
            let (schwarz, s_setup) = build(PreconditionerConfig::default());
            let (diagonal, d_setup) = build(PreconditionerConfig::Diagonal);

            let t = Instant::now();
            let cold =
                mlsmr(&op, &b, &schwarz, TOL, MAXITER, MlsmrOptions::default()).expect("cold");
            let cold_solve = t.elapsed().as_secs_f64();

            let t = Instant::now();
            let probe =
                mlsmr(&op, &b, &diagonal, TOL, PROBE, MlsmrOptions::default()).expect("probe");
            let probe_time = t.elapsed().as_secs_f64();

            let t = Instant::now();
            let warm = mlsmr(
                &op,
                &b,
                &schwarz,
                TOL,
                MAXITER,
                MlsmrOptions {
                    warm_start: Some(&probe.x),
                    ..Default::default()
                },
            )
            .expect("warm");
            let warm_solve = t.elapsed().as_secs_f64();
            assert!(cold.converged && warm.converged, "leg stalled");

            for (slot, v) in best
                .iter_mut()
                .zip([s_setup, cold_solve, d_setup, probe_time, warm_solve])
            {
                *slot = slot.min(v);
            }
            cold_iters = cold.iterations;
            warm_iters = warm.iterations;
            if diag_iters == 0 {
                diag_iters = mlsmr(&op, &b, &diagonal, TOL, MAXITER, MlsmrOptions::default())
                    .expect("diag")
                    .iterations;
            }
        }

        let [s_setup, cold_solve, d_setup, probe_time, warm_solve] = best;
        let cold_total = s_setup + cold_solve;
        let rule_total = d_setup + probe_time + s_setup + warm_solve;
        println!(
            "{:>18} | {diag_iters:>7} | {s_setup:.3}+{cold_solve:.3}={cold_total:.3} ({cold_iters:>2}) \
             | {d_setup:.3}+{probe_time:.3}+{s_setup:.3}+{warm_solve:.3}={rule_total:.3} ({:>2}) \
             | +{:.3}s {:.2}x",
            shape.label(),
            PROBE + warm_iters,
            rule_total - cold_total,
            rule_total / cold_total,
        );
    }
}

/// Searches for a probe statistic that separates "diagonal will finish" from
/// "diagonal is hopeless", against the oracle each design actually wants.
#[test]
#[ignore = "measurement harness, not an assertion"]
fn separating_statistic() {
    const DEPTH: usize = 8;
    let mut rows: Vec<(bool, String, Vec<f64>)> = Vec::new();

    for shape in all_cases() {
        for seed in [42u64, 7] {
            let panel = shape.generate(seed);
            let design = panel.design();
            let op = DesignOperator::new(&design, None);
            let b = op.weighted_rhs(&panel.y);

            let build = |cfg| {
                let t = Instant::now();
                let p = build_preconditioner(&design, None, Some(&cfg))
                    .expect("build")
                    .0
                    .expect("preconditioner");
                (p, t.elapsed().as_secs_f64())
            };
            let (schwarz, s_setup) = build(PreconditionerConfig::default());
            let (diagonal, d_setup) = build(PreconditionerConfig::Diagonal);

            let t = Instant::now();
            let cold =
                mlsmr(&op, &b, &schwarz, TOL, MAXITER, MlsmrOptions::default()).expect("cold");
            let cold_total = s_setup + t.elapsed().as_secs_f64();
            let t = Instant::now();
            let diag =
                mlsmr(&op, &b, &diagonal, TOL, DIAG_CAP, MlsmrOptions::default()).expect("diag");
            let cold_total = if cold.converged {
                cold_total
            } else {
                f64::INFINITY
            };
            let diag_total = if diag.converged {
                d_setup + t.elapsed().as_secs_f64()
            } else {
                f64::INFINITY
            };

            let nq: Vec<f64> = (1..=DEPTH)
                .map(|j| {
                    mlsmr(&op, &b, &diagonal, TOL, j, MlsmrOptions::default())
                        .expect("probe")
                        .normal_eq_residual
                })
                .collect();
            let contractions: Vec<f64> = nq.windows(2).map(|w| w[1] / w[0]).collect();
            rows.push((
                diag_total < cold_total,
                format!("{}#{seed}", shape.label()),
                contractions,
            ));
        }
    }

    // `c[k]` is the contraction entering iteration k+2, so index 1 is j=3.
    let stats = |c: &[f64]| {
        let win = &c[1..];
        let min = win.iter().cloned().fold(f64::MAX, f64::min);
        let max = win.iter().cloned().fold(0.0f64, f64::max);
        (c[1], min, max)
    };

    for want_diag in [true, false] {
        println!(
            "\n=== oracle prefers {} ===",
            if want_diag { "DIAGONAL" } else { "SCHWARZ" }
        );
        println!(
            "{:>28} |    c3 | min(c3..c5) | max(c3..c5) | trajectory",
            "design"
        );
        let mut worst_min = if want_diag { 0.0f64 } else { f64::MAX };
        for (is_diag, label, c) in rows.iter().filter(|r| r.0 == want_diag) {
            let (c3, min, max) = stats(c);
            worst_min = if *is_diag {
                worst_min.max(min)
            } else {
                worst_min.min(min)
            };
            let traj: Vec<String> = c.iter().map(|v| format!("{v:.2}")).collect();
            println!(
                "{label:>28} | {c3:>5.3} | {min:>11.3} | {max:>11.3} | {}",
                traj.join(" ")
            );
        }
        println!(
            "  -> {} min(c3..c8) in this class: {worst_min:.3}",
            if want_diag { "largest" } else { "smallest" }
        );
    }
}

/// The shape grid shared by the validation sweep and the statistic search.
fn sweep_shapes() -> Vec<Shape> {
    let mut shapes: Vec<Shape> = Vec::new();
    for &n_obs in &[50_000usize, 200_000, 1_000_000, 2_000_000] {
        for &n_fe in &[2usize, 3] {
            for &firms in &[Firms::Spread, Firms::Random, Firms::Blocked] {
                shapes.push(Shape {
                    n_obs,
                    n_fe,
                    depth: 10,
                    per_firm: 23,
                    firms,
                });
            }
        }
    }
    for &n_obs in &[200_000usize, 1_000_000] {
        for &depth in &[4usize, 40] {
            for &per_firm in &[5usize, 200] {
                shapes.push(Shape {
                    n_obs,
                    n_fe: 3,
                    depth,
                    per_firm,
                    firms: Firms::Spread,
                });
            }
        }
    }
    shapes
}

/// Design families beyond the worker/period/firm panel, chosen because each
/// stresses conditioning differently: mobility connectivity, level-size skew,
/// ragged panels, and factor count.
enum Case {
    Panel(Shape),
    /// AKM worker-firm mobility: careers that move firm with probability
    /// `mobility` each period. Low mobility weakly connects the bipartite
    /// worker-firm graph, which is what makes AKM hard.
    Akm {
        n_obs: usize,
        depth: usize,
        mobility: f64,
    },
    /// Firm sizes from a power law — a few huge firms, a long tail of tiny ones.
    Zipf {
        n_obs: usize,
        depth: usize,
        exponent: f64,
    },
    /// Ragged panel: observations per worker vary uniformly in `1..=max_depth`.
    Unbalanced {
        n_obs: usize,
        max_depth: usize,
    },
    /// Panel or AKM careers with continuous loadings attached to `slope_on` —
    /// the regime this project already knows degrades the Schwarz preconditioner.
    Slopes {
        n_obs: usize,
        depth: usize,
        /// `Some(p)` builds AKM careers with move probability `p`.
        mobility: Option<f64>,
        slope_on: &'static [usize],
    },
    /// Four-way: worker, period, firm, occupation.
    FourWay {
        n_obs: usize,
        depth: usize,
        n_occ: usize,
    },
}

/// Owned columns for one generated design. `Design` borrows its inputs, so the
/// storage has to outlive it — slope loadings make this unavoidable.
struct Panel {
    levels: Vec<Vec<u32>>,
    /// Continuous loading for the like-indexed factor, if it carries a slope.
    slopes: Vec<Option<Vec<f64>>>,
    y: Vec<f64>,
}

impl Panel {
    fn design(&self) -> Design<'_> {
        let effects = self.levels.iter().zip(&self.slopes).map(|(levels, slope)| {
            let cols: Vec<&[f64]> = slope.iter().map(|z| &z[..]).collect();
            Effect::new(levels, true, cols).expect("valid effect")
        });
        Design::new(effects).expect("valid design")
    }
}

fn finish(levels: Vec<Vec<u32>>, rng: &mut rand::rngs::SmallRng) -> Panel {
    let n = levels.len();
    slopes_on(levels, &[], n, rng)
}

/// Attach a positive tenure-like loading to each factor listed in `on`.
fn slopes_on(
    levels: Vec<Vec<u32>>,
    on: &[usize],
    n_factors: usize,
    rng: &mut rand::rngs::SmallRng,
) -> Panel {
    use rand::RngExt;
    let n_obs = levels[0].len();
    let slopes = (0..n_factors)
        .map(|f| {
            on.contains(&f).then(|| {
                (0..n_obs)
                    .map(|_| rng.random_range(0.5..1.5))
                    .collect::<Vec<f64>>()
            })
        })
        .collect();
    let y = (0..n_obs).map(|_| rng.random_range(-1.0..1.0)).collect();
    Panel { levels, slopes, y }
}

impl Case {
    fn label(&self) -> String {
        match self {
            Case::Panel(s) => s.label(),
            Case::Akm {
                n_obs,
                depth,
                mobility,
            } => format!("{}k/akm/d{depth}/m{mobility}", n_obs / 1000),
            Case::Zipf {
                n_obs,
                depth,
                exponent,
            } => format!("{}k/zipf/d{depth}/s{exponent}", n_obs / 1000),
            Case::Unbalanced { n_obs, max_depth } => {
                format!("{}k/unbal/d1-{max_depth}", n_obs / 1000)
            }
            Case::Slopes {
                n_obs,
                depth,
                mobility,
                slope_on,
            } => format!(
                "{}k/{}/d{depth}/slope{slope_on:?}",
                n_obs / 1000,
                match mobility {
                    Some(m) => format!("akm{m}"),
                    None => "panel".to_string(),
                }
            ),
            Case::FourWay {
                n_obs,
                depth,
                n_occ,
            } => format!("{}k/4FE/d{depth}/o{n_occ}", n_obs / 1000),
        }
    }

    fn generate(&self, seed: u64) -> Panel {
        use rand::rngs::SmallRng;
        use rand::{RngExt, SeedableRng};
        let mut rng = SmallRng::seed_from_u64(seed);

        match *self {
            Case::Panel(ref s) => s.generate(seed),

            Case::Akm {
                n_obs,
                depth,
                mobility,
            } => {
                let n_worker = (n_obs / depth).max(1);
                let n_firm = (n_worker / 23).max(2);
                let (mut w, mut t, mut f) = (
                    Vec::with_capacity(n_obs),
                    Vec::with_capacity(n_obs),
                    Vec::with_capacity(n_obs),
                );
                for worker in 0..n_worker {
                    let mut firm = rng.random_range(0..n_firm);
                    for period in 0..depth {
                        if period > 0 && rng.random_range(0.0..1.0) < mobility {
                            firm = rng.random_range(0..n_firm);
                        }
                        w.push(worker as u32);
                        t.push(period as u32);
                        f.push(firm as u32);
                    }
                }
                finish(vec![w, t, f], &mut rng)
            }

            Case::Zipf {
                n_obs,
                depth,
                exponent,
            } => {
                let n_worker = (n_obs / depth).max(1);
                let n_firm = (n_worker / 23).max(2);
                let weights: Vec<f64> = (1..=n_firm).map(|r| (r as f64).powf(-exponent)).collect();
                let mut cdf = Vec::with_capacity(n_firm);
                let mut acc = 0.0;
                for wt in &weights {
                    acc += wt;
                    cdf.push(acc);
                }
                let (mut w, mut t, mut f) = (
                    Vec::with_capacity(n_obs),
                    Vec::with_capacity(n_obs),
                    Vec::with_capacity(n_obs),
                );
                for i in 0..n_obs {
                    let u = rng.random_range(0.0..acc);
                    let firm = cdf.partition_point(|&c| c < u).min(n_firm - 1);
                    w.push((i / depth) as u32);
                    t.push((i % depth) as u32);
                    f.push(firm as u32);
                }
                finish(vec![w, t, f], &mut rng)
            }

            Case::Unbalanced { n_obs, max_depth } => {
                let (mut w, mut t, mut f) = (
                    Vec::with_capacity(n_obs),
                    Vec::with_capacity(n_obs),
                    Vec::with_capacity(n_obs),
                );
                let n_firm = (n_obs / (max_depth * 12)).max(2);
                let mut worker = 0u32;
                while w.len() < n_obs {
                    let d = rng.random_range(1..=max_depth);
                    let firm = rng.random_range(0..n_firm) as u32;
                    for period in 0..d {
                        if w.len() == n_obs {
                            break;
                        }
                        w.push(worker);
                        t.push(period as u32);
                        f.push(firm);
                    }
                    worker += 1;
                }
                finish(vec![w, t, f], &mut rng)
            }

            Case::Slopes {
                n_obs,
                depth,
                mobility,
                slope_on,
            } => {
                let n_worker = (n_obs / depth).max(1);
                let n_firm = (n_worker / 23).max(2);
                let (mut w, mut t, mut f) = (
                    Vec::with_capacity(n_obs),
                    Vec::with_capacity(n_obs),
                    Vec::with_capacity(n_obs),
                );
                for worker in 0..n_worker {
                    let mut firm = rng.random_range(0..n_firm);
                    for period in 0..depth {
                        let moved = mobility.is_some_and(|m| rng.random_range(0.0..1.0) < m);
                        if period > 0 && moved {
                            firm = rng.random_range(0..n_firm);
                        } else if mobility.is_none() {
                            firm = w.len() % n_firm;
                        }
                        w.push(worker as u32);
                        t.push(period as u32);
                        f.push(firm as u32);
                    }
                }
                slopes_on(vec![w, t, f], slope_on, 3, &mut rng)
            }
            Case::FourWay {
                n_obs,
                depth,
                n_occ,
            } => {
                let n_worker = (n_obs / depth).max(1);
                let n_firm = (n_worker / 23).max(2);
                let (mut w, mut t, mut f, mut o) = (
                    Vec::with_capacity(n_obs),
                    Vec::with_capacity(n_obs),
                    Vec::with_capacity(n_obs),
                    Vec::with_capacity(n_obs),
                );
                for i in 0..n_obs {
                    w.push((i / depth) as u32);
                    t.push((i % depth) as u32);
                    f.push(rng.random_range(0..n_firm) as u32);
                    o.push(rng.random_range(0..n_occ) as u32);
                }
                finish(vec![w, t, f, o], &mut rng)
            }
        }
    }
}

/// Every family and size the rule is validated against.
fn all_cases() -> Vec<Case> {
    let mut cases: Vec<Case> = sweep_shapes().into_iter().map(Case::Panel).collect();
    for &n_obs in &[200_000usize, 1_000_000, 2_000_000] {
        for &mobility in &[0.02f64, 0.1, 0.3, 0.8] {
            cases.push(Case::Akm {
                n_obs,
                depth: 10,
                mobility,
            });
        }
        for &exponent in &[0.6f64, 1.2] {
            cases.push(Case::Zipf {
                n_obs,
                depth: 10,
                exponent,
            });
        }
        cases.push(Case::Unbalanced {
            n_obs,
            max_depth: 20,
        });
        for slope_on in [&[2usize][..], &[0][..], &[0, 2][..]] {
            for mobility in [None, Some(0.1)] {
                cases.push(Case::Slopes {
                    n_obs,
                    depth: 10,
                    mobility,
                    slope_on,
                });
            }
        }
        for &n_occ in &[8usize, 400] {
            cases.push(Case::FourWay {
                n_obs,
                depth: 10,
                n_occ,
            });
        }
    }
    cases
}
