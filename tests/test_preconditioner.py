from __future__ import annotations

import pickle

import numpy as np
import pytest
from conftest import as_solver_categories
from within import (
    Effect,
    LsmrOptions,
    Preconditioner,
    PreconditionerConfig,
    Rung,
    Solver,
    solve,
)
from within._within import (
    ApproxCholConfig,
    ApproxSchurConfig,
    CoarseBuild,
    CoarseCorrection,
    CoarseSpace,
    LocalSolverConfig,
    ScalingConfig,
    Schur,
    Staleness,
    TwoLevelForm,
)


@pytest.fixture()
def problem():
    """Two-factor problem with 50 levels each, 10k observations."""
    np.random.seed(42)
    cats = [
        np.random.randint(0, 50, size=10_000),
        np.random.randint(0, 50, size=10_000),
    ]
    y = np.random.randn(10_000)
    return cats, y


@pytest.fixture()
def solver_and_precond(problem):
    """Build a Schwarz Solver and extract its preconditioner."""
    cats, y = problem
    categories = as_solver_categories(cats)
    solver = Solver(categories, preconditioner=PreconditionerConfig.Schwarz())
    precond = solver.preconditioner
    return solver, precond, categories, y


class TestAdvancedConfigs:
    def test_preconditioner_config_additive_constructor(self):
        implicit = PreconditionerConfig.Schwarz()
        explicit = PreconditionerConfig.Schwarz(
            local_solver=LocalSolverConfig(),
        )

        assert implicit == explicit
        assert implicit != PreconditionerConfig.Diagonal()
        assert implicit.local_solver == LocalSolverConfig()

        with pytest.raises(TypeError):
            PreconditionerConfig.Schwarz(local_solver=None)

    def test_preconditioner_config_variant_introspection(self):
        # Each variant is a subclass of PreconditionerConfig (the 3.9-safe
        # equivalent of ``match``), and Schwarz exposes its fields as getters.
        off = PreconditionerConfig.Off()
        tuned = PreconditionerConfig.Schwarz(
            local_solver=LocalSolverConfig(dense_threshold=0),
        )

        assert isinstance(off, PreconditionerConfig)
        assert isinstance(tuned, PreconditionerConfig.Schwarz)
        assert not isinstance(off, PreconditionerConfig.Schwarz)
        assert tuned.local_solver.dense_threshold == 0

    def test_approx_chol_config_defaults(self):
        cfg = ApproxCholConfig()
        assert cfg.seed == 0
        assert cfg.split_merge is None

    def test_approx_chol_config_custom(self):
        cfg = ApproxCholConfig(seed=42, split_merge=2)
        assert cfg.seed == 42
        assert cfg.split_merge == 2

    def test_approx_schur_config_defaults(self):
        cfg = ApproxSchurConfig()
        assert cfg.split == 1

    def test_schur_complement_defaults(self):
        sc = LocalSolverConfig()
        assert sc.dense_threshold == 24
        assert sc.approx_chol.seed == 0
        assert sc.approx_chol.split_merge == 2
        assert repr(sc.schur) == "Schur.approximate(split=1)"
        assert sc.scaling.on_failure == "warn"

    def test_schur_mode_spellings(self):
        # Omission and Schur.approximate() are the default; Schur.exact() is the
        # explicit opt-in — all three build and solve.
        rng = np.random.default_rng(7)
        cats = as_solver_categories(
            [rng.integers(0, 12, size=600), rng.integers(0, 9, size=600)]
        )
        y = rng.standard_normal(600)
        for schur in (None, Schur.approximate(), Schur.exact()):
            local = LocalSolverConfig(schur=schur)
            expected = Schur.approximate() if schur is None else schur
            assert repr(local.schur) == repr(expected)
            result = solve(
                cats,
                y,
                options=LsmrOptions(maxiter=2000),
                preconditioner=PreconditionerConfig.Schwarz(local_solver=local),
            )
            assert result.converged

    def test_schur_complement_solve(self, problem):
        cats, y = problem
        result = solve(
            as_solver_categories(cats),
            y,
            options=LsmrOptions(),
            preconditioner=PreconditionerConfig.Schwarz(
                local_solver=LocalSolverConfig()
            ),
        )
        assert result.converged


class TestFePreconditioner:
    def test_preconditioner_apply(self, solver_and_precond):
        solver, precond, categories, y = solver_and_precond
        x = np.random.randn(precond.nrows)
        result = precond.apply(x)
        assert result.shape == (precond.nrows,)
        assert np.all(np.isfinite(result))

    def test_preconditioner_apply_wrong_length_raises(self, solver_and_precond):
        solver, precond, categories, y = solver_and_precond
        x = np.random.randn(precond.nrows + 5)
        with pytest.raises(ValueError):
            precond.apply(x)

    def test_preconditioner_repr_schwarz(self, solver_and_precond):
        solver, precond, categories, y = solver_and_precond
        assert "Schwarz" in repr(precond)

    def test_preconditioner_acts_on_all_but_the_largest_factor(
        self, solver_and_precond
    ):
        solver, precond, categories, y = solver_and_precond
        levels = [len(np.unique(column)) for column in categories.T]
        assert precond.nrows == precond.ncols == solver.n_dofs - max(levels)

    def test_preconditioner_exposes_build_duration(self):
        categories = as_solver_categories(
            [np.array([0, 1, 0, 1, 2, 2]), np.array([0, 0, 1, 1, 0, 1])]
        )

        for config in (
            PreconditionerConfig.Schwarz(),
            PreconditionerConfig.Diagonal(),
        ):
            precond = Solver(categories, preconditioner=config).preconditioner
            assert precond is not None
            assert isinstance(precond.build_duration_seconds, float)
            assert np.isfinite(precond.build_duration_seconds)
            assert precond.build_duration_seconds >= 0.0

    def test_preconditioner_constructor_roundtrip(self, solver_and_precond):
        solver, precond, categories, y = solver_and_precond
        data = pickle.dumps(precond)
        precond2 = pickle.loads(data)
        x = np.random.randn(precond.nrows)
        np.testing.assert_array_equal(precond.apply(x), precond2.apply(x))
        assert precond2.config == precond.config
        assert precond2.build_duration_seconds == precond.build_duration_seconds

    def test_preconditioner_exposes_requested_config(self, solver_and_precond):
        solver, precond, categories, y = solver_and_precond
        assert precond.config == PreconditionerConfig.Schwarz()

    def test_preconditioner_exposes_tuned_config(self, problem):
        cats, _ = problem
        requested = PreconditionerConfig.Schwarz(
            local_solver=LocalSolverConfig(
                approx_chol=ApproxCholConfig(seed=41, split_merge=3),
                schur=Schur.approximate(ApproxSchurConfig(split=2)),
                dense_threshold=0,
            ),
        )

        solver = Solver(as_solver_categories(cats), preconditioner=requested)
        precond = solver.preconditioner

        assert precond is not None
        actual = precond.config
        assert actual == requested
        assert isinstance(actual, PreconditionerConfig.Schwarz)
        assert actual.local_solver.approx_chol.seed == 41
        assert actual.local_solver.approx_chol.split_merge == 3
        assert repr(actual.local_solver.schur) == "Schur.approximate(split=2)"
        assert actual.local_solver.dense_threshold == 0
        assert actual.local_solver.scaling.on_failure == "warn"

    def test_preconditioner_corrupt_bytes_raises(self):
        with pytest.raises(ValueError):
            Preconditioner(b"garbage")

    def test_preconditioner_apply_deterministic(self, solver_and_precond):
        solver, precond, categories, y = solver_and_precond
        x = np.random.randn(precond.nrows)
        r1 = precond.apply(x)
        r2 = precond.apply(x)
        np.testing.assert_array_equal(r1, r2)

    def test_preconditioner_repr_diagonal(self):
        categories = as_solver_categories(
            [np.array([0, 1, 0, 1, 2, 2]), np.array([0, 0, 1, 1, 0, 1])]
        )
        solver = Solver(categories, preconditioner=PreconditionerConfig.Diagonal())
        precond = solver.preconditioner

        assert precond is not None
        assert "Diagonal" in repr(precond)
        assert precond.config == PreconditionerConfig.Diagonal()

    def test_single_factor_diagonal_preconditioner_is_cached(self):
        categories = np.asfortranarray(
            np.array([[0], [1], [0], [2], [1]], dtype=np.uint32)
        )
        solver = Solver(categories, preconditioner=PreconditionerConfig.Diagonal())
        precond = solver.preconditioner

        assert precond is not None
        # The one factor is eliminated, so the map acts on no coefficients.
        assert precond.nrows == precond.ncols == 0


class TestAdaptive:
    """Binding surface only; escalation behaviour is covered in crates/within/tests/adaptive.rs."""

    def test_staleness_exposes_its_fields_and_rejects_invalid(self):
        assert (Staleness().window, Staleness().threshold) == (4, 0.7)
        assert Staleness(window=3, threshold=0.25) != Staleness()
        assert repr(Staleness()) == "Staleness(window=4, threshold=0.7)"
        # StalenessError must arrive as ValueError, not a bare RuntimeError.
        with pytest.raises(ValueError):
            Staleness(window=0)

    def test_adaptive_config_round_trips_through_getters(self):
        config = PreconditionerConfig.Adaptive(
            local_solver=LocalSolverConfig(dense_threshold=8),
            stall=Staleness(window=3, threshold=0.25),
        )
        assert isinstance(config, PreconditionerConfig)
        assert config.local_solver.dense_threshold == 8
        assert config.stall == Staleness(window=3, threshold=0.25)

    def test_solver_hands_out_the_tuned_ladder_before_escalating(self, problem):
        cats, _ = problem
        requested = PreconditionerConfig.Adaptive(
            local_solver=LocalSolverConfig(dense_threshold=8),
            stall=Staleness(window=3, threshold=0.25),
        )
        solver = Solver(as_solver_categories(cats), preconditioner=requested)
        assert solver.preconditioner.config == requested
        assert pickle.loads(pickle.dumps(solver.preconditioner)).config == requested

    def test_escalation_is_reachable_and_reported(self, problem):
        cats, y = problem
        solver = Solver(
            as_solver_categories(cats),
            preconditioner=PreconditionerConfig.Adaptive(
                stall=Staleness(window=1, threshold=0.0)
            ),
        )
        assert solver.preconditioner.rung == Rung.Diagonal
        assert solver.solve(y, LsmrOptions(tol=1e-10, maxiter=2000)).converged
        assert solver.preconditioner.rung == Rung.Schwarz

    def test_deferred_build_warnings_surface_on_the_escalating_solve(self, recwarn):
        """The constructor cannot see them: the Schwarz build happens mid-solve."""
        # A third factor, so the reduced diagonal iterates long enough to stall.
        i = np.arange(24)
        f = (i % 6 // 3 + 2 * (i // 6)).astype(np.uint32)
        g = (i % 3 + 3 * (i // 6 % 2)).astype(np.uint32)
        h = ((5 * i + i // 7) % 4).astype(np.uint32)
        z = np.tile([-2.0, 1.0, 1.0, -1.0, -1.0, 2.0], 4) * (1 + 0.1 * (i // 6))
        y = np.sin(1.7 * i)
        effects = [Effect(f, True, [z]), Effect(g, True), Effect(h, True)]
        local_solver = LocalSolverConfig(scaling=ScalingConfig(max_iterations=0))
        options = LsmrOptions(tol=1e-12, maxiter=100)

        def unscalable(ws):
            return [w for w in ws if "dominance" in str(w.message)]

        Solver(
            effects,
            preconditioner=PreconditionerConfig.Schwarz(local_solver=local_solver),
        )
        warned = len(unscalable(recwarn.list))
        assert warned, "fixture no longer warns under Schwarz"

        recwarn.clear()
        solver = Solver(
            effects,
            preconditioner=PreconditionerConfig.Adaptive(
                local_solver=local_solver, stall=Staleness(window=1, threshold=0.0)
            ),
        )
        assert not unscalable(recwarn.list), "the build has not happened yet"

        recwarn.clear()
        solver.solve(y, options)
        assert solver.preconditioner.rung == Rung.Schwarz
        assert len(unscalable(recwarn.list)) == warned

        recwarn.clear()
        solver.solve(y, options)
        assert not recwarn.list, "a built rung must not re-warn on later solves"


def coarse(build):
    return CoarseCorrection(
        space=CoarseSpace.RigidComponents, form=TwoLevelForm.Balanced, build=build
    )


def clustered():
    """Workers in 40 firm clusters with a slope on both terms; each mover's one row elsewhere glues nothing."""
    spells = [(k, k // 3, 8) for k in range(120)]
    for k in range(30):
        a = (7 * k + 3) % 40
        spells += [(120 + k, a, 7), (120 + k, (a + 1 + (13 * k) % 39) % 40, 1)]
    worker = np.repeat([w for w, _, n in spells], [n for _, _, n in spells])
    firm = np.repeat([f for _, f, n in spells], [n for _, _, n in spells])
    rng = np.random.default_rng(11)
    effects = [
        Effect(worker.astype(np.uint32), True, [rng.uniform(-0.5, 0.5, worker.size)]),
        Effect(firm.astype(np.uint32), True, [rng.uniform(-0.5, 0.5, worker.size)]),
    ]
    return effects, rng.uniform(-0.5, 0.5, worker.size)


class TestCoarse:
    """Binding surface only; the two-level map is covered in crates/within/tests/reduced.rs."""

    def test_coarse_config_round_trips_through_getters(self):
        stall = Staleness(window=2, threshold=0.5)
        on_stall = coarse(CoarseBuild.OnStall(stall=stall))
        assert on_stall.space == CoarseSpace.RigidComponents
        assert on_stall.form == TwoLevelForm.Balanced
        assert isinstance(on_stall.build, CoarseBuild.OnStall)
        assert on_stall.build.stall == stall
        assert on_stall != coarse(CoarseBuild.WithSchwarz())
        assert PreconditionerConfig.Schwarz().coarse is None
        assert PreconditionerConfig.Adaptive(coarse=on_stall).coarse == on_stall
        assert (
            PreconditionerConfig.Schwarz(coarse=on_stall)
            != PreconditionerConfig.Schwarz()
        )

    @pytest.mark.parametrize("adaptive", [False, True])
    @pytest.mark.parametrize(
        "build",
        [CoarseBuild.WithSchwarz(), CoarseBuild.OnStall(stall=Staleness(1, 0.0))],
        ids=["with_schwarz", "on_stall"],
    )
    def test_a_coarse_correction_lands_the_solve_on_the_two_level_map(
        self, adaptive, build
    ):
        effects, y = clustered()
        if adaptive:
            requested = PreconditionerConfig.Adaptive(
                stall=Staleness(1, 0.0), coarse=coarse(build)
            )
        else:
            requested = PreconditionerConfig.Schwarz(coarse=coarse(build))
        options = LsmrOptions(tol=1e-12, maxiter=2000)
        solver = Solver(effects, preconditioner=requested)
        result = solver.solve(y, options)
        one_level = solve(
            effects, y, options=options, preconditioner=PreconditionerConfig.Schwarz()
        )
        assert solver.preconditioner.rung == Rung.TwoLevel
        assert solver.preconditioner.config == requested
        assert result.converged and one_level.converged
        np.testing.assert_allclose(result.demeaned, one_level.demeaned, atol=1e-9)

    def test_a_two_level_preconditioner_round_trips(self):
        effects, y = clustered()
        requested = PreconditionerConfig.Schwarz(
            coarse=coarse(CoarseBuild.WithSchwarz())
        )
        built = Solver(effects, preconditioner=requested)
        assert built.preconditioner.rung == Rung.TwoLevel
        restored = pickle.loads(pickle.dumps(built.preconditioner))
        assert restored.rung == Rung.TwoLevel
        assert restored.config == requested
        options = LsmrOptions(tol=1e-12, maxiter=2000)
        a = built.solve(y, options)
        b = Solver(effects, preconditioner=restored).solve(y, options)
        assert a.converged
        assert a.iterations == b.iterations
        np.testing.assert_array_equal(a.x, b.x)

    def test_a_declined_coarse_correction_warns_and_stays_one_level(self, problem):
        cats, _ = problem
        with pytest.warns(UserWarning, match="two-level Schwarz declined"):
            solver = Solver(
                as_solver_categories(cats),
                preconditioner=PreconditionerConfig.Schwarz(
                    coarse=coarse(CoarseBuild.WithSchwarz())
                ),
            )
        assert solver.preconditioner.rung == Rung.Schwarz
