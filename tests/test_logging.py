from __future__ import annotations

import logging

import numpy as np
import pytest

import within


def _problem() -> tuple[np.ndarray, np.ndarray]:
    categories = np.asfortranarray(
        np.array(
            [[0, 0], [1, 0], [0, 1], [1, 1], [2, 0], [2, 1], [3, 2], [0, 2], [3, 0]],
            dtype=np.uint32,
        )
    )
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 2.5, 7.0, 0.5, 4.5])
    return categories, y


def _messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage().split(" ", 1)[0] for r in caplog.records]


def test_solve_reports_phases_on_the_within_logger(
    caplog: pytest.LogCaptureFixture,
) -> None:
    categories, y = _problem()
    with caplog.at_level(logging.INFO, logger="within"):
        within.solve(categories, y)
    messages = _messages(caplog)
    assert {"design", "preconditioner", "solver", "solved"} <= set(messages)
    assert all(r.name.startswith("within") for r in caplog.records)


def test_level_raised_after_first_call_takes_effect(
    caplog: pytest.LogCaptureFixture,
) -> None:
    categories, y = _problem()
    solver = within.Solver(categories)
    with caplog.at_level(logging.WARNING, logger="within"):
        solver.solve(y)
        assert "solved" not in _messages(caplog)
        with caplog.at_level(logging.INFO, logger="within"):
            solver.solve(y)
    assert "solved" in _messages(caplog)


def test_batch_reports_per_rhs_and_never_per_iteration(
    caplog: pytest.LogCaptureFixture,
) -> None:
    categories, y = _problem()
    Y = np.stack([y, 2.0 * y], axis=1)
    with caplog.at_level(5):
        within.solve_batch(categories, Y)
    solved = [
        r.getMessage() for r in caplog.records if r.getMessage().startswith("solved")
    ]
    assert sorted(m.split(" rhs=")[1].split(" ")[0] for m in solved) == ["0", "1"]
    assert not [r for r in caplog.records if r.name.startswith("schwarz_precond")]


def test_single_solve_reports_every_iteration(caplog: pytest.LogCaptureFixture) -> None:
    categories, y = _problem()
    with caplog.at_level(5):
        result = within.solve(categories, y)
    iterations = [r for r in caplog.records if r.name == "schwarz_precond.lsmr"]
    assert len(iterations) == result.iterations
    assert any(r.getMessage().startswith("solving") for r in caplog.records)


class _RaiseOnSolved(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        if record.getMessage().startswith("solved"):
            raise RuntimeError("handler failed")


@pytest.fixture
def raising_on_solved() -> logging.Handler:
    logger = logging.getLogger("within")
    handler = _RaiseOnSolved(level=logging.INFO)
    previous = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    yield handler
    logger.removeHandler(handler)
    logger.setLevel(previous)


def test_raising_handler_surfaces_as_its_own_exception(
    raising_on_solved: logging.Handler,
) -> None:
    categories, y = _problem()
    solver = within.Solver(categories)
    with pytest.raises(RuntimeError, match="handler failed"):
        solver.solve(y)


def test_raising_handler_on_a_batch_worker_is_not_lost(
    raising_on_solved: logging.Handler,
) -> None:
    categories, y = _problem()
    solver = within.Solver(categories)
    Y = np.stack([y, 2.0 * y], axis=1)
    with pytest.raises(RuntimeError, match="handler failed"):
        solver.solve_batch(Y)
