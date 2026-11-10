"""The one-pair uniform map survives Python serialization and solver reuse."""
import pickle

import numpy as np

from within import PreconditionerConfig, Solver


def test_uniform_preconditioner_pickle_and_reuse():
    categories = np.array(
        [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]], dtype=np.uint32
    )
    y = np.array([1.0, -2.0, 0.5, 3.0, -1.5, 2.5])
    original_solver = Solver(categories, preconditioner=PreconditionerConfig.Additive())
    original = original_solver.preconditioner
    expected = original_solver.solve(y)
    assert expected.converged
    restored = pickle.loads(pickle.dumps(original))
    vector = np.arange(original.ncols, dtype=np.float64) + 0.25
    np.testing.assert_array_equal(restored.apply(vector), original.apply(vector))
    assert restored.config == original.config
    assert restored.build_duration_seconds == original.build_duration_seconds
    for _ in range(2):
        result = Solver(categories, preconditioner=restored).solve(y)
        assert result.converged
        np.testing.assert_allclose(result.x, expected.x, rtol=0, atol=1e-9)
