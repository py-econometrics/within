import numpy as np
import within

categories = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.uint32)
y = np.array([1.0, 2.0, 3.0, 4.0])
result = within.solve(categories, y)
assert result.converged, "solve did not converge"
assert len(result.x) > 0, "empty coefficient vector"
print("within wheel: import + smoke solve OK")
