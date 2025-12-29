# Simplex Method Solver

A Python implementation of the **Simplex Algorithm** (supporting Big-M) for Linear Programming problems.

**Dependencies:** `numpy`, `pandas`

**Usage:**

```python
from LinearProgramming import SimplexMethod

# Objective Function: Max Z = 5_x1 - 3_x2 - 3_x3
z = [5, -3, -3]

# Constraints: [coef_x1, coef_x2, ..., sign, rhs]
constraints = [
    [1, 1, 0, '<=', 5],
    [1, 1, 1, '<=', 7]
]

# Solve (use max=False for minimization)
solver = SimplexMethod(obj_func=z, constraints=constraints, max=True)
solver.solve()
```
**Features**:

- Maximization & Minimization

- Constraints: <=, >=, =

- Step-by-step Tableau visualization