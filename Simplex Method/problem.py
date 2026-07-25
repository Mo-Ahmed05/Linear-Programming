from simplex import SimplexSolver

z = [5, 4]

const = [[1,1,0,'<=',5],
         [1,1,1,'<=',7]]

simplex_problem = SimplexSolver(obj_func=z, constraints=const, max=True)
simplex_problem.solve()