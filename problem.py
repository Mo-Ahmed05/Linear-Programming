from LinearProgramming import SimplexMethod

z = [2,3]

const = [[1,2,'>=',8],
         [2,3,'<=',12]]


simplex_problem = SimplexMethod(obj_func=z, constraints=const, max=True)
simplex_problem.solve()