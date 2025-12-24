from LinearProgramming import SimplexMethod

objective_func = [2, 1]

coefs = [[1, 2], [2, 1]]
signs = ['<=', '<=']
rhs = [8, 12]

simplex_problem = SimplexMethod(obj_func=objective_func, coefs=coefs,
                                 signs=signs, rhs=rhs, max=True)
simplex_problem.solve()