from simplex_method import SimplexMethod

objective_func = [12, 8]

coefs = [[5, 2], [2, 3], [4, 2]]
signs = ['<=', '<=', '<=']
rhs = [150, 100, 80]

simplex_problem = SimplexMethod(obj_func=objective_func, coefs=coefs,
                                 signs=signs, rhs=rhs, max=True)
simplex_problem.solve()