from simplex_method import simplex_method

objective_func = [12, 8]

coefs = [[5, 2], [2, 3], [4, 2]]
signs = ['<=', '<=', '<=']
rhs = [150, 100, 80]

simplex_problem = simplex_method(obj_func=objective_func, coefs=coefs,
                                 signs=signs, rhs=rhs, max=False)
simplex_problem.solve()