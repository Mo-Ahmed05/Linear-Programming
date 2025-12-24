from simplex_method import simplex_method

objective_func = [1, 2, 1]

coefs = [[1, 1/2, 1/2], [3/2, 2, 1]]
signs = ['<=', '>=']
rhs = [1, 8]

simplex_problem = simplex_method(obj_func=objective_func, coefs=coefs,
                                 signs=signs, rhs=rhs, max=False)
simplex_problem.solve()