import numpy as np

class simplex_method:

    def __init__(self, obj_func, coefs, signs, rhs, max=True, big_m = 1e12):
        self.obj_func = np.array(obj_func)
        self.coefs = np.array(coefs, dtype=float)
        self.signs = signs
        self.rhs = rhs

        self.isMax = max
        self.m = -big_m if max else big_m

        self.basis = [0]*len(coefs)
        self.n_decision_var = len(coefs[0])

        self.Cb = np.zeros(len(coefs), dtype=float)
        self.Cj = None
        self.tableau = None
        self.variables = []
        self.tableau_history = []
        self.pivot_history = []

        self.build_simplex_tableau()

    def build_simplex_tableau(self):

        self.variables = [f'X{i}' for i in self.n_decision_var]
        n_slack, n_surplus, artificial = 0

        # Number of "New Variables" (slack + surplus + artificial)
        n_new_vars = 0
        for sign in self.signs:
            if sign == '<=':    n_new_vars += 1  # 1 slack
            elif sign == '>=':  n_new_vars += 2  # 1 surplus, 1 artificial
            elif sign == '=':   n_new_vars += 1  # 1 artificial
        
        # Number of total variables
        n_total_var = self.n_decision_var + n_new_vars

        # Number of total tableau columns (Note: +1 for the RHS column)
        n_cols = n_total_var + 1
        n_rows= len(self.rhs)

        self.tableau = np.zeros((n_rows, n_cols))               # Initializing the tableau
        self.tableau[:, :self.n_decision_var] = self.coefs      # Fill Decision Variables Coefficients
        self.tableau[:,-1] = self.rhs                           # Fill RHS values

        self.Cj = np.zeros(n_total_var)                         # Initializing the Objective Function Coefficients
        self.Cj[:self.n_decision_var] = self.obj_func           # Fill Decision Variables Coefficients

        current_col = self.n_decision_var                       # A pointer to track columns
        for i, row in enumerate(self.tableau):

            if self.signs[i] == '<=':
                row[current_col] = 1            # Add "Slack" to the enquality
                self.basis[i] = current_col     # Track Base Variables columns
                current_col += 1

            elif self.signs[i] == '>=':
                row[current_col] = -1           # Add "Surplus" to the enquality
                current_col += 1

                row[current_col] = 1            # Add "Artificial" to the enquality
                self.Cj[current_col] = self.m
                self.Cb[i] = self.m
                self.basis[i] = current_col
                current_col += 1

            elif self.signs[i] == '=':
                row[current_col] = 1            # Add "Artificial" to the enquality
                self.Cj[current_col] = self.m
                self.Cb[i] = self.m
                self.basis[i] = current_col     # Track Artificial Variables columns
                current_col += 1

        self.tableau_history.append(np.copy(self.tableau))

    def pivot(self, pivot_row_i, pivot_col_i):
        pivot_element = self.tableau[pivot_row_i, pivot_col_i]
        self.tableau[pivot_row_i] /=  pivot_element

        for i, row in enumerate(self.tableau):
            if i != pivot_row_i:
                row -= self.tableau[pivot_row_i] * row[pivot_col_i]

        self.Cb[pivot_row_i] = self.Cj[pivot_col_i]
        self.pivot_history.append([pivot_row_i, pivot_col_i])

        self.basis[pivot_row_i] = pivot_col_i   # Update the Basis Variables column

    def solve(self):

        Zj = np.zeros(len(self.Cj) + 1)

        for i, row in enumerate(self.tableau):
            Zj += self.Cb[i] * row              # Multiply each Cb by its row

        Cj_Zj = self.Cj - Zj[:-1]

        # Check for optimality
        if (self.isMax and np.all(Cj_Zj <= 1e-9)) or (not self.isMax and np.all(Cj_Zj >= -1e-9)):
            
            # Check Infeasibility (Artificial and Positive Value)
            for i, base_i in enumerate(self.basis):
                if (abs(self.Cj[base_i]) == self.m) and (self.tableau[i, -1] > 0): 
                    print("This problem has INFEASIBLE solution!\n")
                    self._print_solve(False)
                    return

            self._print_solve()
            return
        
        # Find pivot column
        if self.isMax:  pivot_col_i = np.argmax(Cj_Zj)
        else:           pivot_col_i = np.argmin(Cj_Zj)

        # Find pivot row
        ratios = []
        for row in self.tableau:
            if row[pivot_col_i] > 1e-9 and row[-1] >= 0:
                ratio = row[-1] / row[pivot_col_i]
                ratios.append(ratio)
            else:
                ratios.append(np.inf)

        if np.all(np.isinf(ratios)):
            print("This problem has UNBOUNDED solution!")
            self._print_solve(False)
            return

        pivot_row_i = np.argmin(ratios)
        
        self.pivot(pivot_row_i, pivot_col_i)
        self.tableau_history.append(np.copy(self.tableau))
        self.solve()

    def _print_solve(self, has_solution=True):

        for i in range(len(self.pivot_history)):
            print(self.tableau_history[i])
            print(f'pivot column: {self.pivot_history[i][1]+1}   pivot row: {self.pivot_history[i][0]+1} \n')

        print(self.tableau, '\n')

        if has_solution:
            rhs_values = self.tableau[:,-1]
            print('The decision variables values:')
            for i in range(self.n_decision_var):
                val = 0.0
                if i in self.basis:
                    row_index = self.basis.index(i)
                    val = rhs_values[row_index]
                print(f' X{i+1} = {val}')

            z_text = 'Maximum' if self.isMax else 'Minimum'
            print(f' {z_text} value = {np.sum(np.dot(self.Cb, rhs_values))}')