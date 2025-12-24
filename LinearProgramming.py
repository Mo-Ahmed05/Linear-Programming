import numpy as np
import pandas as pd

class SimplexMethod:

    def __init__(self, obj_func, coefs, signs, rhs, max=True, big_m = 1e12):
        self.obj_func = np.array(obj_func)
        self.coefs = np.array(coefs, dtype=float)
        self.signs = signs
        self.rhs = rhs

        self.isMax = max
        self.m = -big_m if max else big_m

        self.Xb = [0]*len(coefs)
        self.n_decision_var = len(coefs[0])

        self.Cb = np.zeros(len(coefs), dtype=float)
        self.Cj = None
        self.tableau = None
        self.variables = []
        self.tableau_history = []
        self.pivot_history = []

        self.build_simplex_tableau()

    def build_simplex_tableau(self):

        self.variables = [f'X{i}' for i in range(self.n_decision_var)]
        n_slack, n_surplus, n_artificial = 0, 0, 0

        # Number of "New Variables" (slack + surplus + artificial)
        for sign in self.signs:
            if sign == '<=':    
                n_slack += 1
            elif sign == '>=':
                n_surplus += 1
                n_artificial += 1
            elif sign == '=':   
                n_artificial += 1

        self.variables = [f'X{i}' for i in range(1, self.n_decision_var+1)]
        self.variables += [f'S{i}' for i in range(1, n_slack+n_surplus+1)]
        self.variables += [f'a{i}' for i in range(1, n_artificial+1)] + ['RHS']

        # Number of total variables
        n_total_var = self.n_decision_var + n_slack + n_surplus + n_artificial

        # Number of total tableau columns (Note: +1 for the RHS column)
        n_cols = n_total_var + 1
        n_rows= len(self.rhs)

        self.tableau = np.zeros((n_rows, n_cols))               # Initializing the tableau
        self.tableau[:, :self.n_decision_var] = self.coefs      # Fill Decision Variables Coefficients
        self.tableau[:,-1] = self.rhs                           # Fill RHS values

        self.Cj = np.zeros(n_total_var)                         # Initializing the Objective Function Coefficients
        self.Cj[:self.n_decision_var] = self.obj_func           # Fill Decision Variables Coefficients

        available_idx = [i for i in range(self.n_decision_var, n_total_var)]
        for i, row in enumerate(self.tableau):

            if self.signs[i] == '<=':
                row[available_idx[0]] = 1                           # Add "Slack" to the enquality
                self.Xb[i] = self.variables[available_idx[0]]       # Track Base Variables columns
                available_idx.pop(0)

            elif self.signs[i] == '>=':
                row[available_idx[0]] = -1                          # Add "Surplus" to the enquality
                available_idx.pop(0)

                row[available_idx[-1]] = 1                          # Add "Artificial" to the enquality
                self.Cj[available_idx[-1]] = self.m
                self.Cb[i] = self.m
                self.Xb[i] = self.variables[available_idx[-1]]
                available_idx.pop(-1)

            elif self.signs[i] == '=':
                row[available_idx[-1]] = 1                          # Add "Artificial" to the enquality
                self.Cj[available_idx[-1]] = self.m
                self.Cb[i] = self.m
                self.Xb[i] = self.variables[available_idx[-1]]
                available_idx.pop(-1)

    def pivot(self, pivot_row_i, pivot_col_i):
        pivot_element = self.tableau[pivot_row_i, pivot_col_i]
        self.tableau[pivot_row_i] /=  pivot_element

        for i, row in enumerate(self.tableau):
            if i != pivot_row_i:
                row -= self.tableau[pivot_row_i] * row[pivot_col_i]

        self.Cb[pivot_row_i] = self.Cj[pivot_col_i]
        self.pivot_history.append([pivot_row_i, pivot_col_i])

        self.Xb[pivot_row_i] = self.variables[pivot_col_i]   # Update the Xb Variables column

    def solve(self):
        self.tableau_history.append(pd.DataFrame(np.copy(self.tableau), index=self.Xb, columns=self.variables))

        Zj = np.dot(self.Cb, self.tableau[:, :-1])
        Cj_Zj = self.Cj - Zj

        # Check for optimality
        if (self.isMax and np.all(Cj_Zj <= 1e-9)) or (not self.isMax and np.all(Cj_Zj >= -1e-9)):

            solution = True
            # Check Infeasibility (Artificial and Positive Value)
            for i, Xb in enumerate(self.Xb):
                if ("a" in Xb) and (self.tableau[i, -1] > 0): 
                    print("This problem has INFEASIBLE solution!\n")
                    solution = False
                
            for i, x in enumerate(self.variables[:-1]):
                if x not in self.Xb:
                    if np.isclose(Cj_Zj[i], 0):
                        print("This problem has Multi-Optimal solution!\n")

            self._print_solve(solution)
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
        self.solve()

    def _print_solve(self, has_solution=True):
        
        for i in range(len(self.tableau_history)):
            print(self.tableau_history[i])
            if i != len(self.tableau_history)-1:
                print(f'pivot column: {self.pivot_history[i][1]+1}   pivot row: {self.pivot_history[i][0]+1} \n')

        if has_solution:
            rhs_values = self.tableau[:,-1]
            print('\nThe decision variables values:')
            for i in range(self.n_decision_var):
                val = 0.0
                if f'X{i+1}' in self.Xb:
                    row_i = self.Xb.index(f'X{i+1}')
                    val = rhs_values[row_i]
                print(f'  X{i+1} = {val}')

            z_text = 'Maximum' if self.isMax else 'Minimum'
            print(f'{z_text} value = {np.dot(self.Cb, rhs_values)}')