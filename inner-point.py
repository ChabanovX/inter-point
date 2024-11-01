import math
import numpy as np
from numpy.linalg import inv

np.set_printoptions(suppress=True)
# ignore division by zero errors, as we need to get infinities
np.seterr(divide='ignore')

def sign(n):
    return '+' if n > 0 else '-'

def interior_point(lpp: dict) -> tuple:
    c_objective = np.array(lpp["C"], dtype=float)
    a_constraints = np.array(lpp["A"], dtype=float)
    rhs = np.array(lpp["b"], dtype=float)
    precision = lpp["e"]
    maximize = lpp["max"]
    alpha = lpp["a"]
    
    # Number of constraints and original variables
    n_constraints = a_constraints.shape[0]
    n_vars = a_constraints.shape[1]
    # Initial feasible solution x0 with slack variables
    x = np.hstack((np.ones(n_vars), rhs - np.sum(a_constraints, axis=1)))
    # Extend A with identity matrix to add slack variables
    a_constraints = np.hstack((a_constraints, np.eye(n_constraints)))
    # Extend C with zeros for slack variables
    c_objective = np.hstack((c_objective, np.zeros(n_constraints))) * (1 if maximize else -1)
    while True:
        D = np.diag(x)
        a_tilde = a_constraints @ D 
        c_tilde = D @ c_objective 
        try:
            a_tilde_inv = inv(a_tilde @ a_tilde.T)
            P = np.identity(len(c_objective)) - a_tilde.T @ a_tilde_inv @ a_tilde
        except np.linalg.LinAlgError:
            print("Some dumb error. try increasing step size or decreasing precision :P")
            exit(1)
        c_p = P @ c_tilde
        nu = abs(np.min(c_p[c_p < 0]))
        x_tilde = np.ones(len(c_objective)) + (alpha / nu) * c_p
        x_star = D @ x_tilde
        # Exit condition
        if np.linalg.norm(x_star - x) <= precision:
            break
        x = x_star

    # Calculate the objective function value
    z = np.dot(c_objective, x)

    if not maximize:
        z = -z
        
    x = x[:n_vars]
    filtered_x = [(i, sol) for (i, sol) in list(enumerate(x)) if abs(sol) > precision]
    return filtered_x, z
    

def simplex(lpp: dict) -> tuple:
    # parsing input data
    c_objective: np.ndarray = np.array(lpp["C"], dtype=float)
    constraints: np.ndarray = np.array(lpp["A"], dtype=float)
    rhs: np.ndarray = np.array(lpp["b"], dtype=float)
    precision: float = lpp["e"]
    maximize: bool = lpp["max"]

    n_constraints: int = len(constraints)
    n_vars: int = len(c_objective)
    # initialize table with zeros
    table: np.ndarray = np.zeros((n_constraints + 1, n_vars + n_constraints + 1))
    # fill in z-row
    table[0, :n_vars] = -c_objective if maximize else c_objective
    # fill in basic variable rows
    table[1:, :-(1 + n_constraints)] = constraints
    # fill in rhs
    table[1:, -1] = rhs
    # fill in 1's for slack variables
    table[1:, n_vars:-1] = np.identity(n_constraints)
    # simplex loop while there are negative entries in the z-row
    while np.any(table[0, :-1] < -precision):
        # index of the minimal element in z-row
        pivot_column: int = np.argmin(table[0, :-1])
        
        # find ratios (rhs/pivot_column_element)
        ratios: np.ndarray = table[1:, -1] / table[1:, pivot_column]
        
        # problem is unbounded if all elements are < 0 or infinite
        if np.all((ratios < -precision) | np.isinf(ratios)):
            print("Unbounded problem!")
            exit(1)
        
        # ignore negative elements and zeros in the ratios
        ratios[ratios < precision] = np.inf

        degenerate = False

        # degenerate, if there are same raios
        for i, x in enumerate(ratios):
            for j, y in enumerate(ratios):
                if x == y == np.min(ratios) and i != j:
                    degenerate = True
                    break

        # ratios are built starting from the second row, so add one
        pivot_row: int = np.argmin(ratios) + 1

        pivot_element: float = table[pivot_row, pivot_column]
        # normalize pivot row
        table[pivot_row] /= pivot_element
        
        # do this idk
        for i in range(n_constraints + 1):
            if i != pivot_row:
                table[i] -= table[i, pivot_column] * table[pivot_row]

        if (degenerate):
            print("Degenerate solution!")
            break
    
    solution_indexes_values = []
    
    # collect solution
    for i in range(n_vars):
        # find a basic variable
        if (1 - precision < np.sum(table[1:, i])  <  1 + precision) and (np.max(table[1:, i]) == 1):
            # basic row is where the '1' is located
            basic_row: int = np.where(table[:, i] == 1)[0][0]
            basic_value: float = table[basic_row, -1]
            # place basic variable's index and its value
            solution_indexes_values.append((i, float(basic_value)))
    # if finidng min, multiply by -1
    z_value = float(table[0, -1]) * (1 if maximize else -1)
    return solution_indexes_values, z_value

def print_lpp(lpp: dict) -> None:
    max, c_objective, constraints, rhs, *_ = lpp.values()
    str_problem: str = f"Problem:\n{'Max' if max else 'Min'} z = "
    z_row = ' '.join(f'{sign(r)} {abs(r)}x{i + 1}' if r not in (-1, 1) else f'{sign(r)}x{i + 1}' for i, r in enumerate(c_objective))
    z_row = z_row[2:] if z_row.startswith('+') else f'-{z_row[2:]}'
    str_problem += z_row + "\nSubject to the constraints:\n"
    for row, sol in zip(constraints, rhs):
        lhs = ' '.join(f'{sign(r)} {abs(r)}x{i + 1}' if r not in (-1, 1) else f'{sign(r)} x{i + 1}' for i, r in enumerate(row))
        lhs = lhs[2:] if lhs.startswith('+') else f'-{lhs[2:]}'
        str_problem += f"{lhs} {'<=' if max else '>='} {sol}\n"
    print(str_problem[:-1])
    

def print_lpp_solution(res: tuple) -> None:
    output_str: str = f"z = {res[-1]:g},\n"
    for i, value in res[0]:
        output_str += f"x{i + 1} = {value:g},\n"
    output_str = output_str[:-2]
    print(output_str)

lpp = {
    "max": False,         # max or min - True or False
    "C": [-2, 2, -6],     # C - objective function coefficients
    "A": [                # A - constraint coefficients matrix
        [2, 1, -2],
        [1, 2, 4],
        [1, -1, 2],
    ],                    
    "b": [24, 23, 10],    # b - rhs of constraints
    "e": 1e-1,            # e - precision
    "a": 0.5              # alpha - step size
}

print_lpp(lpp)

print("____________________")
print("Simplex algorithm result:")
print_lpp_solution(simplex(lpp))

print("_____________________")
print("Interior-point algorithm (a=0.5) result:")
print_lpp_solution(interior_point(lpp))

lpp["a"] = 0.9
print("_____________________")
print("Interior-point algorithm (a=0.9) result:")
print_lpp_solution(interior_point(lpp))