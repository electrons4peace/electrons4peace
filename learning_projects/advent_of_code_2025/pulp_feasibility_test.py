"""
PuLP Feasibility Testing for Integer Linear Programs
=====================================================

This file demonstrates how to test if there are feasible solutions to A*x ≤ b
where x is a vector of integer variables.

Key concepts:
- Feasibility problems: finding ANY solution (no optimization needed)
- Infeasible systems: constraint sets with no valid solutions
- Detecting infeasibility vs unboundedness
- Integer vs continuous relaxation
"""

import pulp
import numpy as np
from scipy.sparse import lil_matrix, coo_matrix


# ============================================================================
# Example 1: Basic feasibility test with integer variables
# ============================================================================
def example1_basic_feasibility():
    """
    Test if there exists an integer solution to A*x ≤ b

    System:
        x0 + 2*x1 ≤ 10
        3*x0 + x1 ≤ 15
        x0, x1 ≥ 0 (integers)
    """
    print("\n" + "="*70)
    print("Example 1: Basic Feasibility Test (FEASIBLE system)")
    print("="*70)

    # Create feasibility problem (no objective needed, but PuLP requires one)
    # We use a dummy objective of 0
    prob = pulp.LpProblem("Feasibility_Test_1", pulp.LpMinimize)

    # Integer variables
    x = [
        pulp.LpVariable("x_0", lowBound=0, cat=pulp.LpInteger),
        pulp.LpVariable("x_1", lowBound=0, cat=pulp.LpInteger)
    ]

    # Dummy objective (we just want to find ANY feasible solution)
    prob += 0, "Dummy_Objective"

    # Sparse A matrix (using dictionary representation)
    # Constraint 1: x0 + 2*x1 ≤ 10
    prob += x[0] + 2*x[1] <= 10, "constraint_0"

    # Constraint 2: 3*x0 + x1 ≤ 15
    prob += 3*x[0] + x[1] <= 15, "constraint_1"

    print("\nConstraint system A*x ≤ b:")
    print("  x[0] + 2*x[1] ≤ 10")
    print("  3*x[0] + x[1] ≤ 15")
    print("  x[0], x[1] ≥ 0 (integers)")

    # Solve
    status = prob.solve(pulp.PULP_CBC_CMD(msg=0))

    print(f"\nSolver Status: {pulp.LpStatus[status]}")

    if status == pulp.LpStatusOptimal:
        print("✓ FEASIBLE - Integer solution exists!")
        print("\nExample feasible solution:")
        for i, var in enumerate(x):
            print(f"  x[{i}] = {int(var.varValue)}")

        # Verify constraints
        print("\nConstraint verification:")
        lhs1 = x[0].varValue + 2*x[1].varValue
        print(f"  x[0] + 2*x[1] = {lhs1:.0f} ≤ 10? {lhs1 <= 10}")
        lhs2 = 3*x[0].varValue + x[1].varValue
        print(f"  3*x[0] + x[1] = {lhs2:.0f} ≤ 15? {lhs2 <= 15}")
    else:
        print("✗ INFEASIBLE - No integer solution exists!")


# ============================================================================
# Example 2: Infeasible system detection
# ============================================================================
def example2_infeasible_system():
    """
    Test an infeasible integer system.

    System:
        x0 + x1 ≥ 10
        x0 ≤ 3
        x1 ≤ 4
        x0, x1 ≥ 0 (integers)

    This is infeasible because x0 + x1 ≤ 3 + 4 = 7 < 10
    """
    print("\n" + "="*70)
    print("Example 2: Infeasible System Detection")
    print("="*70)

    prob = pulp.LpProblem("Infeasible_Test", pulp.LpMinimize)

    x = [
        pulp.LpVariable("x_0", lowBound=0, cat=pulp.LpInteger),
        pulp.LpVariable("x_1", lowBound=0, cat=pulp.LpInteger)
    ]

    # Dummy objective
    prob += 0

    # Conflicting constraints
    prob += x[0] + x[1] >= 10, "sum_constraint"
    prob += x[0] <= 3, "x0_upper"
    prob += x[1] <= 4, "x1_upper"

    print("\nConstraint system:")
    print("  x[0] + x[1] ≥ 10")
    print("  x[0] ≤ 3")
    print("  x[1] ≤ 4")
    print("  x[0], x[1] ≥ 0 (integers)")
    print("\nAnalysis: x[0] + x[1] ≤ 3 + 4 = 7 < 10 → INFEASIBLE")

    status = prob.solve(pulp.PULP_CBC_CMD(msg=0))

    print(f"\nSolver Status: {pulp.LpStatus[status]}")

    if status == pulp.LpStatusInfeasible:
        print("✓ Correctly detected as INFEASIBLE")
    elif status == pulp.LpStatusOptimal:
        print("✗ Solver found a solution (unexpected!)")
        for i, var in enumerate(x):
            print(f"  x[{i}] = {var.varValue}")
    else:
        print(f"Other status: {pulp.LpStatus[status]}")


# ============================================================================
# Example 3: Integer vs Continuous relaxation
# ============================================================================
def example3_integer_vs_continuous():
    """
    Compare feasibility for integer vs continuous variables.

    Some systems are feasible for continuous variables but infeasible for integers.

    System:
        2*x0 + 3*x1 = 7
        x0, x1 ≥ 0

    Continuous: feasible (e.g., x0=3.5, x1=0 or x0=2, x1=1)
    Integer: feasible (e.g., x0=2, x1=1)
    """
    print("\n" + "="*70)
    print("Example 3: Integer vs Continuous Feasibility")
    print("="*70)

    print("\nConstraint system:")
    print("  2*x[0] + 3*x[1] = 7")
    print("  x[0], x[1] ≥ 0")

    # Test with continuous variables
    print("\n--- Testing with CONTINUOUS variables ---")
    prob_cont = pulp.LpProblem("Continuous_Test", pulp.LpMinimize)
    x_cont = [
        pulp.LpVariable("x_0", lowBound=0, cat=pulp.LpContinuous),
        pulp.LpVariable("x_1", lowBound=0, cat=pulp.LpContinuous)
    ]
    prob_cont += 0
    prob_cont += 2*x_cont[0] + 3*x_cont[1] == 7

    status_cont = prob_cont.solve(pulp.PULP_CBC_CMD(msg=0))
    print(f"Status: {pulp.LpStatus[status_cont]}")
    if status_cont == pulp.LpStatusOptimal:
        print("✓ Feasible with continuous variables")
        print(f"  Solution: x[0]={x_cont[0].varValue:.2f}, x[1]={x_cont[1].varValue:.2f}")

    # Test with integer variables
    print("\n--- Testing with INTEGER variables ---")
    prob_int = pulp.LpProblem("Integer_Test", pulp.LpMinimize)
    x_int = [
        pulp.LpVariable("x_0", lowBound=0, cat=pulp.LpInteger),
        pulp.LpVariable("x_1", lowBound=0, cat=pulp.LpInteger)
    ]
    prob_int += 0
    prob_int += 2*x_int[0] + 3*x_int[1] == 7

    status_int = prob_int.solve(pulp.PULP_CBC_CMD(msg=0))
    print(f"Status: {pulp.LpStatus[status_int]}")
    if status_int == pulp.LpStatusOptimal:
        print("✓ Feasible with integer variables")
        print(f"  Solution: x[0]={int(x_int[0].varValue)}, x[1]={int(x_int[1].varValue)}")
        # Verify
        result = 2*x_int[0].varValue + 3*x_int[1].varValue
        print(f"  Verification: 2*{int(x_int[0].varValue)} + 3*{int(x_int[1].varValue)} = {result:.0f}")


# ============================================================================
# Example 4: Sparse matrix feasibility test (larger system)
# ============================================================================
def example4_sparse_feasibility():
    """
    Test feasibility for a larger sparse system using scipy sparse matrices.

    10 integer variables, 5 constraints
    """
    print("\n" + "="*70)
    print("Example 4: Sparse Matrix Feasibility Test (Larger System)")
    print("="*70)

    n_vars = 10
    n_constraints = 5

    prob = pulp.LpProblem("Sparse_Feasibility", pulp.LpMinimize)

    # Create integer variables
    x = [pulp.LpVariable(f"x_{i}", lowBound=0, upBound=20, cat=pulp.LpInteger)
         for i in range(n_vars)]

    # Dummy objective
    prob += 0

    # Build sparse A matrix
    A = lil_matrix((n_constraints, n_vars))

    # Constraint 0: x0 + x2 + x5 ≤ 25
    A[0, 0] = 1
    A[0, 2] = 1
    A[0, 5] = 1

    # Constraint 1: 2*x1 + x3 + x7 ≤ 30
    A[1, 1] = 2
    A[1, 3] = 1
    A[1, 7] = 1

    # Constraint 2: x4 + x6 + x8 + x9 ≤ 40
    A[2, 4] = 1
    A[2, 6] = 1
    A[2, 8] = 1
    A[2, 9] = 1

    # Constraint 3: x0 + x1 + x2 + x3 ≥ 10 (at least some production)
    A[3, 0] = 1
    A[3, 1] = 1
    A[3, 2] = 1
    A[3, 3] = 1

    # Constraint 4: x5 + x6 + x7 + x8 + x9 ≥ 15
    A[4, 5] = 1
    A[4, 6] = 1
    A[4, 7] = 1
    A[4, 8] = 1
    A[4, 9] = 1

    # Right-hand side
    b = [25, 30, 40, 10, 15]
    constraint_sense = ['<=', '<=', '<=', '>=', '>=']

    # Convert to COO and add constraints
    A_coo = A.tocoo()

    print(f"\nSparse constraint matrix:")
    print(f"  Shape: {n_constraints} constraints × {n_vars} variables")
    print(f"  Non-zero entries: {A_coo.nnz}")
    print(f"  Sparsity: {100 * (1 - A_coo.nnz / (n_constraints * n_vars)):.1f}% zero")

    # Group by row and build constraints
    constraint_dict = {}
    for row, col, val in zip(A_coo.row, A_coo.col, A_coo.data):
        if row not in constraint_dict:
            constraint_dict[row] = []
        constraint_dict[row].append((col, val))

    print("\nConstraints:")
    for row in sorted(constraint_dict.keys()):
        constraint_expr = pulp.lpSum([val * x[int(col)] for col, val in constraint_dict[row]])

        # Build string representation
        terms = " + ".join([f"{int(val)}*x[{int(col)}]" for col, val in constraint_dict[row]])
        sense = constraint_sense[row]
        print(f"  {terms} {sense} {b[row]}")

        if constraint_sense[row] == '<=':
            prob += constraint_expr <= b[row], f"constraint_{row}"
        else:
            prob += constraint_expr >= b[row], f"constraint_{row}"

    # Solve
    print("\nSolving...")
    status = prob.solve(pulp.PULP_CBC_CMD(msg=0))

    print(f"\nSolver Status: {pulp.LpStatus[status]}")

    if status == pulp.LpStatusOptimal:
        print("✓ FEASIBLE - Integer solution exists!")
        print("\nFeasible solution (showing non-zero variables):")
        for i in range(n_vars):
            if x[i].varValue > 0:
                print(f"  x[{i}] = {int(x[i].varValue)}")

        # Verify constraints
        print("\nConstraint verification:")
        for row in sorted(constraint_dict.keys()):
            lhs = sum([val * x[int(col)].varValue for col, val in constraint_dict[row]])
            sense = constraint_sense[row]
            rhs = b[row]

            if sense == '<=':
                satisfied = lhs <= rhs
            else:
                satisfied = lhs >= rhs

            check = "✓" if satisfied else "✗"
            print(f"  {check} Constraint {row}: {lhs:.0f} {sense} {rhs}")
    else:
        print("✗ INFEASIBLE - No integer solution exists!")


# ============================================================================
# Example 5: Programmatic feasibility checking function
# ============================================================================
def check_integer_feasibility(A_sparse, b, sense='<=', bounds=(0, None)):
    """
    Utility function to check if A*x {sense} b has an integer solution.

    Parameters:
    -----------
    A_sparse : list of tuples (row, col, value) in COO format
    b : list of right-hand side values
    sense : str or list, constraint sense ('<=', '>=', or '==')
    bounds : tuple (lower, upper) for variable bounds

    Returns:
    --------
    feasible : bool
    solution : list or None
    """
    n_constraints = max(row for row, _, _ in A_sparse) + 1 if A_sparse else 0
    n_vars = max(col for _, col, _ in A_sparse) + 1 if A_sparse else 0

    prob = pulp.LpProblem("Feasibility_Check", pulp.LpMinimize)

    # Create variables
    lower, upper = bounds
    x = [pulp.LpVariable(f"x_{i}", lowBound=lower, upBound=upper, cat=pulp.LpInteger)
         for i in range(n_vars)]

    # Dummy objective
    prob += 0

    # Build constraints
    constraint_dict = {}
    for row, col, val in A_sparse:
        if row not in constraint_dict:
            constraint_dict[row] = []
        constraint_dict[row].append((col, val))

    # Handle sense as string or list
    if isinstance(sense, str):
        sense_list = [sense] * n_constraints
    else:
        sense_list = sense

    for row in constraint_dict:
        expr = pulp.lpSum([val * x[col] for col, val in constraint_dict[row]])

        if sense_list[row] == '<=':
            prob += expr <= b[row]
        elif sense_list[row] == '>=':
            prob += expr >= b[row]
        elif sense_list[row] == '==':
            prob += expr == b[row]

    # Solve
    status = prob.solve(pulp.PULP_CBC_CMD(msg=0))

    if status == pulp.LpStatusOptimal:
        solution = [x[i].varValue for i in range(n_vars)]
        return True, solution
    else:
        return False, None


def example5_utility_function():
    """
    Demonstrate the utility function for quick feasibility checks.
    """
    print("\n" + "="*70)
    print("Example 5: Programmatic Feasibility Checking Utility")
    print("="*70)

    # Test case 1: Feasible system
    print("\n--- Test Case 1: Feasible System ---")
    A1 = [
        (0, 0, 1), (0, 1, 2),  # x0 + 2*x1 ≤ 10
        (1, 0, 3), (1, 1, 1),  # 3*x0 + x1 ≤ 15
    ]
    b1 = [10, 15]

    feasible, solution = check_integer_feasibility(A1, b1, sense='<=')
    print(f"Feasible: {feasible}")
    if feasible:
        print(f"Solution: x = {[int(v) for v in solution]}")

    # Test case 2: Infeasible system
    print("\n--- Test Case 2: Infeasible System ---")
    A2 = [
        (0, 0, 1), (0, 1, 1),  # x0 + x1 ≥ 10
        (1, 0, 1),             # x0 ≤ 3
        (2, 1, 1),             # x1 ≤ 4
    ]
    b2 = [10, 3, 4]
    sense2 = ['>=', '<=', '<=']

    feasible, solution = check_integer_feasibility(A2, b2, sense=sense2)
    print(f"Feasible: {feasible}")
    if not feasible:
        print("No integer solution exists (as expected)")

    # Test case 3: Equality constraint
    print("\n--- Test Case 3: Equality Constraint ---")
    A3 = [
        (0, 0, 2), (0, 1, 3),  # 2*x0 + 3*x1 = 7
    ]
    b3 = [7]

    feasible, solution = check_integer_feasibility(A3, b3, sense='==')
    print(f"Feasible: {feasible}")
    if feasible:
        print(f"Solution: x = {[int(v) for v in solution]}")
        print(f"Verification: 2*{int(solution[0])} + 3*{int(solution[1])} = {2*solution[0] + 3*solution[1]:.0f}")


# ============================================================================
# Run all examples
# ============================================================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("PuLP Integer Feasibility Testing Tutorial")
    print("="*70)

    example1_basic_feasibility()
    example2_infeasible_system()
    example3_integer_vs_continuous()
    example4_sparse_feasibility()
    example5_utility_function()

    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print("""
    Key Takeaways:

    1. Feasibility vs Optimization:
       - Feasibility: Does ANY solution exist?
       - Optimization: What's the BEST solution?
       - For feasibility, use a dummy objective (e.g., minimize 0)

    2. Detecting Infeasibility:
       - Solver returns LpStatusInfeasible when no solution exists
       - Common causes: conflicting constraints, over-constrained systems

    3. Integer vs Continuous:
       - Continuous relaxation may be feasible when integer problem isn't
       - Always specify cat=pulp.LpInteger for integer variables
       - Integer feasibility is NP-hard (harder than continuous)

    4. Sparse Matrices:
       - Store constraints efficiently with COO format
       - Use dictionaries or scipy.sparse for large systems
       - Sparsity doesn't affect feasibility, just memory/speed

    5. Practical Tips:
       - Start with continuous relaxation to debug constraints
       - If continuous is infeasible, integer is too
       - If continuous is feasible but integer isn't, system is "integrally infeasible"
       - Use bounds carefully (unbounded variables can cause issues)

    Applications:
    - Scheduling: Can we schedule all tasks with given resources?
    - Packing: Can we fit all items in containers?
    - Assignment: Can we assign all workers to tasks?
    - Sudoku/puzzles: Does a valid solution exist?
    """)