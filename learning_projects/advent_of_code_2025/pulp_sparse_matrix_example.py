"""
PuLP Sparse Matrix Examples with Mixed Integer/Continuous Variables
====================================================================

This file demonstrates how to create sparse A matrices for use in PuLP constraints
with a mixed vector of integer and continuous variables.

A sparse matrix is efficient when most elements are zero - we only store non-zero values.

Common use cases:
- Network flow problems (integer flows + continuous costs)
- Assignment problems (binary decisions + continuous quantities)
- Production planning (integer units + continuous resources)
- Facility location (binary locations + continuous capacities)
"""

import pulp
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix, coo_matrix


# ============================================================================
# Example 1: Basic sparse matrix using dictionary of coefficients
#            with mixed integer and continuous variables
# ============================================================================
def example1_dict_based():
    """
    Most natural way in PuLP: use dictionaries to represent sparse coefficients.
    This example uses a mix of integer and continuous variables.

    Problem: Production planning
    - x[0], x[1]: Integer variables (number of products to make)
    - x[2], x[3], x[4]: Continuous variables (resource levels)
    """
    print("\n" + "="*70)
    print("Example 1: Dictionary-based sparse constraint matrix")
    print("         with mixed integer/continuous variables")
    print("="*70)

    # Create a simple problem
    prob = pulp.LpProblem("Sparse_Example_1", pulp.LpMinimize)

    # Define mixed variable types
    # First 2 are integers (e.g., discrete production quantities)
    x0 = pulp.LpVariable("x_0", lowBound=0, cat=pulp.LpInteger)
    x1 = pulp.LpVariable("x_1", lowBound=0, cat=pulp.LpInteger)

    # Last 3 are continuous (e.g., resource levels, percentages)
    x2 = pulp.LpVariable("x_2", lowBound=0, cat=pulp.LpContinuous)
    x3 = pulp.LpVariable("x_3", lowBound=0, cat=pulp.LpContinuous)
    x4 = pulp.LpVariable("x_4", lowBound=0, cat=pulp.LpContinuous)

    # Store in a vector for easy indexing
    x = [x0, x1, x2, x3, x4]

    # Define objective (minimize: 5*x0 + 3*x1 + cost of resources)
    prob += 5*x[0] + 3*x[1] + x[2] + x[3] + x[4]

    print("\nVariable types:")
    for i, var in enumerate(x):
        var_type = "Integer" if var.cat == pulp.LpInteger else "Continuous"
        print(f"  x[{i}] ({var.name}): {var_type}")

    # Create sparse constraints using dictionaries
    # Sparse A matrix representation

    # Constraint 1: x0 + 3*x2 >= 10  (integer product uses continuous resource)
    A_row1 = {0: 1, 2: 3}
    prob += pulp.lpSum([A_row1[i] * x[i] for i in A_row1]) >= 10, "constraint_1"

    # Constraint 2: 2*x1 + 4*x3 + x4 >= 15  (mix of integer and continuous)
    A_row2 = {1: 2, 3: 4, 4: 1}
    prob += pulp.lpSum([A_row2[i] * x[i] for i in A_row2]) >= 15, "constraint_2"

    # Constraint 3: x0 + x4 >= 5  (integer + continuous)
    A_row3 = {0: 1, 4: 1}
    prob += pulp.lpSum([A_row3[i] * x[i] for i in A_row3]) >= 5, "constraint_3"

    # Constraint 4: x1 + 2*x2 + x3 <= 20  (capacity constraint)
    A_row4 = {1: 1, 2: 2, 3: 1}
    prob += pulp.lpSum([A_row4[i] * x[i] for i in A_row4]) <= 20, "constraint_4"

    # Represent the full sparse A matrix
    A_sparse = [A_row1, A_row2, A_row3, A_row4]

    print("\nSparse A matrix (as dictionaries):")
    for i, row in enumerate(A_sparse):
        print(f"  Row {i}: {row}")

    print("\nProblem formulation:")
    print(prob)

    # Solve
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    print(f"\nStatus: {pulp.LpStatus[prob.status]}")
    print(f"Objective value: {pulp.value(prob.objective)}")
    print("Solution:")
    for i, var in enumerate(x):
        var_type = "INT" if var.cat == pulp.LpInteger else "CONT"
        print(f"  x[{i}] ({var_type}): {var.varValue}")


# ============================================================================
# Example 2: Sparse matrix using list of tuples (COO format concept)
#            with mixed integer and continuous variables
# ============================================================================
def example2_tuple_based():
    """
    Using list of (row, col, value) tuples - similar to COO sparse matrix format.
    Good when you're building constraints programmatically.

    Problem: Warehouse distribution
    - x[0-4]: Integer variables (number of shipments)
    - x[5-9]: Continuous variables (fractional capacities/percentages)
    """
    print("\n" + "="*70)
    print("Example 2: Tuple-based sparse matrix (COO-like)")
    print("         with mixed integer/continuous variables")
    print("="*70)

    prob = pulp.LpProblem("Sparse_Example_2", pulp.LpMaximize)

    # 10 variables: first 5 integer, last 5 continuous
    n_vars = 10
    x = []

    # Integer variables (e.g., discrete shipments)
    for i in range(5):
        x.append(pulp.LpVariable(f"x_{i}", lowBound=0, upBound=10, cat=pulp.LpInteger))

    # Continuous variables (e.g., capacity utilization)
    for i in range(5, 10):
        x.append(pulp.LpVariable(f"x_{i}", lowBound=0, upBound=1, cat=pulp.LpContinuous))

    print("\nVariable types:")
    for i, var in enumerate(x):
        var_type = "Integer" if var.cat == pulp.LpInteger else "Continuous"
        bounds = f"[{var.lowBound}, {var.upBound}]"
        print(f"  x[{i}] ({var.name}): {var_type:12s} bounds: {bounds}")

    # Objective: maximize (integer shipments + weighted continuous capacities)
    prob += pulp.lpSum([x[i] for i in range(5)]) + 2 * pulp.lpSum([x[i] for i in range(5, 10)])

    # Define sparse matrix as list of (row, col, value) tuples
    # This represents 4 constraints over 10 variables (mix of int and continuous)
    sparse_A = [
        # Constraint 0: x0 + 2*x3 + x7 <= 8 (integers + continuous)
        (0, 0, 1),
        (0, 3, 2),
        (0, 7, 1),

        # Constraint 1: 3*x1 + x5 + 4*x9 <= 6 (integers + continuous)
        (1, 1, 3),
        (1, 5, 1),
        (1, 9, 4),

        # Constraint 2: x2 + x4 + x6 + x8 <= 5 (all mixed)
        (2, 2, 1),
        (2, 4, 1),
        (2, 6, 1),
        (2, 8, 1),

        # Constraint 3: x0 + x1 + 2*x5 + 3*x6 >= 4 (integer production needs continuous support)
        (3, 0, 1),
        (3, 1, 1),
        (3, 5, 2),
        (3, 6, 3),
    ]

    # Right-hand side values
    b = [8, 6, 5, 4]
    constraint_sense = ['<=', '<=', '<=', '>=']

    print(f"\nSparse A matrix (COO format): {len(sparse_A)} non-zero entries")
    print("Entries (row, col, value):")
    for entry in sparse_A:
        row, col, val = entry
        var_type = "INT" if x[col].cat == pulp.LpInteger else "CONT"
        print(f"  A[{row},{col:2d}] = {val:2.0f}  (x[{col}] is {var_type})")

    # Build constraints from sparse representation
    # Group by row
    constraints = {}
    for row, col, val in sparse_A:
        if row not in constraints:
            constraints[row] = []
        constraints[row].append((col, val))

    # Add constraints to problem
    for row in sorted(constraints.keys()):
        constraint_expr = pulp.lpSum([val * x[col] for col, val in constraints[row]])
        if constraint_sense[row] == '<=':
            prob += constraint_expr <= b[row], f"constraint_{row}"
        else:
            prob += constraint_expr >= b[row], f"constraint_{row}"

    print("\nProblem formulation:")
    print(prob)

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    print(f"\nStatus: {pulp.LpStatus[prob.status]}")
    print(f"Objective value: {pulp.value(prob.objective):.4f}")
    print("\nSolution:")
    for i in range(n_vars):
        var_type = "INT" if x[i].cat == pulp.LpInteger else "CONT"
        if x[i].varValue is not None and x[i].varValue > 0.0001:
            print(f"  x[{i}] ({var_type}): {x[i].varValue:.4f}")
        else:
            print(f"  x[{i}] ({var_type}): 0.0000")


# ============================================================================
# Example 3: Using scipy sparse matrices with mixed integer/continuous
# ============================================================================
def example3_scipy_sparse():
    """
    Using scipy.sparse matrices - useful when interfacing with numpy/scipy code.

    Problem: Resource allocation with discrete and continuous decisions
    - x[0-3]: Integer variables (discrete units)
    - x[4-7]: Continuous variables (fractional allocations)
    """
    print("\n" + "="*70)
    print("Example 3: Using scipy sparse matrices")
    print("         with mixed integer/continuous variables")
    print("="*70)

    prob = pulp.LpProblem("Sparse_Example_3", pulp.LpMinimize)

    n_vars = 8
    n_constraints = 4

    # Create mixed variable vector
    x = []

    # First 4 are integer variables
    for i in range(4):
        x.append(pulp.LpVariable(f"x_{i}", lowBound=0, cat=pulp.LpInteger))

    # Last 4 are continuous variables
    for i in range(4, 8):
        x.append(pulp.LpVariable(f"x_{i}", lowBound=0, cat=pulp.LpContinuous))

    print("\nVariable vector x (mixed integer/continuous):")
    for i, var in enumerate(x):
        var_type = "Integer" if var.cat == pulp.LpInteger else "Continuous"
        print(f"  x[{i}] ({var.name}): {var_type}")

    # Objective: minimize weighted sum (penalize integers more)
    prob += pulp.lpSum([2*x[i] for i in range(4)]) + pulp.lpSum([x[i] for i in range(4, 8)])

    # Create sparse constraint matrix using scipy
    # lil_matrix is efficient for incremental construction
    A = lil_matrix((n_constraints, n_vars))

    # Fill in the sparse matrix
    # Constraint 0: x0 + 2*x1 + 0.5*x4 >= 5 (integers + continuous)
    A[0, 0] = 1
    A[0, 1] = 2
    A[0, 4] = 0.5

    # Constraint 1: 3*x2 + x5 + 2*x6 >= 8 (integer + continuous)
    A[1, 2] = 3
    A[1, 5] = 1
    A[1, 6] = 2

    # Constraint 2: x3 + x4 + x6 >= 6 (mixed)
    A[2, 3] = 1
    A[2, 4] = 1
    A[2, 6] = 1

    # Constraint 3: 2*x0 + x1 + 3*x7 >= 4 (integers + continuous)
    A[3, 0] = 2
    A[3, 1] = 1
    A[3, 7] = 3

    # Right-hand side
    b = np.array([5, 8, 6, 4])

    # Convert to COO format for easy iteration
    A_coo = A.tocoo()

    print(f"\nSparse matrix A shape: {A_coo.shape}")
    print(f"Number of non-zero elements: {A_coo.nnz}")
    print(f"Sparsity: {100 * (1 - A_coo.nnz / (n_constraints * n_vars)):.1f}% zero")

    print("\nNon-zero entries (row, col, value) with variable types:")
    for row, col, val in zip(A_coo.row, A_coo.col, A_coo.data):
        var_type = "INT" if x[col].cat == pulp.LpInteger else "CONT"
        print(f"  A[{row}, {col}] = {val:4.1f}  (x[{col}] is {var_type})")

    # Build constraints from sparse matrix
    constraint_dict = {}
    for row, col, val in zip(A_coo.row, A_coo.col, A_coo.data):
        if row not in constraint_dict:
            constraint_dict[row] = []
        constraint_dict[row].append((col, val))

    for row in sorted(constraint_dict.keys()):
        constraint_expr = pulp.lpSum([val * x[col] for col, val in constraint_dict[row]])
        prob += constraint_expr >= b[row], f"constraint_{row}"

    print("\nProblem formulation:")
    print(prob)

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    print(f"\nStatus: {pulp.LpStatus[prob.status]}")
    print(f"Objective value: {pulp.value(prob.objective):.2f}")
    print("\nSolution:")
    for i in range(n_vars):
        var_type = "INT" if x[i].cat == pulp.LpInteger else "CONT"
        print(f"  x[{i}] ({var_type:4s}): {x[i].varValue:.2f}")


# ============================================================================
# Example 4: Practical application - Production planning with mixed variables
# ============================================================================
def example4_production_planning():
    """
    Real-world example: Production planning with discrete products and continuous resources.

    Problem setup:
    - Binary variables: which products to manufacture (0/1 decisions)
    - Integer variables: quantity of each product
    - Continuous variables: resource allocation percentages and utilization

    Sparse A matrix because:
    - Not all products use all resources
    - Setup decisions only affect specific products
    """
    print("\n" + "="*70)
    print("Example 4: Production Planning with Mixed Variable Types")
    print("="*70)

    products = ["ProductA", "ProductB", "ProductC"]
    resources = ["Labor", "Material", "Energy"]

    # Create problem
    prob = pulp.LpProblem("Production_Planning", pulp.LpMaximize)

    # === DECISION VARIABLES ===

    # Binary variables: setup decision for each product (1 if we produce it, 0 otherwise)
    setup = {}
    for p in products:
        setup[p] = pulp.LpVariable(f"setup_{p}", cat=pulp.LpBinary)

    # Integer variables: quantity to produce of each product
    quantity = {}
    for p in products:
        quantity[p] = pulp.LpVariable(f"qty_{p}", lowBound=0, upBound=100, cat=pulp.LpInteger)

    # Continuous variables: resource utilization percentages
    resource_util = {}
    for r in resources:
        resource_util[r] = pulp.LpVariable(f"util_{r}", lowBound=0, upBound=1, cat=pulp.LpContinuous)

    print("\nDecision Variables:")
    print("  Binary setup decisions:")
    for p in products:
        print(f"    {setup[p].name}: 1 if we produce {p}, 0 otherwise")
    print("  Integer quantity decisions:")
    for p in products:
        print(f"    {quantity[p].name}: how many units of {p} to produce")
    print("  Continuous resource utilization:")
    for r in resources:
        print(f"    {resource_util[r].name}: fraction of {r} capacity used")

    # === OBJECTIVE: Maximize profit ===
    # Profit per unit for each product
    profit = {"ProductA": 50, "ProductB": 70, "ProductC": 60}
    # Fixed setup cost for each product
    setup_cost = {"ProductA": 100, "ProductB": 150, "ProductC": 120}

    prob += (
        pulp.lpSum([profit[p] * quantity[p] for p in products]) -
        pulp.lpSum([setup_cost[p] * setup[p] for p in products])
    ), "Total_Profit"

    # === SPARSE CONSTRAINT MATRIX ===
    # Resource consumption matrix (sparse - not all products use all resources equally)
    # Format: (resource, product, consumption_per_unit)
    resource_consumption = [
        # Labor
        ("Labor", "ProductA", 2.0),
        ("Labor", "ProductC", 1.5),
        # Material (ProductB doesn't use material - sparse!)
        ("Material", "ProductA", 3.0),
        ("Material", "ProductC", 2.0),
        # Energy
        ("Energy", "ProductA", 1.0),
        ("Energy", "ProductB", 2.5),
        ("Energy", "ProductC", 1.8),
    ]

    # Resource capacities (available capacity)
    capacity = {"Labor": 200, "Material": 250, "Energy": 180}

    print("\n=== SPARSE A MATRIX ===")
    print("Resource consumption (resource, product, units_per_item):")
    for entry in resource_consumption:
        print(f"  {entry}")
    print("\nNote: ProductB doesn't use Material (sparse entry)")
    print("      ProductB doesn't use Labor (sparse entry)")

    # Build resource constraints from sparse matrix
    # A * x <= b  where A is sparse
    for resource in resources:
        # Get only non-zero entries for this resource
        relevant_products = [
            (prod, coeff) for res, prod, coeff in resource_consumption
            if res == resource
        ]

        if relevant_products:
            # Resource consumption <= capacity * utilization
            resource_usage = pulp.lpSum([
                coeff * quantity[prod] for prod, coeff in relevant_products
            ])
            prob += resource_usage <= capacity[resource] * resource_util[resource], \
                    f"resource_{resource}_capacity"

    # Logical constraints: can only produce if setup
    # If setup[p] = 0, then quantity[p] must be 0
    # If setup[p] = 1, then quantity[p] can be up to 100
    for p in products:
        prob += quantity[p] <= 100 * setup[p], f"setup_logic_{p}"

    # Minimum production quantity if we decide to produce
    min_production = {"ProductA": 10, "ProductB": 15, "ProductC": 12}
    for p in products:
        prob += quantity[p] >= min_production[p] * setup[p], f"min_production_{p}"

    # Total utilization constraint (can't use more than total available resources)
    prob += pulp.lpSum([resource_util[r] for r in resources]) <= 2.5, "total_utilization"

    print("\n=== PROBLEM FORMULATION ===")
    print(prob)

    # Solve
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    print(f"\n{'='*70}")
    print(f"Status: {pulp.LpStatus[prob.status]}")
    print(f"Total Profit: ${pulp.value(prob.objective):.2f}")

    print("\n=== PRODUCTION PLAN ===")
    for p in products:
        if setup[p].varValue > 0.5:
            print(f"  {p}: PRODUCE {int(quantity[p].varValue)} units (setup cost: ${setup_cost[p]})")
        else:
            print(f"  {p}: DO NOT PRODUCE")

    print("\n=== RESOURCE UTILIZATION ===")
    for r in resources:
        util_pct = resource_util[r].varValue * 100
        actual_used = sum([
            coeff * quantity[prod].varValue
            for res, prod, coeff in resource_consumption
            if res == r
        ])
        print(f"  {r}: {util_pct:.1f}% utilization ({actual_used:.1f} / {capacity[r]} units)")

    print(f"\n{'='*70}")


# ============================================================================
# Run all examples
# ============================================================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("PuLP Sparse Matrix Tutorial")
    print("="*70)

    example1_dict_based()
    example2_tuple_based()
    example3_scipy_sparse()
    example4_production_planning()

    print("\n" + "="*70)
    print("Key Takeaways:")
    print("="*70)
    print("""
    1. Dictionary approach (Example 1): Best for manual constraint building
       - Natural Python syntax
       - Easy to understand and debug
       - Mixed integer/continuous variables work seamlessly

    2. Tuple list approach (Example 2): Best for programmatic generation
       - Similar to COO sparse format
       - Good when reading from data files
       - Easy to specify which variables are integer vs continuous

    3. Scipy sparse matrices (Example 3): Best for numerical integration
       - Efficient memory usage for large problems
       - Integrates with numpy/scipy ecosystem
       - Can handle mixed variable types in the x vector

    4. Real applications (Example 4): Production planning with mixed variables
       - Binary setup decisions (0/1)
       - Integer production quantities
       - Continuous resource utilization
       - Natural sparsity: not all products use all resources

    Key Points for Mixed Integer/Continuous Variables:
    - Define variable types using cat parameter:
      * cat=pulp.LpBinary (0 or 1)
      * cat=pulp.LpInteger (whole numbers)
      * cat=pulp.LpContinuous (default, fractional values)
    - Sparse A matrix works the same regardless of variable types
    - The solver (CBC) automatically handles mixed-integer programming (MIP)

    Choose based on your use case:
    - Small problems or prototyping → dictionaries
    - Data-driven models → tuple lists or scipy
    - Large-scale problems → scipy sparse matrices
    - Mixed decisions (discrete + continuous) → any approach works!
    """)
