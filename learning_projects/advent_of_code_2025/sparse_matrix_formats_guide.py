"""
Sparse Matrix Formats Guide: COO, CSR, CSC, LIL, and DOK
=========================================================

This guide explains different sparse matrix formats and when to use each one.

Sparse matrices are used when most elements are zero. Instead of storing all
elements (including zeros), we only store non-zero values, saving memory.

Common formats:
- COO (COOrdinate): List of (row, col, value) tuples
- CSR (Compressed Sparse Row): Efficient for row operations
- CSC (Compressed Sparse Column): Efficient for column operations
- LIL (List of Lists): Efficient for incremental construction
- DOK (Dictionary of Keys): Efficient for random access
"""

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix, lil_matrix, dok_matrix
import time


def print_matrix_comparison(dense, sparse_format_name, sparse):
    """Helper to print matrix in both dense and sparse formats"""
    print(f"\nDense representation:")
    print(dense)
    print(f"\n{sparse_format_name} representation:")
    print(sparse)
    print(f"Non-zero elements: {sparse.nnz}")
    print(f"Sparsity: {100 * (1 - sparse.nnz / (sparse.shape[0] * sparse.shape[1])):.1f}% zero")


# ============================================================================
# Example 1: COO (COOrdinate) Format
# ============================================================================
def example1_coo_format():
    """
    COO Format: List of (row, col, value) triplets

    Best for:
    - Building sparse matrices from data
    - Converting between formats
    - File I/O (easy to save/load)

    Not good for:
    - Arithmetic operations
    - Slicing
    - Random access

    Storage:
    - Three arrays: row indices, column indices, values
    - Memory: O(nnz) where nnz = number of non-zero elements
    """
    print("\n" + "="*70)
    print("Example 1: COO (COOrdinate) Format")
    print("="*70)

    # Create a sparse matrix using COO format
    # Matrix:
    #   [1  0  0  5]
    #   [0  2  0  0]
    #   [0  0  3  0]
    #   [6  0  0  4]

    rows = [0, 0, 1, 2, 3, 3]  # Row indices of non-zero elements
    cols = [0, 3, 1, 2, 0, 3]  # Column indices of non-zero elements
    data = [1, 5, 2, 3, 6, 4]  # Values of non-zero elements

    # Create COO matrix
    A_coo = coo_matrix((data, (rows, cols)), shape=(4, 4))

    print("\nCOO representation stores three arrays:")
    print(f"  rows: {rows}")
    print(f"  cols: {cols}")
    print(f"  data: {data}")

    print("\nEach triplet (row[i], col[i], data[i]) represents one non-zero element:")
    for i in range(len(data)):
        print(f"  A[{rows[i]}, {cols[i]}] = {data[i]}")

    dense = A_coo.toarray()
    print_matrix_comparison(dense, "COO", A_coo)

    # Alternative: Create from list of tuples (common in optimization)
    print("\n" + "-"*70)
    print("Alternative: List of (row, col, value) tuples")
    print("-"*70)

    triplets = [
        (0, 0, 1),
        (0, 3, 5),
        (1, 1, 2),
        (2, 2, 3),
        (3, 0, 6),
        (3, 3, 4),
    ]

    print("\nTriplet representation:")
    for row, col, val in triplets:
        print(f"  ({row}, {col}, {val})")

    # Convert to COO
    rows, cols, data = zip(*triplets)
    A_coo2 = coo_matrix((data, (rows, cols)), shape=(4, 4))

    print("\nThis is the MOST COMMON format for optimization problems!")
    print("Why? Because constraints naturally come as triplets:")
    print("  'In constraint 2, variable 5 has coefficient 3' → (2, 5, 3)")


# ============================================================================
# Example 2: CSR (Compressed Sparse Row) Format
# ============================================================================
def example2_csr_format():
    """
    CSR Format: Compressed Sparse Row

    Best for:
    - Matrix-vector multiplication (A @ x)
    - Row slicing
    - Fast arithmetic operations
    - Default format for most computations

    Not good for:
    - Incremental construction (slow)
    - Column slicing

    Storage:
    - Three arrays: data, indices, indptr
    - More complex but more efficient than COO
    """
    print("\n" + "="*70)
    print("Example 2: CSR (Compressed Sparse Row) Format")
    print("="*70)

    # Same matrix as before
    triplets = [(0, 0, 1), (0, 3, 5), (1, 1, 2), (2, 2, 3), (3, 0, 6), (3, 3, 4)]
    rows, cols, data = zip(*triplets)

    # Create CSR matrix
    A_csr = csr_matrix((data, (rows, cols)), shape=(4, 4))

    print("\nCSR stores data row-by-row:")
    print(f"  data:    {A_csr.data}")      # Non-zero values
    print(f"  indices: {A_csr.indices}")   # Column indices
    print(f"  indptr:  {A_csr.indptr}")    # Row pointers

    print("\nHow to read indptr:")
    print("  indptr[i] to indptr[i+1] gives the range in 'data' for row i")
    for i in range(4):
        start = A_csr.indptr[i]
        end = A_csr.indptr[i + 1]
        print(f"  Row {i}: data[{start}:{end}] = {A_csr.data[start:end]}")
        print(f"         at columns: {A_csr.indices[start:end]}")

    dense = A_csr.toarray()
    print_matrix_comparison(dense, "CSR", A_csr)

    # Performance comparison
    print("\n" + "-"*70)
    print("Performance: Matrix-vector multiplication")
    print("-"*70)

    # Large sparse matrix
    n = 1000
    nnz = 5000
    rows = np.random.randint(0, n, nnz)
    cols = np.random.randint(0, n, nnz)
    data = np.random.rand(nnz)

    A_coo_large = coo_matrix((data, (rows, cols)), shape=(n, n))
    A_csr_large = A_coo_large.tocsr()
    x = np.random.rand(n)

    # Time COO matrix-vector multiplication
    start = time.time()
    for _ in range(100):
        y = A_coo_large @ x
    time_coo = time.time() - start

    # Time CSR matrix-vector multiplication
    start = time.time()
    for _ in range(100):
        y = A_csr_large @ x
    time_csr = time.time() - start

    print(f"\nMatrix size: {n}x{n}, non-zeros: {nnz}")
    print(f"COO: {time_coo*1000:.2f} ms")
    print(f"CSR: {time_csr*1000:.2f} ms")
    print(f"CSR is {time_coo/time_csr:.1f}x faster for matrix-vector multiplication!")


# ============================================================================
# Example 3: LIL (List of Lists) Format
# ============================================================================
def example3_lil_format():
    """
    LIL Format: List of Lists

    Best for:
    - Incremental construction (adding elements one by one)
    - Changing sparsity structure
    - Building matrices element by element

    Not good for:
    - Arithmetic operations (convert to CSR first)
    - Large matrices (memory overhead)

    Storage:
    - Two lists: one for column indices, one for data
    - Each row is a separate list
    """
    print("\n" + "="*70)
    print("Example 3: LIL (List of Lists) Format")
    print("="*70)

    # Create empty LIL matrix
    A_lil = lil_matrix((4, 4))

    print("\nBuilding matrix incrementally with LIL:")

    # Add elements one by one (efficient with LIL)
    elements = [(0, 0, 1), (0, 3, 5), (1, 1, 2), (2, 2, 3), (3, 0, 6), (3, 3, 4)]

    for row, col, val in elements:
        A_lil[row, col] = val
        print(f"  Set A[{row}, {col}] = {val}")

    print("\nLIL internal structure (list of lists):")
    print(f"  rows (list of lists): {A_lil.rows.tolist()}")
    print(f"  data (list of lists): {A_lil.data.tolist()}")

    print("\nInterpretation:")
    for i in range(4):
        print(f"  Row {i}: columns {A_lil.rows[i]} have values {A_lil.data[i]}")

    dense = A_lil.toarray()
    print_matrix_comparison(dense, "LIL", A_lil)

    # Performance comparison for construction
    print("\n" + "-"*70)
    print("Performance: Incremental construction")
    print("-"*70)

    n = 1000
    nnz = 5000
    elements = [(np.random.randint(0, n), np.random.randint(0, n), np.random.rand())
                for _ in range(nnz)]

    # Build with LIL (efficient)
    start = time.time()
    A_lil_test = lil_matrix((n, n))
    for row, col, val in elements:
        A_lil_test[row, col] = val
    time_lil = time.time() - start

    # Build with CSR (inefficient - need to convert each time)
    start = time.time()
    A_csr_test = csr_matrix((n, n))
    for row, col, val in elements:
        A_csr_test[row, col] = val
    time_csr = time.time() - start

    print(f"\nAdding {nnz} elements to {n}x{n} matrix:")
    print(f"LIL: {time_lil*1000:.2f} ms")
    print(f"CSR: {time_csr*1000:.2f} ms")
    print(f"LIL is {time_csr/time_lil:.1f}x faster for incremental construction!")

    print("\n💡 Best practice: Build with LIL, convert to CSR for computation")


# ============================================================================
# Example 4: DOK (Dictionary of Keys) Format
# ============================================================================
def example4_dok_format():
    """
    DOK Format: Dictionary of Keys

    Best for:
    - Random access (getting/setting individual elements)
    - Changing sparsity structure
    - When you need to check if element exists

    Not good for:
    - Arithmetic operations (convert to CSR first)
    - Iteration over elements

    Storage:
    - Python dictionary: {(row, col): value}
    - Very natural for Python programmers
    """
    print("\n" + "="*70)
    print("Example 4: DOK (Dictionary of Keys) Format")
    print("="*70)

    # Create DOK matrix
    A_dok = dok_matrix((4, 4))

    print("\nDOK stores as Python dictionary:")

    elements = [(0, 0, 1), (0, 3, 5), (1, 1, 2), (2, 2, 3), (3, 0, 6), (3, 3, 4)]

    for row, col, val in elements:
        A_dok[row, col] = val

    print(f"\nInternal dictionary: {dict(A_dok)}")

    print("\nYou can check if element exists:")
    print(f"  (0, 0) in A_dok? {(0, 0) in A_dok}")
    print(f"  (0, 1) in A_dok? {(0, 1) in A_dok}")

    print("\nRandom access is fast:")
    print(f"  A_dok[0, 3] = {A_dok[0, 3]}")
    print(f"  A_dok[1, 1] = {A_dok[1, 1]}")

    dense = A_dok.toarray()
    print_matrix_comparison(dense, "DOK", A_dok)


# ============================================================================
# Example 5: Practical guide - When to use which format?
# ============================================================================
def example5_practical_guide():
    """
    Practical decision tree for choosing sparse format
    """
    print("\n" + "="*70)
    print("Example 5: Practical Guide - Which Format to Use?")
    print("="*70)

    guide = """
    DECISION TREE:

    1. Are you BUILDING a sparse matrix?

       a) From a list of (row, col, value) triplets?
          → Use COO format
          Example: Reading constraints from a file
          Code: coo_matrix((data, (rows, cols)), shape=(m, n))

       b) Adding elements one by one, row by row?
          → Use LIL format, then convert to CSR
          Example: Building constraint matrix in a loop
          Code: A = lil_matrix((m, n))
                A[i, j] = value
                A = A.tocsr()  # Convert when done building

       c) Need random access to set/check elements?
          → Use DOK format, then convert to CSR
          Example: Sparse adjacency matrix from graph
          Code: A = dok_matrix((m, n))
                A[i, j] = value
                A = A.tocsr()  # Convert when done building

    2. Are you USING a sparse matrix?

       a) Matrix-vector multiplication (A @ x)?
          → Use CSR format
          Fastest for row-based operations

       b) Transposed operations (A.T @ x)?
          → Use CSC format
          Fastest for column-based operations

       c) Need to modify matrix structure?
          → Use LIL or DOK, then convert back to CSR

       d) Just storing/transferring data?
          → Use COO format
          Easiest to serialize/deserialize

    3. For OPTIMIZATION problems (PuLP, CVXPY, etc.):

       → BUILD with COO format (triplet list)
       → Solver converts internally as needed

       Why? Constraints are naturally triplets:
         "Constraint i uses variable j with coefficient a_ij"
         → (i, j, a_ij)

    QUICK REFERENCE TABLE:
    ┌─────────────────────────┬─────┬─────┬─────┬─────┬─────┐
    │ Operation               │ COO │ CSR │ CSC │ LIL │ DOK │
    ├─────────────────────────┼─────┼─────┼─────┼─────┼─────┤
    │ Build from triplets     │ ✓✓✓ │  ✓  │  ✓  │  ✓  │  ✓  │
    │ Incremental construction│  ✗  │  ✗  │  ✗  │ ✓✓✓ │ ✓✓  │
    │ Random access (get/set) │  ✗  │  ✗  │  ✗  │  ✓  │ ✓✓✓ │
    │ Matrix @ vector         │  ✓  │ ✓✓✓ │  ✓  │  ✗  │  ✗  │
    │ vector @ Matrix         │  ✓  │  ✓  │ ✓✓✓ │  ✗  │  ✗  │
    │ Arithmetic ops          │  ✓  │ ✓✓✓ │ ✓✓✓ │  ✗  │  ✗  │
    │ Slicing rows            │  ✗  │ ✓✓✓ │  ✓  │  ✓  │  ✓  │
    │ Slicing columns         │  ✗  │  ✓  │ ✓✓✓ │  ✓  │  ✓  │
    │ Memory efficiency       │ ✓✓  │ ✓✓✓ │ ✓✓✓ │  ✓  │  ✓  │
    │ Convert to other formats│ ✓✓✓ │ ✓✓✓ │ ✓✓✓ │ ✓✓  │ ✓✓  │
    └─────────────────────────┴─────┴─────┴─────┴─────┴─────┘

    ✓✓✓ = Excellent,  ✓✓ = Good,  ✓ = OK,  ✗ = Poor

    COMMON PATTERNS:

    Pattern 1: Build then compute
      A = lil_matrix((m, n))      # Build
      # ... add elements ...
      A = A.tocsr()               # Convert
      y = A @ x                   # Compute

    Pattern 2: Load from file (triplets)
      data = load_triplets()      # [(row, col, val), ...]
      rows, cols, vals = zip(*data)
      A = coo_matrix((vals, (rows, cols)))
      A = A.tocsr()               # Convert for computation

    Pattern 3: Optimization constraints
      triplets = [
          (0, 0, 1.0),  # Constraint 0, variable 0, coeff 1.0
          (0, 2, 3.0),  # Constraint 0, variable 2, coeff 3.0
          # ...
      ]
      # Use directly in PuLP (as shown in tutorial)
    """

    print(guide)


# ============================================================================
# Example 6: Conversion between formats
# ============================================================================
def example6_conversions():
    """
    How to convert between different sparse formats
    """
    print("\n" + "="*70)
    print("Example 6: Converting Between Formats")
    print("="*70)

    # Create a matrix in COO format
    triplets = [(0, 0, 1), (0, 3, 5), (1, 1, 2), (2, 2, 3), (3, 0, 6), (3, 3, 4)]
    rows, cols, data = zip(*triplets)
    A_coo = coo_matrix((data, (rows, cols)), shape=(4, 4))

    print("\nStarting with COO format:")
    print(A_coo)

    # Convert to different formats
    print("\nConversions:")

    A_csr = A_coo.tocsr()
    print(f"\n  COO → CSR: A_coo.tocsr()")
    print(f"    Type: {type(A_csr)}")

    A_csc = A_coo.tocsc()
    print(f"\n  COO → CSC: A_coo.tocsc()")
    print(f"    Type: {type(A_csc)}")

    A_lil = A_coo.tolil()
    print(f"\n  COO → LIL: A_coo.tolil()")
    print(f"    Type: {type(A_lil)}")

    A_dok = A_coo.todok()
    print(f"\n  COO → DOK: A_coo.todok()")
    print(f"    Type: {type(A_dok)}")

    A_dense = A_coo.toarray()
    print(f"\n  COO → Dense: A_coo.toarray()")
    print(f"    Type: {type(A_dense)}")
    print(A_dense)

    print("\n💡 All formats support .tocoo(), .tocsr(), .tocsc(), .tolil(), .todok()")
    print("   Conversions are cheap - use the right format for each operation!")


# ============================================================================
# Run all examples
# ============================================================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("Sparse Matrix Formats Guide")
    print("="*70)

    example1_coo_format()
    example2_csr_format()
    example3_lil_format()
    example4_dok_format()
    example5_practical_guide()
    example6_conversions()

    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print("""
    KEY POINTS:

    1. COO (Coordinate): List of (row, col, value) triplets
       - Most intuitive format
       - Best for construction from data
       - Use for optimization constraints

    2. CSR (Compressed Sparse Row): Efficient for computation
       - Best for matrix-vector multiplication
       - Best for row slicing
       - Default format for computations

    3. LIL (List of Lists): Best for incremental construction
       - Fast element-by-element construction
       - Convert to CSR when done building

    4. DOK (Dictionary of Keys): Best for random access
       - Natural Python dictionary
       - Fast membership testing
       - Convert to CSR for computation

    5. CSC (Compressed Sparse Column): Like CSR but for columns
       - Best for transposed operations
       - Best for column slicing

    GOLDEN RULE:
    → Build with LIL or COO
    → Compute with CSR
    → Convert is cheap, so use the right format for each task!

    FOR PULP/OPTIMIZATION:
    → Use COO format (list of triplets)
    → Natural way to represent constraints
    → Each triplet = (constraint_id, variable_id, coefficient)
    """)
