import utils.common_utils as common_utils
import re
import pulp
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix

"""
There are a few Elves here frantically decorating before the deadline. They think they'll be able to finish most of the work, but the one thing they're worried about is the presents for all the young Elves that live here at the North Pole. It's an ancient tradition to put the presents under the trees, but the Elves are worried they won't fit.

The presents come in a few standard but very weird shapes. The shapes and the regions into which they need to fit are all measured in standard units. To be aesthetically pleasing, the presents need to be placed into the regions in a way that follows a standardized two-dimensional unit grid; you also can't stack presents.

As always, the Elves have a summary of the situation (your puzzle input) for you. First, it contains a list of the presents' shapes. Second, it contains the size of the region under each tree and a list of the number of presents of each shape that need to fit into that region. For example:

0:
###
##.
##.

1:
###
##.
.##

2:
.##
###
##.

3:
##.
###
##.

4:
###
#..
###

5:
###
.#.
###

4x4: 0 0 0 0 2 0
12x5: 1 0 1 0 2 2
12x5: 1 0 1 0 3 2
The first section lists the standard present shapes. For convenience, each shape starts with its index and a colon; then, the shape is displayed visually, where # is part of the shape and . is not.

The second section lists the regions under the trees. Each line starts with the width and length of the region; 12x5 means the region is 12 units wide and 5 units long. The rest of the line describes the presents that need to fit into that region by listing the quantity of each shape of present; 1 0 1 0 3 2 means you need to fit one present with shape index 0, no presents with shape index 1, one present with shape index 2, no presents with shape index 3, three presents with shape index 4, and two presents with shape index 5.

Presents can be rotated and flipped as necessary to make them fit in the available space, but they have to always be placed perfectly on the grid. Shapes can't overlap (that is, the # part from two different presents can't go in the same place on the grid), but they can fit together (that is, the . part in a present's shape's diagram does not block another present from occupying that space on the grid).

The Elves need to know how many of the regions can fit the presents listed. In the above example, there are six unique present shapes and three regions that need checking.

The first region is 4x4:

....
....
....
....
In it, you need to determine whether you could fit two presents that have shape index 4:

###
#..
###
After some experimentation, it turns out that you can fit both presents in this region. Here is one way to do it, using A to represent one present and B to represent the other:

AAA.
ABAB
ABAB
.BBB
The second region, 12x5: 1 0 1 0 2 2, is 12 units wide and 5 units long. In that region, you need to try to fit one present with shape index 0, one present with shape index 2, two presents with shape index 4, and two presents with shape index 5.

It turns out that these presents can all fit in this region. Here is one way to do it, again using different capital letters to represent all the required presents:

....AAAFFE.E
.BBBAAFFFEEE
DDDBAAFFCECE
DBBB....CCC.
DDD.....C.C.
The third region, 12x5: 1 0 1 0 3 2, is the same size as the previous region; the only difference is that this region needs to fit one additional present with shape index 4. Unfortunately, no matter how hard you try, there is no way to fit all of the presents into this region.

So, in this example, 2 regions can fit all of their listed presents.

Consider the regions beneath each tree and the presents the Elves would like to fit into each of them. How many of the regions can fit all of the presents listed?



"""


def get_data(type_mode: str, date: str) -> list[str]:
    """Return the raw input lines for the given Advent of Code date.

    This is a thin, typed wrapper around common_utils.get_data_list.
    """
    data = common_utils.get_data_list(type_mode, date)
    return data


def get_shapes_list(data: list[str]) -> list[list]:
    shapes = []
    current_shape = []

    for line in data:
        #Check if x is in line, brake loop
        if re.search(r'x', line):
            break
        #check if int is in line
        if re.search(r'\d', line):
            current_shape = []
            continue
        #If the line is empty, store current shape and reset
        if line == '':
             if current_shape:
                shapes.append(current_shape)
        else:
            current_shape.append(line)
    return shapes

def get_coords_list(data: list[str]) -> list[tuple[int,int], list]:
    coords: list = []
    number_of_shapes_list = []
    for line in data:
        if re.search(r'x', line):
            #Get the two intergers before : and separate by the character x
            size_part = line.split(':')[0].strip()
            width, height = map(int, size_part.split('x'))
            number_of_shapes_list.append((width, height))
            # Get the integers separated by space to a list of tuples
            coord_parts = line.split(':')[1].strip().split()
            coord_tuple = tuple(map(int, coord_parts))
            coords.append(coord_tuple)
    return coords, number_of_shapes_list


def get_matrix_from_shape(shape: list[str]) -> np.ndarray:
    matrix = np.array([[1 if char == '#' else 0 for char in line] for line in shape])
    return matrix

def prune_matrix(matrix_list: list[np.ndarray]) -> list[np.ndarray]:
    # Remove identical matrices
    unique_matrices = []
    seen = set()
    
    for matrix in matrix_list:
        # Convert matrix to a hashable tuple for comparison
        matrix_tuple = tuple(map(tuple, matrix))
        if matrix_tuple not in seen:
            seen.add(matrix_tuple)
            unique_matrices.append(matrix)
    
    return unique_matrices


def get_all_shape_rotations(matrix:np.array) -> list[np.array]:
    # Rotate the shape 90 degrees clockwise
    # Additional shapes comes from flipping the rotated shapes
    flipped_matrix = np.fliplr(matrix)
    rotations = []
    rotations.append(matrix)
    rotations.append(flipped_matrix)
    rotations.append(np.rot90(matrix))
    rotations.append(np.rot90(flipped_matrix))
    rotations.append(np.rot90(np.rot90(matrix)))
    rotations.append(np.rot90(np.rot90(flipped_matrix)))
    rotations.append(np.rot90(np.rot90(np.rot90(matrix))))
    rotations.append(np.rot90(np.rot90(np.rot90(flipped_matrix))))
    unique_rotations = prune_matrix(rotations)
    return unique_rotations


def map_matrix_position_to_vector_position(pos_list: list[tuple[int,int]], size: tuple[int,int]) -> list[int]:
    """Convert 2D positions to 1D vector indices.

    Args:
        pos_list: List of (row, col) positions in the 2D grid
        size: (width, height) of the grid

    Returns:
        List of vector indices corresponding to the positions
    """
    width, height = size
    vector_indices = []
    for row, col in pos_list:
        # Convert (row, col) to single index: row * width + col
        index = row * width + col
        vector_indices.append(index)
    return vector_indices


def get_shape_positions(shape: np.ndarray) -> list[tuple[int,int]]:
    """Get all positions where the shape has a '#' (value 1).

    Args:
        shape: Binary matrix representing the shape

    Returns:
        List of (row, col) tuples where shape has value 1
    """
    positions = []
    rows, cols = shape.shape
    for i in range(rows):
        for j in range(cols):
            if shape[i, j] == 1:
                positions.append((i, j))
    return positions


def get_all_placements(shape: np.ndarray, grid_size: tuple[int,int]) -> list[list[tuple[int,int]]]:
    """Get all valid placements of a shape in a grid.

    Args:
        shape: Binary matrix representing the shape
        grid_size: (width, height) of the grid

    Returns:
        List of placements, where each placement is a list of (row, col) positions
    """
    width, height = grid_size
    shape_height, shape_width = shape.shape
    placements = []

    # Try all possible top-left corner positions
    for start_row in range(height - shape_height + 1):
        for start_col in range(width - shape_width + 1):
            # Get positions for this placement
            placement = []
            for i in range(shape_height):
                for j in range(shape_width):
                    if shape[i, j] == 1:
                        placement.append((start_row + i, start_col + j))
            placements.append(placement)

    return placements


def quick_feasibility_check(shapes_list: list[list[np.ndarray]], size: tuple[int,int], quantities: tuple[int]) -> bool:
    """Quick check if problem is obviously infeasible based on total area.

    Args:
        shapes_list: List of lists, where shapes_list[i] contains all rotations of shape i
        size: (width, height) of the grid
        quantities: Tuple of required quantities for each shape type

    Returns:
        True if problem might be feasible, False if obviously infeasible
    """
    width, height = size
    total_grid_area = width * height

    # Calculate total area needed by all shapes
    total_shapes_area = 0
    for shape_idx, shape_rotations in enumerate(shapes_list):
        quantity = quantities[shape_idx]
        if quantity == 0:
            continue

        # Use the first rotation to calculate the shape's area (all rotations have same area)
        if shape_rotations:
            shape = shape_rotations[0]
            shape_area = np.sum(shape)  # Count the number of 1s
            total_shapes_area += shape_area * quantity

    # If total shapes area exceeds grid area, it's infeasible
    if total_shapes_area > total_grid_area:
        return False

    return True


def get_A_matrix(shapes_list: list[list[np.ndarray]], size: tuple[int,int], quantities: tuple[int], use_sparse: bool = True):
    """Build the constraint matrix A for the linear programming problem.

    Each column represents one possible placement of one shape TYPE (not instance).
    Each row represents one cell in the grid.
    A[i,j] = 1 if placement j uses cell i, 0 otherwise.

    Args:
        shapes_list: List of lists, where shapes_list[i] contains all rotations of shape i
        size: (width, height) of the grid
        quantities: Tuple of how many of each shape type we need
        use_sparse: If True, return a scipy sparse matrix (much more memory efficient)

    Returns:
        A tuple of:
        - A matrix (grid_cells x total_placement_options) - sparse or dense
        - shape_indices: List mapping column index to shape type index
    """
    width, height = size
    total_cells = width * height

    all_placements = []  # List of all possible placements
    shape_indices = []   # Track which shape type each column corresponds to

    # For each shape type
    for shape_idx, shape_rotations in enumerate(shapes_list):
        quantity = quantities[shape_idx]

        # Skip shapes we don't need
        if quantity == 0:
            continue

        # For each rotation of this shape
        for rotation in shape_rotations:
            # Get all valid placements of this rotation
            placements = get_all_placements(rotation, size)

            # Add each placement as a column in our matrix
            for placement in placements:
                # Convert to vector indices
                vector_indices = map_matrix_position_to_vector_position(placement, size)
                all_placements.append(vector_indices)
                shape_indices.append(shape_idx)  # Track which shape this column represents

    # Build the matrix
    num_columns = len(all_placements)

    if use_sparse:
        # Use LIL (List of Lists) format for efficient construction
        A = lil_matrix((total_cells, num_columns), dtype=np.int8)

        for col_idx, placement in enumerate(all_placements):
            for cell_idx in placement:
                A[cell_idx, col_idx] = 1

        # Convert to CSR (Compressed Sparse Row) for efficient arithmetic operations
        A = csr_matrix(A)
    else:
        # Dense matrix (original implementation)
        A = np.zeros((total_cells, num_columns), dtype=int)

        for col_idx, placement in enumerate(all_placements):
            for cell_idx in placement:
                A[cell_idx, col_idx] = 1

    return A, shape_indices


def solve_placement_problem(A, shape_indices: list[int], quantities: tuple[int], size: tuple[int,int]) -> dict:
    """Solve the shape placement problem using linear programming.

    Args:
        A: Constraint matrix where A[i,j] = 1 if placement j uses cell i (can be sparse or dense)
        shape_indices: List mapping column index to shape type index
        quantities: Tuple of required quantities for each shape type
        size: (width, height) of the grid

    Returns:
        Dictionary with:
        - 'feasible': bool indicating if solution exists
        - 'solution': list of selected placement indices (if feasible)
        - 'grid': 2D array showing the solution (if feasible)
    """
    from scipy.sparse import issparse

    width, height = size
    num_placements = A.shape[1]
    num_cells = A.shape[0]

    # Create the LP problem
    prob = pulp.LpProblem("Shape_Placement", pulp.LpMinimize)

    # Create binary decision variables (x[i] = 1 if we use placement i)
    x = [pulp.LpVariable(f"x_{i}", cat='Binary') for i in range(num_placements)]

    # Objective: We just want feasibility, so use a dummy objective
    prob += 0

    # Constraint 1: Each cell can be occupied by at most one placement (A @ x <= 1)
    if issparse(A):
        # For sparse matrices, iterate over non-zero elements more efficiently
        A_csr = csr_matrix(A)  # Ensure CSR format for efficient row slicing
        for cell_idx in range(num_cells):
            # Get non-zero column indices for this row
            row = A_csr.getrow(cell_idx)
            nonzero_cols = row.nonzero()[1]
            if len(nonzero_cols) > 0:
                prob += pulp.lpSum(x[j] for j in nonzero_cols) <= 1, f"Cell_{cell_idx}"
    else:
        # Dense matrix version
        for cell_idx in range(num_cells):
            prob += pulp.lpSum(A[cell_idx, j] * x[j] for j in range(num_placements)) <= 1, f"Cell_{cell_idx}"

    # Constraint 2: Must use exactly the required quantity of each shape type
    num_shape_types = len(quantities)
    for shape_idx in range(num_shape_types):
        if quantities[shape_idx] == 0:
            continue

        # Find all placement columns for this shape type
        cols_for_shape = [j for j, s_idx in enumerate(shape_indices) if s_idx == shape_idx]

        if cols_for_shape:
            prob += pulp.lpSum(x[j] for j in cols_for_shape) == quantities[shape_idx], f"Shape_{shape_idx}_quantity"

    # Solve the problem
    prob.solve(pulp.PULP_CBC_CMD(msg=0))  # msg=0 suppresses solver output

    # Check if solution is feasible
    is_feasible = prob.status == pulp.LpStatusOptimal

    result = {
        'feasible': is_feasible,
        'solution': None,
        'grid': None
    }

    if is_feasible:
        # Extract which placements were selected
        selected_placements = [i for i in range(num_placements) if pulp.value(x[i]) == 1]
        result['solution'] = selected_placements

        # Build a visual grid showing the solution
        grid = np.full((height, width), -1, dtype=int)  # -1 means empty

        for placement_idx in selected_placements:
            shape_type = shape_indices[placement_idx]

            # Get the cells occupied by this placement
            if issparse(A):
                # For sparse matrix, get non-zero row indices
                col = A.getcol(placement_idx)
                cells = col.nonzero()[0]
            else:
                # Dense matrix version
                cells = [cell_idx for cell_idx in range(num_cells) if A[cell_idx, placement_idx] == 1]

            # Mark these cells with the shape type
            for cell_idx in cells:
                row = cell_idx // width
                col = cell_idx % width
                grid[row, col] = shape_type

        result['grid'] = grid

    return result


def visualize_solution(grid: np.ndarray) -> str:
    """Convert a solution grid to a visual string representation.

    Args:
        grid: 2D array where each cell contains shape type index or -1 for empty

    Returns:
        String representation using letters A-Z for shapes, . for empty
    """
    height, width = grid.shape
    lines = []

    for row in range(height):
        line = ""
        for col in range(width):
            value = grid[row, col]
            if value == -1:
                line += "."
            else:
                # Use letters A-Z for shape types
                line += chr(ord('A') + value)
        lines.append(line)

    return "\n".join(lines)


def solve_region(shapes_list: list[list[np.ndarray]], size: tuple[int,int], quantities: tuple[int], use_sparse: bool = True) -> bool:
    """Solve a single region placement problem.

    Args:
        shapes_list: List of lists, where shapes_list[i] contains all rotations of shape i
        size: (width, height) of the grid
        quantities: Tuple of required quantities for each shape type
        use_sparse: If True, use sparse matrices for better memory efficiency

    Returns:
        True if the region can fit all required shapes, False otherwise
    """
    from scipy.sparse import issparse

    print(f"\nSolving region {size[0]}x{size[1]} with {quantities}")

    # Quick feasibility check based on total area
    if not quick_feasibility_check(shapes_list, size, quantities):
        # Calculate actual areas for reporting
        width, height = size
        total_grid_area = width * height
        total_shapes_area = sum(
            np.sum(shape_rotations[0]) * quantities[shape_idx]
            for shape_idx, shape_rotations in enumerate(shapes_list)
            if quantities[shape_idx] > 0 and shape_rotations
        )
        print(f"✗ INFEASIBLE - Area check failed: need {total_shapes_area} cells, have {total_grid_area} cells")
        return False

    # Build the A matrix
    A, shape_indices = get_A_matrix(shapes_list, size, quantities, use_sparse=use_sparse)

    print(f"A matrix shape: {A.shape} (cells x placements)")

    # Show memory efficiency of sparse matrix
    if issparse(A):
        nnz = A.nnz
        total_elements = A.shape[0] * A.shape[1]
        sparsity = 100 * (1 - nnz / total_elements)
        print(f"Sparse matrix: {nnz:,} non-zeros out of {total_elements:,} ({sparsity:.1f}% sparse)")
        # Memory estimate
        dense_bytes = total_elements * 4  # 4 bytes per int32
        sparse_bytes = nnz * 12  # Roughly 12 bytes per non-zero (data + indices)
        print(f"Memory: ~{sparse_bytes/1024:.1f}KB (sparse) vs ~{dense_bytes/1024:.1f}KB (dense) - {100*sparse_bytes/dense_bytes:.1f}% of dense")
    else:
        print("Using dense matrix")

    # Solve the problem
    result = solve_placement_problem(A, shape_indices, quantities, size)

    if result['feasible']:
        print("✓ FEASIBLE - Solution found!")
        #print("\nSolution grid:")
        #print(visualize_solution(result['grid']))
        #print(f"\nUsed {len(result['solution'])} placements: {result['solution']}")
        return True
    else:
        print("✗ INFEASIBLE - No solution exists")
        return False


def playground(data: list[str]):
    """Test the solver with example data."""
    shapes = get_shapes_list(data)
    coords, size_list = get_coords_list(data)

    # Get all shape rotations
    all_shape_rotations = []
    for i, shape in enumerate(shapes):
        matrix = get_matrix_from_shape(shape)
        rotations = get_all_shape_rotations(matrix)
        all_shape_rotations.append(rotations)
        print(f"\nShape {i} has {len(rotations)} unique rotations")

    # Test each region
    print("\n" + "="*60)
    print("TESTING REGIONS")
    print("="*60)

    feasible_count = 0
    for i, (quantities, size) in enumerate(zip(coords, size_list)):
        print(f"\n{'='*60}")
        print(f"Region {i+1}")
        print(f"{'='*60}")

        is_feasible = solve_region(all_shape_rotations, size, quantities)
        if is_feasible:
            feasible_count += 1

    print(f"\n{'='*60}")
    print(f"SUMMARY: {feasible_count}/{len(coords)} regions are feasible")
    print(f"{'='*60}")
    


def runa(type_mode: str, date: str) -> None:
    """Solve the Advent of Code Day 12 problem."""
    data = get_data(type_mode, date)
    playground(data)




if __name__ == "__main__":
    date = "dec12"
    #type_mode = "test"
    type_mode = "data"
    runa(type_mode, date)
    #runb(type_mode, date)