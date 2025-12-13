import utils.common_utils as common_utils
import re
import pulp
import numpy as np
from pprint import pprint

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


def get_A_matrix(shapes_list: list[list[np.ndarray]], size: tuple[int,int], quantities: tuple[int]) -> np.ndarray:
    """Build the constraint matrix A for the linear programming problem.

    Each column represents one possible placement of one shape instance.
    Each row represents one cell in the grid.
    A[i,j] = 1 if placement j uses cell i, 0 otherwise.

    Args:
        shapes_list: List of lists, where shapes_list[i] contains all rotations of shape i
        size: (width, height) of the grid
        quantities: Tuple of how many of each shape type we need

    Returns:
        A matrix (grid_cells x total_placements)
    """
    width, height = size
    total_cells = width * height

    all_placements = []  # List of all possible placements across all shapes

    # For each shape type
    for shape_idx, shape_rotations in enumerate(shapes_list):
        quantity = quantities[shape_idx]

        # Skip shapes we don't need
        if quantity == 0:
            continue

        # For each instance we need of this shape
        for instance in range(quantity):
            # For each rotation of this shape
            for rotation in shape_rotations:
                # Get all valid placements of this rotation
                placements = get_all_placements(rotation, size)

                # Add each placement as a column in our matrix
                for placement in placements:
                    # Convert to vector indices
                    vector_indices = map_matrix_position_to_vector_position(placement, size)
                    all_placements.append(vector_indices)

    # Build the matrix
    num_columns = len(all_placements)
    A = np.zeros((total_cells, num_columns), dtype=int)

    for col_idx, placement in enumerate(all_placements):
        for cell_idx in placement:
            A[cell_idx, col_idx] = 1

    return A



def playground(data: list[str]):
    shapes = get_shapes_list(data)
    coords = get_coords_list(data)
    matrix = get_matrix_from_shape(shapes[0])
    print(matrix)
    rotations = get_all_shape_rotations(matrix)
    for rotation in rotations:
        pprint(rotation.tolist())
    


def runa(type_mode: str, date: str) -> None:
    """Entry point for running the solution for the given date.

    For now this just loads the input data and reports how many lines
    were loaded so you can plug in puzzle-specific logic later.
    """
    data = get_data(type_mode, date)
    shapes = get_shapes_list(data)
    coords, size_list = get_coords_list(data)
    print(f"Loaded {len(shapes)} shapes and {len(coords)} coordinate sets.")
    for i, shape in enumerate(shapes):
        print(f"Shape {i}:")
        for row in shape:
            print(row)
    for i, coord in enumerate(coords):
        print(f"Coord set {i}: {coord}, Size: {size_list[i]}")
    playground(data)
#    print(shapes)
#    print(coords)
#    print(size_list)




if __name__ == "__main__":
    date = "dec12"
    type_mode = "test"
    #type_mode = "data"
    runa(type_mode, date)
    #runb(type_mode, date)