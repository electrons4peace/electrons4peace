import utils.common_utils as common_utils


def get_data(type_mode: str, date: str) -> list[str]:
    """Return the raw input lines for the given Advent of Code date.

    This is a thin, typed wrapper around common_utils.get_data_list.
    """
    data = common_utils.get_data_list(type_mode, date)
    return data




def get_roll_map_dict(data: list[str]) -> dict[tuple[int, int], str]:
    # Create a dictionary to hold the roll map of char "@". The data look like:
    #   ..@@.@@@@.
    #    @@@.@.@.@@  
    rows = len(data)
    cols = len(data[0])
    roll_map = {}
    for r in range(rows):
        for c in range(cols):
            if data[r][c] == "@":
                roll_map[(r, c)] = data[r][c]
    return roll_map

def is_neighbor(pos1: tuple[int, int], pos2: tuple[int, int]) -> bool:
    # Check if pos2 is a neighbor of pos1 (horizontally or vertically)
    r1, c1 = pos1
    r2, c2 = pos2
    return (abs(r1 - r2) == 1 and c1 == c2) or (r1 == r2 and abs(c1 - c2) == 1) 

def get_tuple_of_possible_neighbors(pos: tuple[int, int]) -> list[tuple[int, int]]:
    # Given a position, return a list of possible neighbor positions with diagonals
    r, c = pos
    neighbors = [
        (r - 1, c),     # Up
        (r + 1, c),     # Down
        (r, c - 1),     # Left
        (r, c + 1),     # Right
        (r - 1, c - 1), # Top-left
        (r - 1, c + 1), # Top-right
        (r + 1, c - 1), # Bottom-left
        (r + 1, c + 1)  # Bottom-right
    ]
    return neighbors

def get_positions_of_neighbors_with_less_than_n_neighbors(roll_map: dict[tuple[int, int], str], n: int) -> list[tuple[int, int]]:
    # For each position in roll_map, check how many neighbors it has in roll_map
    positions_with_few_neighbors = []
    for pos in roll_map.keys():
        possible_neighbors = get_tuple_of_possible_neighbors(pos)
        neighbor_count = sum(1 for neighbor in possible_neighbors if neighbor in roll_map)
        if neighbor_count < n:
            positions_with_few_neighbors.append(pos)
    return positions_with_few_neighbors

def runa(type_mode: str, date: str) -> None:
    """Entry point for running the solution for the given date.

    For now this just loads the input data and reports how many lines
    were loaded so you can plug in puzzle-specific logic later.
    """
    data = get_data(type_mode, date)
    print(f"Loaded {len(data)} lines for {date!r} with mode {type_mode!r}.")
    print(data)
    roll_map = get_roll_map_dict(data)
    print(f"Roll map positions with '@': {roll_map.keys()}")
    positions_with_few_neighbors = get_positions_of_neighbors_with_less_than_n_neighbors(roll_map, 4)
    print(f"Positions with less than 4 neighbors: {positions_with_few_neighbors}")
    print(f"Count of such positions: {len(positions_with_few_neighbors)}")

def runb(type_mode: str, date: str) -> None:
    data = get_data(type_mode, date)
    roll_map = get_roll_map_dict(data)
    count_removed = 0
    while True:
        positions_to_remove = get_positions_of_neighbors_with_less_than_n_neighbors(roll_map, 4)
        if not positions_to_remove:
            break
        #Remove the positions from the roll_map
        for pos in positions_to_remove:
            del roll_map[pos]
            count_removed += 1
    print(f"Final roll map positions with '@': {roll_map.keys()}")
    print(f"Count of removed positions: {count_removed}")


if __name__ == "__main__":
    date = "dec04"
    #type_mode = "test"
    type_mode = "data"
    #runa(type_mode, date)
    runb(type_mode, date)