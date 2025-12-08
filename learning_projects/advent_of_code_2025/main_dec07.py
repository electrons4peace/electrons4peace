import utils.common_utils as common_utils


def get_data(type_mode: str, date: str) -> list[str]:
    """Return the raw input lines for the given Advent of Code date.

    This is a thin, typed wrapper around common_utils.get_data_list.
    """
    data = common_utils.get_data_list(type_mode, date)
    return data

class BeamSplitter:
    def __init__(self, data: list[str]):
        self.data = data
        self.num_rows = len(data)
        self.num_cols = len(data[0])
        self.start_pos = self.get_start_pos()
    def get_start_pos(self) -> int:
        for idx, ch in enumerate(self.data[0]):
            if ch == 'S':
                return idx
        return -1

    def get_splitting_indices_of_row_idx(self, row_idx: int) -> list[int]:
        line = self.data[row_idx]
        splitting_indices = []
        for idx, ch in enumerate(line):
            if ch == '^':
                splitting_indices.append(idx)
        return splitting_indices
    def get_split_positions(self, pos: int) -> list[int]:
        new_positions = [pos - 1, pos + 1]
        for new_pos in new_positions:
            if (new_pos < 0) or (new_pos >= self.num_cols):
                new_positions.remove(new_pos)
        return new_positions

    def get_new_beam_indices_after_split(self, current_beam_indices: list[int], splitting_indices: list[int]) -> list[int]:
        # For each current beam index, produce new beam indices for each splitting index in the adjecent positions.
        new_beam_indices = []
        for beam_index in current_beam_indices:
            for split_index in splitting_indices:
                if split_index == beam_index:
                    split_positions = self.get_split_positions(beam_index)
                    new_beam_indices.extend(split_positions)
        return new_beam_indices




def get_after_split_beam(data: list[str]) -> list[str]:
    # Find the splitter char '^' and produce '|.|' below it.
    beam_splitter = BeamSplitter(data)
    current_beam_indices = [beam_splitter.start_pos]
    for row_idx in range(1, beam_splitter.num_rows):
        splitting_indices = beam_splitter.get_splitting_indices_of_row_idx(row_idx)
        current_beam_indices = beam_splitter.get_new_beam_indices_after_split(current_beam_indices, splitting_indices)
        # Create new row with '.' and '|' at current_beam_indices
        new_row = ['.'] * beam_splitter.num_cols
        for beam_index in current_beam_indices:
            new_row[beam_index] = '|'
        data[row_idx] = ''.join(new_row)

from collections import deque
from typing import List, Set, Tuple
# BFS approach to simulate the beam flow through the splitter grid. Each position in the grid can be represented as (row, col).
def get_start_position(data: List[str]) -> Tuple[int, int]:
    for row_idx, line in enumerate(data):
        for col_idx, ch in enumerate(line):
            if ch == 'S':
                return (row_idx, col_idx)
    return (-1, -1)


def get_energized_positions(data: List[str]) -> Set[Tuple[int, int]]:
    start_pos = get_start_position(data)
    num_rows = len(data)
    num_cols = len(data[0])
    energized_positions = set([start_pos])
    queue = deque([start_pos])
    number_of_splittings = 0
    # Keep track of positions that have already been processed: track where beams have already started falling from
    # to avoid re-calculating the same vertical path.
    processed_sources = set([start_pos])

    while queue:
        r, c = queue.popleft()
        # Calculate position directly below
        next_r, next_c = r + 1, c
        
        # Boundary check: if beam falls off the bottom, it stops
        if next_r >= num_rows:
            continue
        
        cell = data[next_r][next_c]
        # The tile below is energized regardless of what it is
        energized_positions.add((next_r, next_c))
        
        if cell == '.' or cell == 'S':
            # Beam passes freely: Continue downward from this new spot
            if (next_r, next_c) not in processed_sources:
                processed_sources.add((next_r, next_c))
                queue.append((next_r, next_c))
                
        elif cell == '^':
            # Splitter: Beam stops here. 
            # New beams form at immediate left and right (same row).
            number_of_splittings += 1            
            # Left Branch
            left_pos = (next_r, c - 1)
            if 0 <= left_pos[1] < num_cols:
                energized_positions.add(left_pos) # The start of the new beam is energized
                if left_pos not in processed_sources:
                    processed_sources.add(left_pos)
                    queue.append(left_pos)

            # Right Branch
            right_pos = (next_r, c + 1)
            if 0 <= right_pos[1] < num_cols:
                energized_positions.add(right_pos) # The start of the new beam is energized
                if right_pos not in processed_sources:
                    processed_sources.add(right_pos)
                    queue.append(right_pos)


# 3. Visualization
    # Create a copy of the grid to draw the path
    vis_grid = [row[:] for row in data]
    for r, c in energized_positions:
        # Don't overwrite the Splitters or Start for clarity, 
        # but you can remove this condition to see raw coverage.
        if vis_grid[r][c] not in ('^', 'S'):
            vis_grid[r] = vis_grid[r][:c] + '|' + vis_grid[r][c+1:]
            
    vis_output = '\n'.join(''.join(row) for row in vis_grid)
    
    return len(energized_positions), vis_output, number_of_splittings

def solve_quantum_manifold_iterative(data: List[str]) -> int:
    start_pos = get_start_position(data)
    rows = len(data)
    cols = len(data[0])
    # DP Table: dp[r][c] stores the number of successful timelines starting at (r, c).
    dp = [[0] * cols for _ in range(rows)] 

    # 2. Iterate Bottom-Up
    # We start calculating from the last row (r = rows - 1) and move upwards to the top row (r = 0).
    for r in range(rows - 1, -1, -1):
        for c in range(cols):
            
            # --- BASE CASE: Falling Off the Bottom ---
            # If the particle is on the last row and attempts to move down, 
            # it successfully exits (1 timeline).
            if r == rows - 1:
                 dp[r][c] = 1
                 continue

            # --- Movement and Split Logic ---
            next_r = r + 1
            cell_below = data[next_r][c]
            
            if cell_below == '.' or cell_below == 'S':
                # Path continues: The number of timelines is the same as the cell below it.
                dp[r][c] = dp[next_r][c]
                
            elif cell_below == '^':
                # Path splits: Sum the path counts from the left/right split locations.
                
                left_c = c - 1
                right_c = c + 1
                
                left_count = 0
                if 0 <= left_c < cols:
                    # Look up the pre-calculated count in the row below
                    left_count = dp[next_r][left_c]
                    
                right_count = 0
                if 0 <= right_c < cols:
                    # Look up the pre-calculated count in the row below
                    right_count = dp[next_r][right_c]
                    
                dp[r][c] = left_count + right_count

            # Note: The actual content of grid[r][c] is irrelevant, as the action 
            # always depends on the cell below (grid[r+1][c]).

    # 3. Final Result
    # The answer is the path count starting from the 'S' position.
    return dp[start_pos[0]][start_pos[1]]


def runa(type_mode: str, date: str) -> None:
    
    """Entry point for running the solution for the given date.

    For now this just loads the input data and reports how many lines
    were loaded so you can plug in puzzle-specific logic later.
    """
    data = get_data(type_mode, date)
    print(f"Loaded {len(data)} lines for {date} ({type_mode} mode).")
    print(data)
    num_energized, vis_output, number_of_splittings = get_energized_positions(data)
    print(f"Number of energized positions: {num_energized}")
    print(f"Visualization of energized positions:\n{vis_output}")
    print(f"Number of splittings: {number_of_splittings}")

def runb(type_mode: str, date: str) -> None:
    data = get_data(type_mode, date)
    num_timelines = solve_quantum_manifold_iterative(data)
    print(f"Number of successful timelines: {num_timelines}")


if __name__ == "__main__":
    date = "dec07"
    #type_mode = "test"
    type_mode = "data"
    #runa(type_mode, date)
    runb(type_mode, date)