import utils.common_utils as common_utils


def get_data(type_mode: str, date: str) -> list[str]:
    """Return the raw input lines for the given Advent of Code date.

    This is a thin, typed wrapper around common_utils.get_data_list.
    """
    data = common_utils.get_data_list(type_mode, date)
    return data

def get_list_of_positions(data: list[str]) -> list[tuple[int, int]]:
    positions = []
    for line in data:
        parts = line.split(',')
        pos = (int(parts[0]), int(parts[1]))
        positions.append(pos)
    return positions

def get_spanned_area(pos_0: tuple[int, int], pos_1: tuple[int, int]) -> int:
    #Calculate the area of the rectangle spanned by pos_0 and pos_1 (inclusive).
    width = abs(pos_1[0] - pos_0[0]) + 1
    height = abs(pos_1[1] - pos_0[1]) + 1
    return width * height

def get_all_neighboring_positions_of_same_row_or_column(pos: tuple[int, int], positions: list[tuple[int, int]]) -> list[tuple[int, int]]:
    neighbors = []
    for other_pos in positions:
        if other_pos == pos:
            continue
        if other_pos[0] == pos[0] or other_pos[1] == pos[1]:
            neighbors.append(other_pos)
    return neighbors

def prune_graph_to_connected_components(position_graph: dict[tuple[int, int], list[tuple[int, int]]]) -> dict[tuple[int, int], list[tuple[int, int]]]:
    # Only keep positions that are connected to at least two other position.
    pruned_graph = {}
    for pos, neighbors in position_graph.items():
        if len(neighbors) >= 2:
            pruned_graph[pos] = neighbors
    return pruned_graph

def get_closest_posistions_in_graph(position_graph: dict[tuple[int, int], list[tuple[int, int]]]) -> dict[tuple[int, int], tuple[int, int]]:
    # For each position in the graph, keep the closest neighboring position for the same row or column.
    closest_map = {}

def make_position_graph(positions: list[tuple[int, int]]) -> dict[tuple[int, int], list[tuple[int, int]]]:
    position_graph = {}
    for pos in positions:
        neighbors = get_all_neighboring_positions_of_same_row_or_column(pos, positions)
        position_graph[pos] = neighbors
    return position_graph

def get_al_possible_spanned_areas(positions: list[tuple[int, int]]) -> list[int]:
    areas = []
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            area = get_spanned_area(positions[i], positions[j])
            areas.append(area)
    return areas

def build_red_tile_graph(red_tiles):
    """
    Build connectivity graph of red tiles.
    Returns dict mapping each red tile to its neighbors (prev, next in list).
    The list wraps around, so first connects to last.
    """
    n = len(red_tiles)
    graph = {}
    
    for i, tile in enumerate(red_tiles):
        prev_tile = red_tiles[(i - 1) % n]
        next_tile = red_tiles[(i + 1) % n]
        graph[tile] = (prev_tile, next_tile)
    
    return graph


def trace_edge(p1, p2):
    """
    Trace the green tiles between two adjacent red tiles.
    Returns list of coordinates (excluding p1 and p2 themselves).
    """
    x1, y1 = p1
    x2, y2 = p2
    
    edge_tiles = []
    
    if x1 == x2:  # Vertical line
        step = 1 if y2 > y1 else -1
        for y in range(y1 + step, y2, step):
            edge_tiles.append((x1, y))
    elif y1 == y2:  # Horizontal line
        step = 1 if x2 > x1 else -1
        for x in range(x1 + step, x2, step):
            edge_tiles.append((x, y1))
    else:
        raise ValueError(f"Tiles {p1} and {p2} are not aligned horizontally or vertically")
    
    return edge_tiles


def get_polygon_boundary(red_tiles):
    """
    Get all boundary tiles (red + green edges between them).
    Returns set of coordinates.
    """
    boundary = set(red_tiles)
    
    n = len(red_tiles)
    for i in range(n):
        p1 = red_tiles[i]
        p2 = red_tiles[(i + 1) % n]
        boundary.update(trace_edge(p1, p2))
    
    return boundary


def get_interior_tiles(boundary):
    """
    Get all interior tiles using flood-fill from outside.
    
    Strategy:
    1. Find bounding box, expand by 1 to guarantee exterior starting point
    2. Flood fill all exterior tiles (boundary blocks the fill)
    3. Interior = everything in bounding box that's not exterior and not boundary
    """
    # Find bounding box, expanded by 1
    xs = [x for x, y in boundary]
    ys = [y for x, y in boundary]
    min_x, max_x = min(xs) - 1, max(xs) + 1
    min_y, max_y = min(ys) - 1, max(ys) + 1
    
    # Flood fill exterior starting from corner (guaranteed outside)
    exterior = set()
    stack = [(min_x, min_y)]
    
    while stack:
        x, y = stack.pop()
        
        if (x, y) in exterior or (x, y) in boundary:
            continue
        if x < min_x or x > max_x or y < min_y or y > max_y:
            continue
        
        exterior.add((x, y))
        
        # Add 4-connected neighbors
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            stack.append((x + dx, y + dy))
    
    # Interior = bounding box minus exterior minus boundary
    interior = set()
    for x in range(min_x + 1, max_x):
        for y in range(min_y + 1, max_y):
            if (x, y) not in exterior and (x, y) not in boundary:
                interior.add((x, y))
    
    return interior


def get_all_valid_tiles(red_tiles):
    """
    Get all valid (red + green) tiles = boundary + interior.
    """
    boundary = get_polygon_boundary(red_tiles)
    interior = get_interior_tiles(boundary)
    return boundary | interior


def visualize_polygon(red_tiles, valid_tiles=None):
    """Debug visualization matching the problem's format."""
    boundary = get_polygon_boundary(red_tiles)
    if valid_tiles is None:
        valid_tiles = get_all_valid_tiles(red_tiles)
    
    red_set = set(red_tiles)
    green_tiles = valid_tiles - red_set
    
    xs = [x for x, y in valid_tiles]
    ys = [y for x, y in valid_tiles]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    # Note: problem uses (x, y) where y is row (top to bottom)
    for y in range(min_y, max_y + 1):
        row = ""
        for x in range(min_x, max_x + 1):
            if (x, y) in red_set:
                row += "#"
            elif (x, y) in green_tiles:
                row += "X"
            else:
                row += "."
        print(row)



def runa(type_mode: str, date: str) -> None:
    """Entry point for running the solution for the given date.

    For now this just loads the input data and reports how many lines
    were loaded so you can plug in puzzle-specific logic later.
    """
    data = get_data(type_mode, date)
    positions = get_list_of_positions(data)
    print(f"Loaded {len(positions)} positions.")
    areas = get_al_possible_spanned_areas(positions)
    sorted_areas = sorted(areas)
    # Print the largest area.
    print(f"Largest spanned area: {sorted_areas[-1]}")

###############
def get_interior_tiles(boundary):
    """
    Get all interior tiles using flood-fill from outside.
    
    Strategy:
    1. Find bounding box, expand by 1 to guarantee exterior starting point
    2. Flood fill all exterior tiles (boundary blocks the fill)
    3. Interior = everything in bounding box that's not exterior and not boundary
    """
    # Find bounding box, expanded by 1
    xs = [x for x, y in boundary]
    ys = [y for x, y in boundary]
    min_x, max_x = min(xs) - 1, max(xs) + 1
    min_y, max_y = min(ys) - 1, max(ys) + 1
    
    # Flood fill exterior starting from corner (guaranteed outside)
    exterior = set()
    stack = [(min_x, min_y)]
    
    while stack:
        x, y = stack.pop()
        
        if (x, y) in exterior or (x, y) in boundary:
            continue
        if x < min_x or x > max_x or y < min_y or y > max_y:
            continue
        
        exterior.add((x, y))
        
        # Add 4-connected neighbors
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            stack.append((x + dx, y + dy))
    
    # Interior = bounding box minus exterior minus boundary
    interior = set()
    for x in range(min_x + 1, max_x):
        for y in range(min_y + 1, max_y):
            if (x, y) not in exterior and (x, y) not in boundary:
                interior.add((x, y))
    
    return interior


def get_all_valid_tiles(red_tiles):
    """
    Get all valid (red + green) tiles = boundary + interior.
    """
    boundary = get_polygon_boundary(red_tiles)
    interior = get_interior_tiles(boundary)
    return boundary | interior

def build_row_intervals(valid_tiles):
    """
    For each row, build sorted list of valid x-intervals.
    This lets us quickly check if a horizontal span is fully valid.
    """
    from collections import defaultdict
    
    rows = defaultdict(set)
    for x, y in valid_tiles:
        rows[y].add(x)
    
    # Convert each row's x-coordinates to merged intervals
    row_intervals = {}
    for y, x_coords in rows.items():
        x_sorted = sorted(x_coords)
        intervals = []
        start = end = x_sorted[0]
        
        for x in x_sorted[1:]:
            if x == end + 1:
                end = x
            else:
                intervals.append((start, end))
                start = end = x
        intervals.append((start, end))
        
        row_intervals[y] = intervals
    
    return row_intervals
def rectangle_area(p1, p2):
    """Calculate area of rectangle with opposite corners p1 and p2."""
    x1, y1 = p1
    x2, y2 = p2
    return (abs(x2 - x1) + 1) * (abs(y2 - y1) + 1)



def x_span_valid_in_row(min_x, max_x, intervals):
    """Check if [min_x, max_x] is fully contained in one interval."""
    for start, end in intervals:
        if start <= min_x and max_x <= end:
            return True
        if start > max_x:  # Intervals are sorted, no need to continue
            break
    return False


def is_valid_rectangle_fast(p1, p2, row_intervals, valid_bbox):
    """
    Fast rectangle validation using AABB and row intervals.
    """
    x1, y1 = p1
    x2, y2 = p2
    
    min_x, max_x = min(x1, x2), max(x1, x2)
    min_y, max_y = min(y1, y2), max(y1, y2)
    
    # AABB quick reject: check against valid region bounding box
    bbox_min_x, bbox_max_x, bbox_min_y, bbox_max_y = valid_bbox
    if min_x < bbox_min_x or max_x > bbox_max_x:
        return False
    if min_y < bbox_min_y or max_y > bbox_max_y:
        return False
    
    # Check each row's x-span is fully within a valid interval
    for y in range(min_y, max_y + 1):
        if y not in row_intervals:
            return False
        if not x_span_valid_in_row(min_x, max_x, row_intervals[y]):
            return False
    
    return True


def find_largest_valid_rectangle_fast(red_tiles, valid_tiles):
    """
    Optimized rectangle search using AABB techniques.
    """
    # Precompute spatial structures
    row_intervals = build_row_intervals(valid_tiles)
    
    xs = [x for x, y in valid_tiles]
    ys = [y for x, y in valid_tiles]
    valid_bbox = (min(xs), max(xs), min(ys), max(ys))
    
    best_area = 0
    best_corners = None
    
    n = len(red_tiles)
    
    for i in range(n):
        for j in range(i + 1, n):
            p1 = red_tiles[i]
            p2 = red_tiles[j]
            
            area = rectangle_area(p1, p2)
            
            if area <= best_area:
                continue
            
            if is_valid_rectangle_fast(p1, p2, row_intervals, valid_bbox):
                best_area = area
                best_corners = (p1, p2)
    
    return best_area, best_corners



def runb(type_mode: str, date: str) -> None:
    data = get_data(type_mode, date)
    red_tiles = get_list_of_positions(data)
    valid_tiles = get_all_valid_tiles(red_tiles)
    area, corners = find_largest_valid_rectangle_fast(red_tiles, valid_tiles)
    print(f"Largest valid rectangle area: {area} between corners {corners}")


if __name__ == "__main__":
    date = "dec09"
    #type_mode = "test"
    type_mode = "data"
    runa(type_mode, date)
    #runb(type_mode, date)