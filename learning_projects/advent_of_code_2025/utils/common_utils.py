from pathlib import Path


def get_data_list(type: str, date_key: str) -> list:
    base_dir = Path(__file__).resolve().parents[1]
    input_path = base_dir / "data" / f"{date_key}_{type}.txt"
    input_file = open(input_path)
    data: list[str] = [x.strip() for x in input_file.readlines()]
    return data

def string_to_int_list(string_list):
#    int_list = []
#    for string in string_list:
#        int_list.append([int(num) for num in string.split(' ')])
#    return int_list
    return [int(num) for num in string_list.split()]

def read_input_file(type: str, date_key: str) -> str:
    base_dir = Path(__file__).resolve().parents[1]
    input_path = base_dir / "data" / f"{date_key}_{type}.txt"
    with open(input_path, 'r') as input_file:
        data = input_file.read()
    return data    

from collections import deque

def bfs_shortest_path(grid, start, goal):
    rows, cols = len(grid), len(grid[0])
    queue = deque([(start, [start])])  # (position, path)
    visited = {start}
    
    while queue:
        (r, c), path = queue.popleft()
        
        if (r, c) == goal:
            return path
        
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # 4-connected
            nr, nc = r + dr, c + dc
            if (0 <= nr < rows and 0 <= nc < cols 
                and (nr, nc) not in visited 
                and grid[nr][nc] != '#'):  # assuming '#' is wall
                visited.add((nr, nc))
                queue.append(((nr, nc), path + [(nr, nc)]))
    
    return None  # no path exists


def get_int_list(data: list) -> list:
    int_list = []
    for elem in data:
        int_list.append(string_to_int_list(elem))
    return int_list

class NumBracketSet:
    def __init__(self, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value

    @staticmethod
    def is_num_in_range(range_list: list, num_val: int) -> bool:
        if (num_val >= range_list[0]) & (num_val <= range_list[1]):
            return True
        else:
            return False

def get_map_dict(data) -> dict:
    map_dict = {}
    for idx_row, line in enumerate(data):
        for idx_col, ch in enumerate(line):
            map_dict[(idx_row, idx_col)] = ch
    return map_dict


def is_num_in_range(num, range_list: list) -> bool:
    if (num >= range_list[0]) & (num <= range_list[1]):
        return True
    else:
        return False


def get_intersection_set_pair(first_list: list, second_list: list) -> list:
    min_val = max(first_list[0], second_list[0])
    max_val = min(first_list[1], second_list[1])
    return_list = []
    if min_val <= max_val:
        return_list.append(min_val)
        return_list.append(max_val)
    return return_list


class Graph:
  def __init__(self):
    self.nodes = set()
    self.edges = defaultdict(list)
    self.distances = {}

  def add_node(self, value):
    self.nodes.add(value)

  def add_edge(self, from_node, to_node, distance):
    self.edges[from_node].append(to_node)
    self.edges[to_node].append(from_node)
    self.distances[(from_node, to_node)] = distance


