import utils.common_utils as common_utils
from functools import cache


def get_data(type_mode: str, date: str) -> list[str]:
    """Return the raw input lines for the given Advent of Code date.

    This is a thin, typed wrapper around common_utils.get_data_list.
    """
    data = common_utils.get_data_list(type_mode, date)
    return data


def split_string(input_string):
    return [x.strip().split() for x in input_string.split(':') if x.strip()]


def make_graph_dict(data: list[str]) -> dict[str, list[str]]:
    graph_dict = {}
    for line in data:
        parts = split_string(line)
        node = parts[0][0]
        edges = parts[1] if len(parts) > 1 else []
        graph_dict[node] = edges
    return graph_dict
def can_visit(neighbor, visited):
    return neighbor not in visited

def count_paths(graph, current, end, visited):
    if current == end:

        return 1
    total = 0
    for neighbor in graph[current]:
        if can_visit(neighbor, visited):  # your constraint logic
            visited.add(neighbor)
            total += count_paths(graph, neighbor, end, visited)
            visited.remove(neighbor)  # backtrack
    return total

def count_paths_part2(graph, current, end, visited):
    if current == end:
        if ('fft' in visited) and ('dac' in visited):
            return 1
        else:
            return 0
    total = 0
    for neighbor in graph[current]:
        if can_visit(neighbor, visited):  # your constraint logic
            visited.add(neighbor)
            total += count_paths_part2(graph, neighbor, end, visited)
            visited.remove(neighbor)  # backtrack
    return total

def has_cycle(graph):
    """Check if directed graph has cycles using DFS."""
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {node: WHITE for node in graph}
    
    def dfs(node):
        color[node] = GRAY
        for neighbor in graph.get(node, []):
            if color.get(neighbor, WHITE) == GRAY:  # back edge = cycle
                return True
            if color.get(neighbor, WHITE) == WHITE and dfs(neighbor):
                return True
        color[node] = BLACK
        return False
    
    return any(dfs(n) for n in graph if color[n] == WHITE)

def count_paths_dag(graph, start, end):
    @cache
    def dp(node, seen_dac, seen_fft):
        # Update flags based on current node
        seen_dac = seen_dac or (node == 'dac')
        seen_fft = seen_fft or (node == 'fft')
        
        if node == end:
            return 1 if (seen_dac and seen_fft) else 0
        
        return sum(dp(neighbor, seen_dac, seen_fft) 
                   for neighbor in graph.get(node, []))
    
    return dp(start, False, False)

def runa(type_mode: str, date: str) -> None:
    """Entry point for running the solution for the given date.

    For now this just loads the input data and reports how many lines
    were loaded so you can plug in puzzle-specific logic later.
    """
    data = get_data(type_mode, date)
    graph = make_graph_dict(data)
    print(f"Graph has {len(graph)} nodes.")
    for node, edges in graph.items():
        print(f"Node {node} has edges to: {edges}")
    start_node = 'you'
    end_node = 'out'
    total_paths = count_paths(graph, start_node, end_node, set([start_node]))
    print(f"Total distinct paths from {start_node} to {end_node}: {total_paths}")

def runb(type_mode: str, date: str) -> None:
    """Entry point for running the solution for the given date.

    For now this just loads the input data and reports how many lines
    were loaded so you can plug in puzzle-specific logic later.
    """
    data = get_data(type_mode, date)
    graph = make_graph_dict(data)
    # Has cycle check
    if has_cycle(graph):
        print("Graph has cycles.")
    
    #for node, edges in graph.items():
    #    print(f"Node {node} has edges to: {edges}")
    start_node = 'svr'
    end_node = 'out'
    total_paths = count_paths_dag(graph, start_node, end_node)
    #total_paths = count_paths_part2(graph, start_node, end_node, set([start_node]))
    print(f"Total distinct paths from {start_node} to {end_node}: {total_paths}")


if __name__ == "__main__":
    date = "dec11"
    #type_mode = "test"
    type_mode = "data"
    #runa(type_mode, date)
    runb(type_mode, date)