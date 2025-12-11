import utils.common_utils as common_utils


def get_data(type_mode: str, date: str) -> list[str]:
    """Return the raw input lines for the given Advent of Code date.

    This is a thin, typed wrapper around common_utils.get_data_list.
    """
    data = common_utils.get_data_list(type_mode, date)
    return data

def split_string(input_string):
    return [x.strip() for x in input_string.split(':') if x.strip()]

def make_graph_dict(data: list[str]) -> dict[str, list[str]]:
    graph_dict = {}
    for line in data:
        parts = split_string(line)
        node = parts[0]
        edges = parts[1].split(',') if len(parts) > 1 else []
        graph_dict[node] = [edge.strip() for edge in edges]
    return graph_dict

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


if __name__ == "__main__":
    date = "dec11"
    type_mode = "test"
    #type_mode = "data"
    runa(type_mode, date)
    #runb(type_mode, date)