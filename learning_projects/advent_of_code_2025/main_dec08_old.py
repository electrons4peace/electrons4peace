import utils.common_utils as common_utils


def get_data(type_mode: str, date: str) -> list[str]:
    """Return the raw input lines for the given Advent of Code date.

    This is a thin, typed wrapper around common_utils.get_data_list.
    """
    data = common_utils.get_data_list(type_mode, date)
    return data

def get_coordinates_list(data: list[str]) -> list[tuple[int, int, int]]:
    coord_map = []
    for line in data:
        parts = line.split(',')
        coord = (int(parts[0]), int(parts[1]), int(parts[2]))
        coord_map.append(coord)
    return coord_map

def get_euclidean_distance_squared(coord1: tuple[int, int, int], coord2: tuple[int, int, int]) -> int:
    dx = coord1[0] - coord2[0]
    dy = coord1[1] - coord2[1]
    dz = coord1[2] - coord2[2]
    return dx * dx + dy * dy + dz * dz

def get_closest_pair(coord_list: list[tuple[int, int, int]]) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    min_distance_sq = float('inf')
    closest_pair = ((), ())
    for i in range(len(coord_list)):
        for j in range(i + 1, len(coord_list)):
            dist_sq = get_euclidean_distance_squared(coord_list[i], coord_list[j])
            if dist_sq < min_distance_sq:
                min_distance_sq = dist_sq
                closest_pair = (coord_list[i], coord_list[j])
    return closest_pair

def get_closest_coordinate(coord: tuple[int, int, int], coord_list: list[tuple[int, int, int]]) -> tuple[int, int, int]:
    #Find the closest coordinate to the given coord from coord_list (excluding itself).
    min_distance_sq = float('inf')
    closest_coord = (0, 0, 0)
    for other_coord in coord_list:
        if other_coord == coord:
            continue
        dist_sq = get_euclidean_distance_squared(coord, other_coord)
        if dist_sq < min_distance_sq:
            min_distance_sq = dist_sq
            closest_coord = other_coord
    return closest_coord


def get_closest_map_pair(coord_list: list[tuple[int, int, int]]) -> dict[int, int, int]:
    closest_map = {}
    for coord in coord_list:
        closest_coord = get_closest_coordinate(coord, coord_list)
        closest_map[coord] = closest_coord
    return closest_map

def to_undirected_neighbors(closest_neighbor_graph: dict) -> dict[str, set]:
    neighbors = {v: set() for v in closest_neighbor_graph}
    
    for vertex, closest in closest_neighbor_graph.items():
        neighbors[vertex].add(closest)
        neighbors[closest].add(vertex)
    
    return neighbors

def find_clusters(neighbors: dict[str, set]) -> list[set]:
    visited = set()
    clusters = []
    
    for start in neighbors:
        if start in visited:
            continue
        
        # BFS to find all vertices in this component
        cluster = set()
        queue = [start]
        
        while queue:
            vertex = queue.pop()
            if vertex in visited:
                continue
            visited.add(vertex)
            cluster.add(vertex)
            queue.extend(neighbors[vertex] - visited)
        
        clusters.append([cluster])
    
    return clusters



def runa(type_mode: str, date: str) -> None:
    """Entry point for running the solution for the given date.

    For now this just loads the input data and reports how many lines
    were loaded so you can plug in puzzle-specific logic later.
    """
    data = get_data(type_mode, date)
    coord_list = get_coordinates_list(data)
    #closest_pair = get_closest_pair(coord_list)
    #print(f"Closest pair of coordinates: {closest_pair}")
    closest_map = get_closest_map_pair(coord_list)
    print(f"Closest map of coordinates: {closest_map}")
    # Count clusters with more than one coordinate
    neighbors = to_undirected_neighbors(closest_map)
    clusters = find_clusters(neighbors)
    print(f"Number of clusters: {clusters}")
    # Print clusters
    for idx, cluster in enumerate(clusters):
        print(f"Cluster {idx + 1}: {cluster}")


if __name__ == "__main__":
    date = "dec08"
    type_mode = "test"
    #type_mode = "data"
    runa(type_mode, date)
    #runb(type_mode, date)