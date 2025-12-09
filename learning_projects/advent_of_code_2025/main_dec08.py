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

def get_n_closest_coordinate_pairs(coord_list: list[tuple[int, int, int]], n: int) -> list[tuple[tuple[int, int, int], tuple[int, int, int]]]:
    # Find the n closest pairs of coordinates in coord_list.
    distance_pairs = []
    for i in range(len(coord_list)):
        for j in range(i + 1, len(coord_list)):
            dist_sq = get_euclidean_distance_squared(coord_list[i], coord_list[j])
            distance_pairs.append((dist_sq, (coord_list[i], coord_list[j])))
    distance_pairs.sort(key=lambda x: x[0])
    # Remove the distance and keep only the coordinate pairs in clusters.
    distance_pairs = [pair for _, pair in distance_pairs]
    return distance_pairs[:n]



def find_clusters_in_a_list_of_neighbors(neighbors: list[tuple[tuple[int, int, int], tuple[int, int, int]]]) -> list[set]:
    #Find clusters of connected coordinates from a list of neighbor pairs. Return a list of sets of connected coordinates.
    clusters = []
    visited = set()
    for neighbor_pair in neighbors:
        coord1, coord2 = neighbor_pair
        if coord1 in visited and coord2 in visited:
            continue
        new_cluster = set()
        stack = [coord1, coord2]
        while stack:
            coord = stack.pop()
            if coord not in visited:
                visited.add(coord)
                new_cluster.add(coord)
                for n_pair in neighbors:
                    if n_pair[0] == coord and n_pair[1] not in visited:
                        stack.append(n_pair[1])
                    elif n_pair[1] == coord and n_pair[0] not in visited:
                        stack.append(n_pair[0])
        clusters.append(new_cluster)

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
    closest_map = get_n_closest_coordinate_pairs(coord_list, 1000)
    print(f"Closest map of coordinates: {closest_map}")
    clusters = find_clusters_in_a_list_of_neighbors(closest_map)
    print(f"Clusters found: {clusters}")
    print(f"Number of clusters: {len(clusters)}")
    # Print clusters
    for idx, cluster in enumerate(clusters):
        print(f"Cluster {idx + 1}: {cluster}")
    # Print the three largest clusters with size and the product of their sizes
    clusters.sort(key=lambda x: len(x), reverse=True)
    largest_clusters = clusters[:3]
    product_of_sizes = 1
    for cluster in largest_clusters:
        size = len(cluster)
        product_of_sizes *= size
        print(f"Largest cluster size: {size}")
    print(f"Product of largest cluster sizes: {product_of_sizes}")



class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # path compression
        return self.parent[x]
    
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False  # already same circuit
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True  # merged two circuits

def solve(points):
    n = len(points)
    
    # Generate all pairs with squared distances (avoid sqrt)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = points[i][0] - points[j][0]
            dy = points[i][1] - points[j][1]
            dz = points[i][2] - points[j][2]
            dist_sq = dx*dx + dy*dy + dz*dz
            edges.append((dist_sq, i, j))
    
    edges.sort()  # O(N² log N)
    
    uf = UnionFind(n)
    components = n
    
    for dist_sq, i, j in edges:
        if uf.union(i, j):
            components -= 1
            if components == 1:
                return points[i][0] * points[j][0]
    
    return None

def run_claude(type_mode: str, date: str) -> None:
    data = get_data(type_mode, date)
    coord_list = get_coordinates_list(data)
    print(solve(coord_list))


if __name__ == "__main__":
    date = "dec08"
    #type_mode = "test"
    type_mode = "data"
    runa(type_mode, date)
    #runb(type_mode, date)
    run_claude(type_mode, date)