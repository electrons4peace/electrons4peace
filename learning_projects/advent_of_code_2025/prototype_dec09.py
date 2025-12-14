"""
Scanline-based solution for AoC Day 9 Part 2.
Replaces flood-fill with O(n log n) band construction.

Key insight: For a simple rectilinear polygon, we need to handle:
1. Interior rows (strictly between horizontal edges): use vertical edge even-odd pairing
2. Boundary rows (at horizontal edges): merge horizontal edge intervals with interior
"""

from collections import defaultdict


def extract_edges(red_tiles):
    """
    Extract horizontal and vertical edges from the polygon.
    Returns:
        h_edges: list of (y, x_min, x_max) for horizontal edges
        v_edges: list of (x, y_min, y_max) for vertical edges
    """
    h_edges = []
    v_edges = []
    n = len(red_tiles)
    
    for i in range(n):
        x1, y1 = red_tiles[i]
        x2, y2 = red_tiles[(i + 1) % n]
        
        if y1 == y2:  # Horizontal edge
            h_edges.append((y1, min(x1, x2), max(x1, x2)))
        else:  # Vertical edge
            v_edges.append((x1, min(y1, y2), max(y1, y2)))
    
    return h_edges, v_edges


def merge_intervals(intervals):
    """Merge overlapping/adjacent intervals."""
    if not intervals:
        return []
    
    # Sort by start, then by end
    sorted_intervals = sorted(intervals)
    merged = [sorted_intervals[0]]
    
    for start, end in sorted_intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + 1:  # Overlapping or adjacent
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    
    return merged


def build_bands_scanline(red_tiles):
    """
    Build bands using scanline algorithm.
    
    For interior rows: use even-odd pairing of vertical edges that SPAN the row
    For boundary rows: merge horizontal edge intervals with vertical edge contributions
    
    Returns:
        bands: list of (y_start, y_end, intervals) sorted by y_start
        bbox: (min_x, max_x, min_y, max_y) bounding box
    """
    h_edges, v_edges = extract_edges(red_tiles)
    
    if not v_edges and not h_edges:
        return [], None
    
    # Collect all interesting y-coordinates (vertices)
    y_set = set()
    for y, x1, x2 in h_edges:
        y_set.add(y)
    for x, y_min, y_max in v_edges:
        y_set.add(y_min)
        y_set.add(y_max)
    
    y_sorted = sorted(y_set)
    
    # Build horizontal edge lookup
    h_edge_at_y = defaultdict(list)
    for y, x1, x2 in h_edges:
        h_edge_at_y[y].append((x1, x2))
    
    bands = []
    global_min_x = float('inf')
    global_max_x = float('-inf')
    global_min_y = float('inf')
    global_max_y = float('-inf')
    
    for i, y in enumerate(y_sorted):
        # === Compute intervals at row y (vertex row) ===
        
        # 1. Vertical edges that COUNT for even-odd at y
        #    Standard convention: y_min <= y < y_max (lower endpoint in, upper out)
        crossing_xs = [x for x, y_min, y_max in v_edges if y_min <= y < y_max]
        crossing_xs.sort()
        
        # Even-odd pairing for interior
        interior_intervals = []
        for j in range(0, len(crossing_xs) - 1, 2):
            interior_intervals.append((crossing_xs[j], crossing_xs[j + 1]))
        
        # 2. Horizontal edge intervals at this y (boundary contribution)
        h_intervals = h_edge_at_y[y]
        
        # 3. ALL vertical edge points at this y (any point on a vertical edge is boundary)
        v_edge_xs = [x for x, y_min, y_max in v_edges if y_min <= y <= y_max]
        point_intervals = [(x, x) for x in v_edge_xs]
        
        # Merge all contributions
        all_intervals = interior_intervals + h_intervals + point_intervals
        merged = merge_intervals(all_intervals)
        
        if merged:
            bands.append((y, y, tuple(merged)))
            for x1, x2 in merged:
                global_min_x = min(global_min_x, x1)
                global_max_x = max(global_max_x, x2)
            global_min_y = min(global_min_y, y)
            global_max_y = max(global_max_y, y)
        
        # === Compute intervals for interior rows y+1 to y_next-1 ===
        if i + 1 < len(y_sorted):
            y_next = y_sorted[i + 1]
            
            if y_next > y + 1:
                # Vertical edges that cross rows in (y, y_next)
                # An edge crosses row y' if y_min <= y' < y_max
                # For all y' in (y, y_next-1], need y_min <= y+1 and y_max > y_next-1
                # Simplified: y_min <= y and y_max >= y_next
                spanning_xs_band = [x for x, y_min, y_max in v_edges 
                                   if y_min <= y and y_max >= y_next]
                spanning_xs_band.sort()
                
                # Even-odd pairing
                band_intervals = []
                for j in range(0, len(spanning_xs_band) - 1, 2):
                    band_intervals.append((spanning_xs_band[j], spanning_xs_band[j + 1]))
                
                if band_intervals:
                    bands.append((y + 1, y_next - 1, tuple(band_intervals)))
                    for x1, x2 in band_intervals:
                        global_min_x = min(global_min_x, x1)
                        global_max_x = max(global_max_x, x2)
                    global_min_y = min(global_min_y, y + 1)
                    global_max_y = max(global_max_y, y_next - 1)
    
    if global_min_x == float('inf'):
        return [], None
    
    bbox = (global_min_x, global_max_x, global_min_y, global_max_y)
    return bands, bbox


def x_span_valid_in_intervals(min_x, max_x, intervals):
    """Check if [min_x, max_x] is fully contained in one interval."""
    for start, end in intervals:
        if start <= min_x and max_x <= end:
            return True
        if start > max_x:  # Intervals are sorted
            break
    return False


def is_valid_rectangle_bands(p1, p2, bands, bbox):
    """
    Fast rectangle validation using bands.
    
    Checks that the rectangle [min_x, max_x] × [min_y, max_y] is fully
    covered by valid tiles using the precomputed band structure.
    
    Complexity: O(log(bands) + bands_in_range)
    """
    x1, y1 = p1
    x2, y2 = p2
    
    min_x, max_x = min(x1, x2), max(x1, x2)
    min_y, max_y = min(y1, y2), max(y1, y2)
    
    # AABB quick reject against overall bounding box
    bbox_min_x, bbox_max_x, bbox_min_y, bbox_max_y = bbox
    if min_x < bbox_min_x or max_x > bbox_max_x:
        return False
    if min_y < bbox_min_y or max_y > bbox_max_y:
        return False
    
    # Binary search for first band that could contain min_y
    lo, hi = 0, len(bands)
    while lo < hi:
        mid = (lo + hi) // 2
        if bands[mid][1] < min_y:  # band ends before min_y
            lo = mid + 1
        else:
            hi = mid
    
    # Walk through bands, verifying:
    # 1. Bands cover [min_y, max_y] contiguously (no gaps)
    # 2. Each band's intervals contain [min_x, max_x]
    expected_y = min_y
    i = lo
    
    while expected_y <= max_y:
        if i >= len(bands):
            return False  # Ran out of bands
        
        y_start, y_end, intervals = bands[i]
        
        if y_start > expected_y:
            return False  # Gap in y coverage
        
        if not x_span_valid_in_intervals(min_x, max_x, intervals):
            return False  # x-span not covered in this band
        
        expected_y = y_end + 1
        i += 1
    
    return True


def rectangle_area(p1, p2):
    """Calculate area of rectangle with opposite corners p1 and p2."""
    return (abs(p2[0] - p1[0]) + 1) * (abs(p2[1] - p1[1]) + 1)


def find_largest_valid_rectangle_scanline(red_tiles):
    """
    Find largest valid rectangle using scanline-built bands.
    
    Complexity: O(n log n) for band construction + O(n² × bands_per_rect) for search
    """
    bands, bbox = build_bands_scanline(red_tiles)
    
    if not bands:
        return 0, None
    
    best_area = 0
    best_corners = None
    n = len(red_tiles)
    
    for i in range(n):
        for j in range(i + 1, n):
            p1 = red_tiles[i]
            p2 = red_tiles[j]
            
            area = rectangle_area(p1, p2)
            
            # Skip if can't beat current best
            if area <= best_area:
                continue
            
            if is_valid_rectangle_bands(p1, p2, bands, bbox):
                best_area = area
                best_corners = (p1, p2)
    
    return best_area, best_corners


# =============================================================================
# Drop-in replacement for runb
# =============================================================================

def runb_scanline(type_mode: str, date: str) -> None:
    """
    Drop-in replacement for runb using scanline algorithm.
    No flood fill, works with huge coordinate spaces.
    """
    import utils.common_utils as common_utils
    
    data = common_utils.get_data_list(type_mode, date)
    red_tiles = [(int(p.split(',')[0]), int(p.split(',')[1])) for p in data]
    
    print(f"Loaded {len(red_tiles)} red tiles")
    
    # Build bands (replaces flood fill)
    bands, bbox = build_bands_scanline(red_tiles)
    print(f"Built {len(bands)} bands, bbox={bbox}")
    
    # Find largest rectangle
    area, corners = find_largest_valid_rectangle_scanline(red_tiles)
    print(f"Largest valid rectangle area: {area} between corners {corners}")


# =============================================================================
# Test with example from problem
# =============================================================================

def test_example():
    """Test with the example from the problem description."""
    red_tiles = [
        (7, 1),
        (11, 1),
        (11, 7),
        (9, 7),
        (9, 5),
        (2, 5),
        (2, 3),
        (7, 3),
    ]
    
    print("Red tiles:", red_tiles)
    print()
    
    # Build bands
    bands, bbox = build_bands_scanline(red_tiles)
    print(f"Bands ({len(bands)} total):")
    for y_start, y_end, intervals in bands:
        print(f"  y=[{y_start}, {y_end}]: intervals={intervals}")
    print(f"Bounding box: {bbox}")
    print()
    
    # Find largest rectangle
    area, corners = find_largest_valid_rectangle_scanline(red_tiles)
    print(f"Largest valid rectangle: area={area}, corners={corners}")
    print(f"Expected: area=24")
    
    return area == 24


if __name__ == "__main__":
    success = test_example()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")