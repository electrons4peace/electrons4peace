import utils.common_utils as common_utils
#Data format:
#3-5
#10-14
#
#1
#5
#8

def get_data(type_mode: str, date: str) -> list[str]:
    """Return the raw input lines for the given Advent of Code date.

    This is a thin, typed wrapper around common_utils.get_data_list.
    """
    data = common_utils.get_data_list(type_mode, date)
    return data

def get_id_ranges(data: list[str]) -> list[tuple[int, int]]:
    id_ranges = []
    for elem in data:
        if elem == "":
            continue
        parts = elem.split("-")
        if len(parts) != 2:
            continue
        start_id = int(parts[0])
        end_id = int(parts[1])
        id_ranges.append((start_id, end_id))
    return id_ranges

def get_available_ids(data: list[str]) -> list[int]:
    # Get all IDs that after the line feed
    avalible_ids = []
    for elem in data:
        if elem == "":
            continue
        if "-" in elem:
            continue
        avalible_ids.append(int(elem))
    return avalible_ids

def remove_overlapping_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    # Remove overlapping ranges from the list of ranges
    if not ranges:
        return []
    # Sort ranges by start value
    sorted_ranges = sorted(ranges, key=lambda x: x[0])
    merged_ranges = [sorted_ranges[0]]
    for current in sorted_ranges[1:]:
        last_merged = merged_ranges[-1]
        if current[0] <= last_merged[1]:  # Overlap
            merged_ranges[-1] = (last_merged[0], max(last_merged[1], current[1]))
        else:
            merged_ranges.append(current)
    return merged_ranges

def get_brute_force_approach_non_overlapping_ranges(data:list) -> int:
    available_ids = get_available_ids(data)
    id_ranges = get_id_ranges(data)
    non_overlapping_ranges = remove_overlapping_ranges(id_ranges)
    number_of_free_ids = 0
    for id in available_ids:
        for id_range in non_overlapping_ranges:
            if id_range[0] <= id <= id_range[1]:
                number_of_free_ids += 1
    return number_of_free_ids



def runa(type_mode: str, date: str) -> None:
    """Entry point for running the solution for the given date.

    For now this just loads the input data and reports how many lines
    were loaded so you can plug in puzzle-specific logic later.
    """
    data = get_data(type_mode, date)
    print(f"Loaded {len(data)} lines for {date!r} with mode {type_mode!r}.")
    print(data)
    id_ranges = get_id_ranges(data)
    print(f"ID Ranges: {id_ranges}")
    available_ids = get_available_ids(data)
    print(f"Available IDs: {available_ids}")
    non_overlapping_ranges = remove_overlapping_ranges(id_ranges)
    print(f"Non-overlapping ID Ranges: {non_overlapping_ranges}")
    number_of_free_ids = get_brute_force_approach_non_overlapping_ranges(data)
    print(f"Number of free IDs (brute-force approach): {number_of_free_ids}")

def runb(type_mode: str, date: str) -> None:
    """Entry point for running the solution for the given date in part B.

    For now this just loads the input data and reports how many lines
    were loaded so you can plug in puzzle-specific logic later.
    """
    data = get_data(type_mode, date)
    id_ranges = get_id_ranges(data)
    non_overlapping_ranges = remove_overlapping_ranges(id_ranges)
    sum_of_ranges = 0
    for id_range in non_overlapping_ranges:
        range_size = id_range[1] - id_range[0] + 1
        sum_of_ranges += range_size
    print(f"Sum of sizes of non-overlapping ID ranges: {sum_of_ranges}")


if __name__ == "__main__":
    date = "dec05"
    #type_mode = "test"
    type_mode = "data"
    #runa(type_mode, date)
    runb(type_mode, date)