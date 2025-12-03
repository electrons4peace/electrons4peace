import utils.common_utils as common_utils



def get_data_to_list(data: str, split_char: str) -> list:
    # Remove \n characters
    data_list = data.split(split_char)
    for idx, elem in enumerate(data_list):
        data_list[idx] = elem.strip()
    return data_list

def get_problem_list(data: str) -> list:
    output_list = []
    temp_list = get_data_to_list(data, ',')
    for elem in temp_list:
        new_elem = elem.split('-')
        output_list.append([int(new_elem[0]), int(new_elem[1])])
    return output_list

def is_repeat_in_int(input_int: int) -> bool:
    input_str = str(input_int)
    half_len = len(input_str) // 2
    first_half = input_str[:half_len]
    second_half = input_str[half_len:]
    return first_half == second_half

def number_of_repeats(input_list: list) -> int:
    sum_count_repeats = 0
    lower_bound = input_list[0]
    upper_bound = input_list[1]
    for num in range(lower_bound, upper_bound + 1):
        if is_repeat_in_int(num):
            sum_count_repeats += num
    return sum_count_repeats

def generate_repeated_patterns(max_val: int, min_val: int = 1) -> set[int]:
    """
    Generate all repeated-pattern numbers in [min_val, max_val].
    
    For each pattern length p (1, 2, 3, ...):
        For each pattern value (avoiding leading zeros):
            For each repetition count r (2, 3, 4, ...):
                Create: pattern * r
    """
    result = set()
    max_digits = len(str(max_val))
    
    # Pattern can be at most half the total digits (need ≥2 repetitions)
    for pattern_len in range(1, max_digits // 2 + 1):
        # Avoid leading zeros: pattern >= 10^(p-1) for p > 1
        min_pattern = 1 if pattern_len == 1 else 10 ** (pattern_len - 1)
        max_pattern = 10 ** pattern_len - 1
        
        for pattern in range(min_pattern, max_pattern + 1):
            pattern_str = str(pattern)
            
            # Try 2, 3, 4, ... repetitions
            for reps in range(2, max_digits // pattern_len + 1):
                num = int(pattern_str * reps)
                if num > max_val:
                    break
                if num >= min_val:
                    result.add(num)
    
    return sorted(result)

def get_max_int_from_list(input_data: list) -> int:
    max_int = -999999999
    for elems in input_data:
        elem = elems[1]
        if elem > max_int:
            max_int = elem
    return max_int

def runa(type_mode: str, date: str) -> None:
    data = common_utils.read_input_file(type_mode, date)
    print(data)
    data_list = get_data_to_list(data, ',')
    print(data_list)
    problem_list = get_problem_list(data)
    print(problem_list)
    sum_repeats = 0
    for elem in problem_list:
        count_repeats = number_of_repeats(elem)
        sum_repeats += count_repeats
        print(f"Number of repeats between {elem[0]} and {elem[1]}: {count_repeats} -> {sum_repeats}")

def runb(type_mode: str, date: str) -> None:
    data = common_utils.read_input_file(type_mode, date)
    problem_list = get_problem_list(data)
    print(problem_list)
    #Print max int from data
    max_int = get_max_int_from_list(problem_list)
    print(f"Max int from data: {max_int}")
    invalid_numbers = generate_repeated_patterns(max_int, 1)
    print(f"Generated {len(invalid_numbers)} repeated-pattern numbers up to {max_int}:")
    sum_of_invalids_idx = 0
    for elem in problem_list:
        count_invalids = 0
        for invalid_num in invalid_numbers:
            if (invalid_num >= elem[0]) & (invalid_num <= elem[1]):
                count_invalids += 1
                sum_of_invalids_idx += invalid_num
        print(f"Number of repeated-pattern numbers between {elem[0]} and {elem[1]}: {count_invalids} -> Sum so far: {sum_of_invalids_idx}")

if __name__ == "__main__":
    date = "dec02"
#    type_mode = "test"
    type_mode = "data"
#    runa(type_mode, date)
    runb(type_mode, date)
