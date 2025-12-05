import utils.common_utils as common_utils


def get_data(type_mode: str, date: str) -> list[str]:
    """Return the raw input lines for the given Advent of Code date.

    This is a thin, typed wrapper around common_utils.get_data_list.
    """
    data = common_utils.get_data_list(type_mode, date)
    return data


def get_max_and_remaining(input_str: str) -> tuple[int, str]:
    # Extract the maximum number and the remaining string from the input
    max_int = -1
    remaining_str = ""
    for index, char in enumerate(input_str):
        if int(char) > max_int:
            max_int = int(char)
            remaining_str = input_str[index + 1 :]
    return max_int, remaining_str

#def get_max_and_remaining_v2(input_str: str) -> tuple[int, str]:
#    max_idx = max(range(len(input_str)), key=lambda i: int(input_str[i]))
#    return int(input_str[max_idx]), input_str[max_idx + 1:]


def get_max_candidates(input_str: str) -> list[int]:
    candidates = []
    for idx_1 in range(len(input_str)-1):
        first_char = input_str[idx_1]
        tmp = get_max_and_remaining(input_str[idx_1+1:])
        max_after_first, rem_str = tmp
        second_char = str(max_after_first)
        candidates.append(int(first_char + second_char))
    return candidates


def get_max_str(input_str: str) -> int:
    candidates = get_max_candidates(input_str)
    return max(candidates)

# def get_long_max_str(input_str: str, max_len: int) -> int:
#     max_str = ""
#     active_str = input_str
#     #Itererate in reverse
#     for idx in range(max_len, 0, -1):
#         #remove the last idx characters
#         tmp_str = active_str[:-idx]
#         #save the last idx characters
#         last_idx_chars = active_str[-idx:]
#         tmp_max_int , rem_str = get_max_and_remaining(tmp_str)
#         max_str += str(tmp_max_int)
#         active_str = rem_str + last_idx_chars
#     return int(max_str)

def get_long_max_str(input_str: str, max_len: int) -> int:
    max_str = ""
    active_str = input_str
    for idx in range(max_len, 0, -1):
        # Reserve (idx - 1) characters, not idx
        if idx > 1:
            tmp_str = active_str[:-(idx - 1)]
            last_idx_chars = active_str[-(idx - 1):]
        else:
            tmp_str = active_str  # Search entire string for last digit
            last_idx_chars = ""
        
        tmp_max_int, rem_str = get_max_and_remaining(tmp_str)
        max_str += str(tmp_max_int)
        active_str = rem_str + last_idx_chars
    return int(max_str)


def get_long_max_str_v2(input_str: str, max_len: int) -> int:
    max_str = ""
    active_str = ""
    # Iterate in forward direction
    for idx in range(max_len, len(input_str)):
        # Check if the current character is greater than the previous character
        if input_str[idx] > input_str[idx - 1]:
            active_str += input_str[idx]
        else:
            # If not, compare the length of the current active string with the max string
            if len(active_str) > len(max_str):
                max_str = active_str
            active_str = ""
    # Compare the length of the last active string with the max string
    if len(active_str) > len(max_str):
        max_str = active_str
    return int(max_str)

def get_long_max_str_queue(input_str: str, max_len: int) -> int:
    max_str = ""
    queue = [(input_str, max_str)]
    while queue:
        current_str, current_max_str = queue.pop(0)
        if len(current_str) < max_len:
            if len(current_max_str) > len(max_str):
                max_str = current_max_str
            continue
        next_char = current_str[0]
        new_max_str = current_max_str + next_char
        remaining_str = current_str[1:]
        queue.append((remaining_str, new_max_str))
        queue.append((remaining_str, current_max_str))
    return int(max_str)



def runa(type_mode: str, date: str) -> None:
    """Entry point for running the solution for the given date.

    For now this just loads the input data and reports how many lines
    were loaded so you can plug in puzzle-specific logic later.
    """
    data = get_data(type_mode, date)
    print(f"Loaded {len(data)} lines for {date!r} with mode {type_mode!r}.")
    print(data)
    sum_max = 0
    for line in data:
        max_str = get_max_str_v2(line)
        sum_max += max_str
        print(f"Max string value for line {line} is {max_str}")
    print(f"Sum of max string values: {sum_max}")

def runb(type_mode: str, date: str) -> None:
    """Entry point for running the solution for the given date.

    For now this just loads the input data and reports how many lines
    were loaded so you can plug in puzzle-specific logic later.
    """
    data = get_data(type_mode, date)
    print(f"Loaded {len(data)} lines for {date!r} with mode {type_mode!r}.")
    print(data)
    # Implement part B logic here
    str_len = 12
    sum_long_max = 0
    for line in data:
        long_max_str = get_long_max_str(line, str_len)
        print(f"Long max string value for line {line} with length {str_len} is {long_max_str}")
        sum_long_max += long_max_str
    print(f"Sum of long max string values: {sum_long_max}")


def run_debug() -> None:
    temp_str = "234234234234278"
    long_max_str = get_long_max_str(temp_str, 12)
    print(f"Long max string value for line {temp_str} with length 3 is {long_max_str}")


if __name__ == "__main__":
#    temp_str = "1234321"
#    result, rem_str = get_max_and_remaining_v2(temp_str)
#    print(f"Max value: {result}, Remaining string: {rem_str}")
    date = "dec03"
    #type_mode = "test"
    type_mode = "data"
    #runa(type_mode, date)
    runb(type_mode, date)
    #run_debug()