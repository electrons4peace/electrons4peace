import utils.common_utils as common_utils
import os
print(os.getcwd())

def get_data(type_mode: str, date: str) -> list[str]:
    """Return the raw input lines for the given Advent of Code date.

    This is a thin, typed wrapper around common_utils.get_data_list.
    """
    data = common_utils.get_data_list(type_mode, date)
    return data



def get_input_data(data: list[str]) -> list[str]:
    row_input_numbers = []
    tmp_row = data[0].split()
    row_input_numbers.extend([tmp_row])
    for line in data[1:]:
        #split line into parts by one or more whitespace
        row_data = line.split()
        if row_data[0].isdigit():
            row_input_numbers.extend([row_data])
        else:
            row_operations = line.split()

    return row_input_numbers, row_operations

def get_input_matrix() -> list[list[str]]:
    with open("c:/Users/mathe/OneDrive/electrons4peace/learning_projects/advent_of_code_2025/data/dec06_data.txt") as f:
        lines = f.read().split("\n")[:-1]
    matrix = []
    #Build the matrix from the input lines except the last line
    for line in lines[:-1]:
        tmp_list = []
        for char in line:
            tmp_list.append(char)
        matrix.append(tmp_list)
        print(f"Added row: {len(tmp_list)} elements")
    return matrix

def get_right_to_left_columns_old(input_matrix: list[list[str]]) -> list[list[str]]:
    # Get the columns from right to left of the input matrix
    # If there are " " in all columns, they should be excluded from the result and all current numbers should be included in the sub list
    # If a " " is found this char should be excluded from the sub list and the current sub list should be added to the result
    rows = len(input_matrix)
    cols = len(input_matrix[0])
    right_to_left_columns = []
    for c in range(cols-1, -1, -1):
        col_str = ""
        col_list = [] 
        is_all_spaces = True
        for r in range(rows):
            if input_matrix[r][c] != " ":
                col_str += input_matrix[r][c]
                is_all_spaces = False
        if not is_all_spaces:
            col_list.append(col_str)
        else:
            right_to_left_columns.append(col_list)
    return right_to_left_columns

def get_right_to_left_columns(input_matrix: list[list[str]]) -> list[list[str]]:
    """
    Extract numbers grouped by problems in cephalopod math format.
    - Reads columns from right to left
    - All-space columns separate problems
    - Each column forms one number (reading top to bottom, excluding spaces)
    """
    rows = len(input_matrix)
    cols = len(input_matrix[0])
    right_to_left_columns = []
    current_problem = []
    
    for c in range(cols - 1, -1, -1):
        # Check if this column is all spaces (separator)
        is_all_spaces = True
        for r in range(rows):
            if input_matrix[r][c] != " ":
                is_all_spaces = False
                break
        
        if is_all_spaces:
            # Separator found - save current problem if it has numbers
            if current_problem:
                right_to_left_columns.append(current_problem)
                current_problem = []
        else:
            # Extract the number from this column (top to bottom, skip spaces)
            col_str = ""
            for r in range(rows):
                if input_matrix[r][c] != " ":
                    col_str += input_matrix[r][c]
            current_problem.append(col_str)
    
    # Don't forget the last problem (leftmost)
    if current_problem:
        right_to_left_columns.append(current_problem)
    
    return right_to_left_columns



def get_transposed_list(row_input_numbers: list[list[str]]) -> list[list[str]]:
    #Transpose the list of lists
    transposed = []
    for col_idx in range(len(row_input_numbers[0])):
        new_col = []
        for row in row_input_numbers:
            new_col.append(row[col_idx])
        transposed.append(new_col)
    return transposed

def calculate_operation_on_column(column: list[str], operation: str) -> int:
    # Calculate the result of the operation on the column by use of eval()
    tmp_str = ""
    #Exclude the last element
    for elem in column[:-1]:
        tmp_str += elem + operation
    tmp_str += column[-1]
    result = eval(tmp_str)
    return result


def runa(type_mode: str, date: str) -> None:
    """Entry point for running the solution for the given date.

    For now this just loads the input data and reports how many lines
    were loaded so you can plug in puzzle-specific logic later.
    """
    data = get_data(type_mode, date)
    print(f"Loaded {len(data)} lines for {date!r} with mode {type_mode!r}.")
    print(data)
    row_input_numbers, row_operations = get_input_data(data)
    print(f"Input numbers: {row_input_numbers}")
    print(f"Operations: {row_operations}")
    transposed = get_transposed_list(row_input_numbers)
    print(f"Transposed input numbers: {transposed}")
    tot_sum = 0
    for idx in range(len(row_operations)):
        operation = row_operations[idx]
        column = transposed[idx]
        result = calculate_operation_on_column(column, operation)
        print(f"Result of operation {operation} on column {column}: {result}")
        tot_sum += result
    print(f"Total sum of all operations: {tot_sum}")

def runb(type_mode: str, date: str) -> None:
    input_matrix = get_input_matrix()
    print(f"Input matrix: {input_matrix}")
    right_to_left_columns = get_right_to_left_columns(input_matrix)
    print(f"Right to left columns: {right_to_left_columns}")
    data = get_data(type_mode, date)
    _ , row_operations = get_input_data(data)
    reversed_row_operations = row_operations[::-1]
    tot_sum = 0

    for problem_idx in range(len(right_to_left_columns)):
        problem_columns = right_to_left_columns[problem_idx]
        problem_operations = reversed_row_operations[problem_idx]
        print(f"Processing problem {problem_idx+1} with columns {problem_columns} and operations {problem_operations}")
        result = calculate_operation_on_column(problem_columns, problem_operations)
        print(f"Result of problem {problem_idx+1}: {result}")
        tot_sum += result
    print(f"Total sum of all problems: {tot_sum}")    





if __name__ == "__main__":
    date = "dec06"
    #type_mode = "test"
    type_mode = "data"
    #runa(type_mode, date)
    runb(type_mode, date)