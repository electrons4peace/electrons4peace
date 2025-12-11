import utils.common_utils as common_utils
import re
from collections import deque
import pulp
from pulp import LpProblem, LpMinimize, LpVariable, LpInteger, lpSum, LpStatus, value
def get_data(type_mode: str, date: str) -> list[str]:
    """Return the raw input lines for the given Advent of Code date.

    This is a thin, typed wrapper around common_utils.get_data_list.
    """
    data = common_utils.get_data_list(type_mode, date)
    return data

def parse_machine_line(line: str) -> tuple[str, list[tuple[int, ...]], list[int]]:
    diagram_match = re.search(r'\[([.#]+)\]', line)
    indicator_str = diagram_match.group(1)
    
    # Extract button combinations (content in parentheses)
    button_matches = re.findall(r'\(([0-9,]*)\)', line)
    button_tuples = []
    for match in button_matches:
        if match:
            # Parse comma-separated integers into tuple of ints if there are more than one. Otherwise, single int tuple.
            button_tuples.append(tuple(map(int, match.split(','))))
        else:
            # Empty parentheses case
            button_tuples.append(())
    
    # Extract joltage requirements (content in curly braces)
    joltage_match = re.search(r'\{([0-9,]+)\}', line)
    joltage_list = list(map(int, joltage_match.group(1).split(',')))
    
    return indicator_str, button_tuples, joltage_list

def toggle_lights(state: str, buttons: tuple[int, ...]) -> str:
    """
    Toggle light states according to button presses.
    
    Args:
        state: String representing light states ('.' = off, '#' = on)
        buttons: Tuple of button indices to press
        
    Returns:
        New state string after toggling those lights
    """
    state_list = list(state)
    
    for button_idx in buttons:
        # Toggle the light at this position
        state_list[button_idx] = '#' if state_list[button_idx] == '.' else '.'
    
    return ''.join(state_list)


def solve_joltage_machine(buttons: list[tuple[int, ...]], targets: list[int]) -> int | None:
    """
    Solve the joltage configuration problem for a single machine.
    
    Returns the minimum number of button presses, or None if infeasible.
    """
    n_counters = len(targets)
    n_buttons = len(buttons)
    
    # Build incidence matrix A[counter][button] = 1 if button affects counter
    A = [[0] * n_buttons for _ in range(n_counters)]
    for j, button in enumerate(buttons):
        for counter_idx in button:
            if counter_idx < n_counters:  # Safety check
                A[counter_idx][j] = 1
    
    # Create the ILP problem
    prob = LpProblem("JoltageConfig", LpMinimize)
    
    # Decision variables: x_j = number of times button j is pressed
    x = [LpVariable(f"x_{j}", lowBound=0, cat=LpInteger) for j in range(n_buttons)]
    
    # Objective: minimize total button presses
    prob += lpSum(x), "TotalPresses"
    
    # Constraints: for each counter, sum of contributions = target
    for i in range(n_counters):
        prob += (
            lpSum(A[i][j] * x[j] for j in range(n_buttons)) == targets[i],
            f"Counter_{i}"
        )
    
    # Solve (suppress solver output)
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    
    # Check feasibility
    if LpStatus[prob.status] != "Optimal":
        return None
    
    return int(value(prob.objective))


def runb(type_mode: str, date: str):
    """Solve Part 2: minimum total button presses for all machines."""
    total_presses = 0
    lines = get_data(type_mode, date)
    for i, line in enumerate(lines):
        if not line.strip():
            continue
            
        _, buttons, targets = parse_machine_line(line)
        result = solve_joltage_machine(buttons, targets)
        
        if result is None:
            print(f"Warning: Machine {i+1} is infeasible!")
            # Could raise an exception or handle differently
            continue
            
        total_presses += result
    print(f"Total minimum button presses for all machines: {total_presses}")    




def runa(type_mode: str, date: str) -> None:
    """Entry point for running the solution for the given date.

    For now this just loads the input data and reports how many lines
    were loaded so you can plug in puzzle-specific logic later.
    """
    data = get_data(type_mode, date)
    
    for line in data:
        indicator_str, button_tuples, joltage_list = parse_machine_line(line)
        print(f"Indicator: {indicator_str}, Buttons: {button_tuples}, Joltage: {joltage_list}")

    line = data[0]
    indicator_str, button_tuples, joltage_list = parse_machine_line(line)
    state = indicator_str
    print(f"Initial state: {state}")
    for buttons in button_tuples:
        state = toggle_lights(state, buttons)
        print(f"After pressing {buttons}: {state}")    
    print(f"Final state: {state}")
    tot_sum = 0
    for line in data:
        indicator_str, button_tuples, joltage_list = parse_machine_line(line)    
        steps = bfs_shortest_path(indicator_str, button_tuples)
        print(f"Minimum button presses to turn off all lights for indicator {indicator_str}: {steps}")
        tot_sum += steps
    print(f"Total sum of minimum button presses: {tot_sum}")

if __name__ == "__main__":
    date = "dec10"
    #type_mode = "test"
    type_mode = "data"
    #runa(type_mode, date)
    runb(type_mode, date)