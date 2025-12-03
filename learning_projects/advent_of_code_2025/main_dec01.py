import utils.common_utils as common_utils


def get_data(type_mode: str, date: str) -> list[str]:
    """Return the raw input lines for the given Advent of Code date.

    This is a thin, typed wrapper around common_utils.get_data_list.
    """
    data = common_utils.get_data_list(type_mode, date)
    return data

class NumDial():
    def __init__(self, pos: int = 0) -> None:
        self.pos = pos

    def get_next(self, input: str):
        if input[0] == 'L':
            self.pos -= int(input[1:])
            self.pos %= 100
        elif input[0] == 'R':
            self.pos += int(input[1:])
            self.pos %= 100

    def get_next_pos_with_passing_zero(self, input: str) -> int:
        start_pos = self.pos
        if input[0] == 'L':
            if start_pos == 0:
                start_pos = 100
            end_pos = start_pos - int(input[1:])
            number_of_passings = abs((end_pos -1) // 100)
            self.pos = end_pos % 100
        elif input[0] == 'R':
            end_pos = self.pos + int(input[1:])
            number_of_passings = (end_pos)  // 100
            self.pos = end_pos % 100
        return number_of_passings


    def get_pos(self) -> int:
        return self.pos

def runa(type_mode: str, date: str) -> None:
    """Entry point for running the solution for the given date.

    For now this just loads the input data and reports how many lines
    were loaded so you can plug in puzzle-specific logic later.
    """
    count_zeros = 0
    data = get_data(type_mode, date)
    print(f"Loaded {len(data)} lines for {date!r} with mode {type_mode!r}.")
    print(data)
    dial = NumDial(50)
    for line in data:
        dial.get_next(line)
        print(f"Current position: {dial.get_pos()}")
        if dial.get_pos() == 0:
            count_zeros += 1
    print(f"Number of times dial hit 0: {count_zeros}")
    print(f"Final position: {dial.get_pos()}")

def runb(type_mode: str, date: str) -> None:
    """Entry point for running the solution for the given date in part B.

    For now this just loads the input data and reports how many lines
    were loaded so you can plug in puzzle-specific logic later.
    """
    data = get_data(type_mode, date)
    dial = NumDial(50)
    tot_passings = 0
    for line in data:
        num_of_passings = dial.get_next_pos_with_passing_zero(line)
        tot_passings += num_of_passings
        print(f"The dial is rotated {line} ; Current position: {dial.get_pos()}, number of passings: {num_of_passings}")
    print(f"Total number of times dial hit 0: {tot_passings}")


if __name__ == "__main__":
    date = "dec01"
#    type_mode = "test"
    type_mode = "data"
    #runa(type_mode, date)
    runb(type_mode, date)