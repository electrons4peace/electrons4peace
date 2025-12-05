import utils.common_utils as common_utils


def get_data(type_mode: str, date: str) -> list[str]:
    """Return the raw input lines for the given Advent of Code date.

    This is a thin, typed wrapper around common_utils.get_data_list.
    """
    data = common_utils.get_data_list(type_mode, date)
    return data

def runa(type_mode: str, date: str) -> None:
    """Entry point for running the solution for the given date.

    For now this just loads the input data and reports how many lines
    were loaded so you can plug in puzzle-specific logic later.
    """

if __name__ == "__main__":
    date = "dec00"
    type_mode = "test"
    #type_mode = "data"
    runa(type_mode, date)
    #runb(type_mode, date)