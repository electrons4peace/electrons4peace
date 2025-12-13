"""
--- Day 12: Christmas Tree Farm ---
You're almost out of time, but there can't be much left to decorate. Although there are no stairs, elevators, escalators, tunnels, chutes, teleporters, firepoles, or conduits here that would take you deeper into the North Pole base, there is a ventilation duct. You jump in.

After bumping around for a few minutes, you emerge into a large, well-lit cavern full of Christmas trees!

There are a few Elves here frantically decorating before the deadline. They think they'll be able to finish most of the work, but the one thing they're worried about is the presents for all the young Elves that live here at the North Pole. It's an ancient tradition to put the presents under the trees, but the Elves are worried they won't fit.

The presents come in a few standard but very weird shapes. The shapes and the regions into which they need to fit are all measured in standard units. To be aesthetically pleasing, the presents need to be placed into the regions in a way that follows a standardized two-dimensional unit grid; you also can't stack presents.

As always, the Elves have a summary of the situation (your puzzle input) for you. First, it contains a list of the presents' shapes. Second, it contains the size of the region under each tree and a list of the number of presents of each shape that need to fit into that region. For example:

0:
###
##.
##.

1:
###
##.
.##

2:
.##
###
##.

3:
##.
###
##.

4:
###
#..
###

5:
###
.#.
###

4x4: 0 0 0 0 2 0
12x5: 1 0 1 0 2 2
12x5: 1 0 1 0 3 2
The first section lists the standard present shapes. For convenience, each shape starts with its index and a colon; then, the shape is displayed visually, where # is part of the shape and . is not.

The second section lists the regions under the trees. Each line starts with the width and length of the region; 12x5 means the region is 12 units wide and 5 units long. The rest of the line describes the presents that need to fit into that region by listing the quantity of each shape of present; 1 0 1 0 3 2 means you need to fit one present with shape index 0, no presents with shape index 1, one present with shape index 2, no presents with shape index 3, three presents with shape index 4, and two presents with shape index 5.

Presents can be rotated and flipped as necessary to make them fit in the available space, but they have to always be placed perfectly on the grid. Shapes can't overlap (that is, the # part from two different presents can't go in the same place on the grid), but they can fit together (that is, the . part in a present's shape's diagram does not block another present from occupying that space on the grid).

The Elves need to know how many of the regions can fit the presents listed. In the above example, there are six unique present shapes and three regions that need checking.

The first region is 4x4:

....
....
....
....
In it, you need to determine whether you could fit two presents that have shape index 4:

###
#..
###
After some experimentation, it turns out that you can fit both presents in this region. Here is one way to do it, using A to represent one present and B to represent the other:

AAA.
ABAB
ABAB
.BBB
The second region, 12x5: 1 0 1 0 2 2, is 12 units wide and 5 units long. In that region, you need to try to fit one present with shape index 0, one present with shape index 2, two presents with shape index 4, and two presents with shape index 5.

It turns out that these presents can all fit in this region. Here is one way to do it, again using different capital letters to represent all the required presents:

....AAAFFE.E
.BBBAAFFFEEE
DDDBAAFFCECE
DBBB....CCC.
DDD.....C.C.
The third region, 12x5: 1 0 1 0 3 2, is the same size as the previous region; the only difference is that this region needs to fit one additional present with shape index 4. Unfortunately, no matter how hard you try, there is no way to fit all of the presents into this region.

So, in this example, 2 regions can fit all of their listed presents.

Consider the regions beneath each tree and the presents the Elves would like to fit into each of them. How many of the regions can fit all of the presents listed?
"""

from dataclasses import dataclass
import re

import pysat
import z3

from . import challenge, example

from .utils.dancing_links import DancingLinks
from .utils.polyomino import Coord, Polyomino


@dataclass
class Region:
    """A rectangular region to pack polyominoes into."""

    width: int
    height: int
    required_pieces: dict[int, int]  # shape_id -> count

    @property
    def area(self) -> int:
        return self.width * self.height


@dataclass(frozen=True, slots=True)
class Placement:
    """A specific placement of a polyomino on the grid."""

    polyomino: Polyomino
    position: Coord  # Top-left offset applied

    @property
    def occupied_cells(self) -> frozenset[Coord]:
        return self.polyomino.cells


class PolyominoPacker:
    """Solves polyomino packing problems using Dancing Links."""

    def __init__(self, shapes: dict[int, Polyomino]) -> None:
        self.shapes = shapes
        # Precompute all orientations for each shape
        self.orientations: dict[int, set[Polyomino]] = {
            shape_id: shape.all_orientations() for shape_id, shape in shapes.items()
        }

    def _generate_placements(
        self, piece: Polyomino, width: int, height: int
    ) -> Iterator[Placement]:
        """Generate all valid placements of a piece in a region."""
        for orientation in self.orientations[piece.shape_id]:
            oriented_piece = orientation.with_instance_id(piece.instance_id)
            for x in range(width - oriented_piece.width + 1):
                for y in range(height - oriented_piece.height + 1):
                    offset = Coord(x, y)
                    translated = oriented_piece.translate(offset)
                    yield Placement(polyomino=translated, position=offset)

    def can_pack(self, region: Region) -> bool:
        """Check if all required pieces can fit in the region."""
        # Quick area check
        total_piece_area = sum(
            self.shapes[shape_id].area * count
            for shape_id, count in region.required_pieces.items()
        )
        if total_piece_area > region.area:
            return False

        # Build DLX matrix
        dlx: DancingLinks[Placement] = DancingLinks()

        # Add cell columns (primary - each cell covered at most once)
        # But we use secondary columns for cells since not all need to be filled
        for x in range(region.width):
            for y in range(region.height):
                dlx.add_column(f"cell_{x}_{y}", primary=False)

        # Add piece columns (primary - each piece instance must be placed exactly once)
        pieces_to_place: list[Polyomino] = []
        for shape_id, count in region.required_pieces.items():
            if shape_id not in self.shapes:
                continue
            base_shape = self.shapes[shape_id]
            for instance in range(count):
                piece = base_shape.with_instance_id(instance)
                pieces_to_place.append(piece)
                dlx.add_column(f"piece_{shape_id}_{instance}", primary=True)

        # Add rows for each valid placement
        for piece in pieces_to_place:
            for placement in self._generate_placements(
                piece, region.width, region.height
            ):
                columns = [f"piece_{piece.shape_id}_{piece.instance_id}"]
                columns.extend(
                    f"cell_{c.x}_{c.y}" for c in placement.occupied_cells
                )
                dlx.add_row(columns, placement)

        return dlx.has_solution()

    def find_packing(self, region: Region) -> list[Placement] | None:
        """Find a valid packing if one exists."""
        total_piece_area = sum(
            self.shapes[shape_id].area * count
            for shape_id, count in region.required_pieces.items()
        )
        if total_piece_area > region.area:
            return None

        dlx: DancingLinks[Placement] = DancingLinks()

        for x in range(region.width):
            for y in range(region.height):
                dlx.add_column(f"cell_{x}_{y}", primary=False)

        pieces_to_place: list[Polyomino] = []
        for shape_id, count in region.required_pieces.items():
            if shape_id not in self.shapes:
                continue
            base_shape = self.shapes[shape_id]
            for instance in range(count):
                piece = base_shape.with_instance_id(instance)
                pieces_to_place.append(piece)
                dlx.add_column(f"piece_{shape_id}_{instance}", primary=True)

        for piece in pieces_to_place:
            for placement in self._generate_placements(
                piece, region.width, region.height
            ):
                columns = [f"piece_{piece.shape_id}_{piece.instance_id}"]
                columns.extend(
                    f"cell_{c.x}_{c.y}" for c in placement.occupied_cells
                )
                dlx.add_row(columns, placement)

        return next(dlx.solve(), None)


def parse_input(lines: list[str]) -> tuple[dict[int, Polyomino], list[Region]]:
    """Parse the puzzle input into shapes and regions."""
    shapes: dict[int, Polyomino] = {}
    regions: list[Region] = []

    # Split into shape definitions and region definitions
    i = 0

    # Parse shapes
    while i < len(lines):
        line = lines[i].strip()

        # Check for region line (WxH: ...)
        if re.match(r"^\d+x\d+:", line):
            break

        # Check for shape header (N:)
        if match := re.match(r"^(\d+):$", line):
            shape_id = int(match.group(1))
            i += 1
            pattern_lines: list[str] = []

            while i < len(lines) and lines[i].strip() and not re.match(r"^\d+:", lines[i]):
                if re.match(r"^\d+x\d+:", lines[i].strip()):
                    break
                pattern_lines.append(lines[i])
                i += 1

            if pattern_lines:
                pattern = "\n".join(pattern_lines)
                shapes[shape_id] = Polyomino.from_pattern(pattern, shape_id)
        else:
            i += 1

    # Parse regions
    while i < len(lines):
        line = lines[i].strip()
        if match := re.match(r"^(\d+)x(\d+):\s*(.+)$", line):
            width = int(match.group(1))
            height = int(match.group(2))
            counts = list(map(int, match.group(3).split()))

            required: dict[int, int] = {}
            for shape_id, count in enumerate(counts):
                if count > 0:
                    required[shape_id] = count

            regions.append(Region(width=width, height=height, required_pieces=required))
        i += 1

    return shapes, regions


from pysat.solvers import Solver
from pysat.card import CardEnc, EncType
from pysat.formula import CNF, IDPool


class SATPacker:
    def __init__(self, shapes: dict[int, Polyomino], verbose: bool = False) -> None:
        self.shapes = shapes
        self.shape_areas = {sid: s.area for sid, s in shapes.items()}
        self.verbose = verbose

        self.orientations: dict[int, list[frozenset[Coord]]] = {}
        for shape_id, shape in shapes.items():
            self.orientations[shape_id] = [o.cells for o in shape.all_orientations()]

    def can_pack(self, region: Region) -> bool:
        # Quick area check
        total_area = sum(
            self.shape_areas[sid] * cnt
            for sid, cnt in region.required_pieces.items()
            if sid in self.shapes
        )
        if total_area > region.area:
            return False

        if all(c == 0 for c in region.required_pieces.values()):
            return True

        # Generate all placements: (shape_id, cells as tuple of flat indices)
        placements: list[tuple[int, tuple[int, ...]]] = []

        previous_shapes: set[frozenset[Coord]] = set()

        for shape_id, cnt in region.required_pieces.items():
            if cnt <= 0 or shape_id not in self.orientations:
                continue

            for cells in self.orientations[shape_id]:
                if cells in previous_shapes:
                    if self.verbose:
                        print(f"Pruned orientation for shape {shape_id}")
                    continue
                previous_shapes.add(cells)

                pw = max(c.x for c in cells) + 1
                ph = max(c.y for c in cells) + 1

                for py in range(region.height - ph + 1):
                    for px in range(region.width - pw + 1):
                        flat = tuple((c.y + py) * region.width + (c.x + px) for c in cells)
                        placements.append((shape_id, flat))

        if self.verbose:
            print(f"    {len(placements)} placements")

        pool = IDPool()
        cnf = CNF()

        placement_vars = [pool.id(('p', i)) for i in range(len(placements))]

        # Group by shape
        placements_by_shape: dict[int, list[int]] = {sid: [] for sid, cnt in region.required_pieces.items() if cnt > 0}
        for i, (sid, _) in enumerate(placements):
            placements_by_shape[sid].append(placement_vars[i])

        # Exactly N of each shape
        for shape_id, count in region.required_pieces.items():
            if count <= 0:
                continue
            shape_vars = placements_by_shape[shape_id]
            if len(shape_vars) < count:
                return False
            clauses = CardEnc.equals(
                lits=shape_vars, bound=count, vpool=pool, encoding=EncType.seqcounter
            )
            cnf.extend(clauses.clauses)

        # At most one per cell
        cell_placements: dict[int, list[int]] = {}
        for i, (_, cells) in enumerate(placements):
            for c in cells:
                if c not in cell_placements:
                    cell_placements[c] = []
                cell_placements[c].append(placement_vars[i])

        for vars_covering in cell_placements.values():
            if len(vars_covering) > 1:
                clauses = CardEnc.atmost(
                    lits=vars_covering, bound=1, vpool=pool, encoding=EncType.seqcounter
                )
                cnf.extend(clauses.clauses)

        with Solver(name='cadical153', bootstrap_with=cnf) as solver:
            return solver.solve()


from rich.progress import track


@example("""\
0:
###
##.
##.

1:
###
##.
.##

2:
.##
###
##.

3:
##.
###
##.

4:
###
#..
###

5:
###
.#.
###

4x4: 0 0 0 0 2 0
12x5: 1 0 1 0 2 2
12x5: 1 0 1 0 3 2
""", result=2)
@challenge(day=12)
def num_regions(lines: list[str]) -> int:
    shapes, regions = parse_input(lines)
    # packer = PolyominoPacker(shapes)
    packer = SATPacker(shapes)

    # packable_count = 0
    # for i, region in enumerate(track(regions, description="regions")):
    #     if packer.can_pack(region):
    #         packable_count += 1
    #         print(f"Region {i + 1} ({region.width}x{region.height}): ✓ CAN pack")
    #     else:
    #         print(f"Region {i + 1} ({region.width}x{region.height}): ✗ cannot pack")
    #
    # return packable_count

    return sum(
        1
        for region in track(regions, description="regions")
        if packer.can_pack(region)
    )