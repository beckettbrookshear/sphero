"""Grid coverage planner for auto pilot.

The patrol area is a square grid of cells centred on where the robot started.
The planner keeps track of which cells have been visited and which are known
to be blocked (learned from bumps), and picks the nearest unvisited cell it can
still reach without going through a blocked one.
"""

import heapq
import math
from collections import deque
from typing import Callable, Dict, List, Optional, Set, Tuple

Cell = Tuple[int, int]

_NEIGHBOUR_STEPS = ((0, 1), (0, -1), (1, 0), (-1, 0))


class PatrolPlanner:
    def __init__(self, half_cells: int = 3, cell_cm: float = 30.0, visit_radius_cm: float = 12.0) -> None:
        self.half_cells = int(half_cells)
        self.cell_cm = float(cell_cm)
        self.visit_radius_cm = float(visit_radius_cm)
        self.obstacles: Set[Cell] = set()
        self.visited: Set[Cell] = set()
        self.laps = 0

    def in_bounds(self, cell: Cell) -> bool:
        return abs(cell[0]) <= self.half_cells and abs(cell[1]) <= self.half_cells

    def cell_of(self, x: float, y: float) -> Cell:
        # Positions outside the arena map to the nearest edge cell.
        i = int(math.floor(x / self.cell_cm + 0.5))
        j = int(math.floor(y / self.cell_cm + 0.5))
        limit = self.half_cells
        return (max(-limit, min(limit, i)), max(-limit, min(limit, j)))

    def center(self, cell: Cell) -> Tuple[float, float]:
        return (cell[0] * self.cell_cm, cell[1] * self.cell_cm)

    @property
    def half_extent_cm(self) -> float:
        return (self.half_cells + 0.5) * self.cell_cm

    def mark_obstacle(self, x: float, y: float) -> Optional[Cell]:
        """Record that something blocks the cell containing (x, y)."""
        if abs(x) > self.half_extent_cm or abs(y) > self.half_extent_cm:
            return None
        cell = self.cell_of(x, y)
        self.obstacles.add(cell)
        self.visited.discard(cell)
        return cell

    def walls(self, min_cells: int = 3) -> List[Dict[str, float]]:
        """Straight runs of at least `min_cells` neighbouring blocked cells, as segments in cm.

        A single blocked cell is an object; a long straight line of them is a wall.
        """
        found: List[Dict[str, float]] = []
        for along_x in (True, False):
            lines: Dict[int, List[int]] = {}
            for cell in self.obstacles:
                key, value = (cell[1], cell[0]) if along_x else (cell[0], cell[1])
                lines.setdefault(key, []).append(value)
            for key, values in lines.items():
                values.sort()
                run = [values[0]]
                for value in values[1:] + [None]:
                    if value is not None and value == run[-1] + 1:
                        run.append(value)
                        continue
                    if len(run) >= min_cells:
                        a, b = run[0] * self.cell_cm, run[-1] * self.cell_cm
                        fixed = key * self.cell_cm
                        found.append(
                            {"x1": a, "y1": fixed, "x2": b, "y2": fixed}
                            if along_x
                            else {"x1": fixed, "y1": a, "x2": fixed, "y2": b}
                        )
                    if value is not None:
                        run = [value]
        return found

    def note_position(self, x: float, y: float) -> None:
        """Mark the cell as visited once the robot is close to its centre."""
        cell = self.cell_of(x, y)
        cx, cy = self.center(cell)
        if math.hypot(x - cx, y - cy) <= self.visit_radius_cm and cell not in self.obstacles:
            self.visited.add(cell)

    def _search(self, start: Cell, prefer: Optional[Cell] = None) -> Dict[Cell, Optional[Cell]]:
        # Breadth-first, so insertion order is nearest-first. The start cell may
        # itself be marked blocked (the robot is standing in it) and is still allowed.
        # `prefer` is a (dx, dy) step tried first, so ties favour going straight on.
        steps = _NEIGHBOUR_STEPS
        if prefer in steps:
            steps = (prefer,) + tuple(s for s in steps if s != prefer)
        parents: Dict[Cell, Optional[Cell]] = {start: None}
        queue = deque([start])
        while queue:
            cell = queue.popleft()
            for step in steps:
                neighbour = (cell[0] + step[0], cell[1] + step[1])
                if neighbour in parents or not self.in_bounds(neighbour) or neighbour in self.obstacles:
                    continue
                parents[neighbour] = cell
                queue.append(neighbour)
        return parents

    def near_obstacle(self, cell: Cell, radius: int = 1) -> bool:
        """True when a known blocked cell is within `radius` cells (sides or corners)."""
        span = range(-radius, radius + 1)
        return any(
            (cell[0] + dx, cell[1] + dy) in self.obstacles
            for dx in span
            for dy in span
            if dx or dy
        )

    def next_goal(self, current: Cell, prefer: Optional[Cell] = None) -> Optional[Cell]:
        """Nearest reachable unvisited cell; starts a new lap when everything is done.

        Cells next to known objects are only chosen once nothing else is left.
        """
        goal = self._nearest_unvisited(current, prefer)
        if goal is None and self.visited:
            self.visited = set()
            self.laps += 1
            goal = self._nearest_unvisited(current, prefer)
        return goal

    def _nearest_unvisited(self, current: Cell, prefer: Optional[Cell] = None) -> Optional[Cell]:
        fallback: Optional[Cell] = None
        for cell in self._search(current, prefer):
            if cell in self.visited or cell in self.obstacles:
                continue
            if not self.near_obstacle(cell):
                return cell
            if fallback is None:
                fallback = cell
        return fallback

    def _cheapest_search(self, start: Cell, cost: Callable[[Cell], float]) -> Dict[Cell, Optional[Cell]]:
        # Dijkstra: same reachable cells as _search, but prefers cheaper cells over fewer cells.
        parents: Dict[Cell, Optional[Cell]] = {start: None}
        best: Dict[Cell, float] = {start: 0.0}
        heap: List[Tuple[float, int, Cell]] = [(0.0, 0, start)]
        counter = 0
        done: Set[Cell] = set()
        while heap:
            total, _, cell = heapq.heappop(heap)
            if cell in done:
                continue
            done.add(cell)
            for step in _NEIGHBOUR_STEPS:
                neighbour = (cell[0] + step[0], cell[1] + step[1])
                if neighbour in done or not self.in_bounds(neighbour) or neighbour in self.obstacles:
                    continue
                new_total = total + max(0.01, cost(neighbour))
                if new_total < best.get(neighbour, math.inf):
                    best[neighbour] = new_total
                    parents[neighbour] = cell
                    counter += 1
                    heapq.heappush(heap, (new_total, counter, neighbour))
        return parents

    def path(self, start: Cell, goal: Cell, cost: Optional[Callable[[Cell], float]] = None) -> Optional[List[Cell]]:
        """Cells to pass through to get from start to goal (excluding start).

        Without `cost` the route has the fewest cells; with it, the lowest total cost.
        """
        parents = self._search(start) if cost is None else self._cheapest_search(start, cost)
        if goal not in parents:
            return None
        steps: List[Cell] = []
        cell: Optional[Cell] = goal
        while cell is not None and cell != start:
            steps.append(cell)
            cell = parents[cell]
        steps.reverse()
        return steps
